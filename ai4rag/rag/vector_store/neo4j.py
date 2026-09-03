# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import asyncio
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from json_repair import repair_json

import neo4j
from docling_core.types.doc import DoclingDocument

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.utils import iter_unique_chunks, resolve_embedding_dimension

__all__ = ["Neo4jGraphStore"]

# Shared vector index for KG-pipeline-created Chunk nodes (used by graph search).
# Named without a collection prefix because SimpleKGPipeline creates plain :Chunk
# nodes; collection scoping is done via the node's ``collection`` property instead.
_KG_CHUNK_INDEX = "Chunk__embedding"


try:
    from neo4j_graphrag.embeddings.base import Embedder as _NeoEmbedder
    from neo4j_graphrag.llm.base import LLMInterface as _NeoLLMInterface

    _neo4j_graphrag_bases_available = True
except ImportError:
    _neo4j_graphrag_bases_available = False
    _NeoEmbedder = object  # type: ignore[assignment,misc]
    _NeoLLMInterface = object  # type: ignore[assignment,misc]


class _EmbedderAdapter(_NeoEmbedder):  # type: ignore[misc]
    """Wraps :class:`BaseEmbeddingModel` to satisfy ``neo4j_graphrag``'s embedder interface."""

    def __init__(self, model: BaseEmbeddingModel) -> None:
        self._model = model

    def embed_query(self, text: str, **kwargs) -> list[float]:
        return self._model.embed_query(text)


class _LLMAdapter(_NeoLLMInterface):  # type: ignore[misc]
    """Wraps :class:`OpenAIFoundationModel` to satisfy ``neo4j_graphrag``'s LLM interface.

    ``SimpleKGPipeline`` calls ``ainvoke`` (async); we bridge the sync
    :meth:`OpenAIFoundationModel.chat` via ``run_in_executor``.  Response JSON
    is normalised so that models returning a JSON *array* (``[{...}]``) are
    converted to the expected object format (``{"nodes": [...], "relationships": [...]}``)
    before the extractor parses the output.
    """

    def __init__(self, model: OpenAIFoundationModel) -> None:
        super().__init__(model_name=model.model_id)
        self._model = model

    def invoke(
        self,
        input: str,
        message_history=None,
        system_instruction: str | None = None,
    ):
        from neo4j_graphrag.llm.types import LLMResponse

        messages: list[dict] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": input})
        choices = self._model.chat(messages)
        content = choices[0].message.content or ""
        return LLMResponse(content=_normalize_kg_json(content))

    async def ainvoke(
        self,
        input: str,
        message_history=None,
        system_instruction: str | None = None,
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input, message_history, system_instruction)


class Neo4jGraphStore(BaseVectorStore):
    """Vector store backed by Neo4j with optional graph traversal search.

    Supports two independent workflows:

    **Vector workflow** — ``add_documents`` + ``search(mode="vector")``
        Stores chunks as collection-scoped ``{collection_name}:Chunk`` nodes,
        creates a per-collection vector index, and performs dense ANN retrieval.

    **Graph RAG workflow** — ``build_knowledge_graph_from_documents`` + ``search(mode="graph")``
        Uses :class:`neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline`
        to chunk documents, embed them, extract entities/relations with an LLM, and
        write the resulting knowledge graph to Neo4j.  Vector seeds are retrieved
        from the shared ``Chunk__embedding`` index; context is expanded via
        ``__Entity__`` → ``FROM_CHUNK`` traversal and ``NEXT_CHUNK`` sequential links.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    config : Neo4jConfig
        Connection parameters for the Neo4j instance.
    distance_metric : str, default="cosine"
        Distance metric used for the vector index.
    collection_name : str | None, default=None
        Existing collection to reuse; must start with the ``ai4rag`` prefix.
        When omitted, a new compliant name is generated.
    """

    _BATCH_SIZE = 512

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: Neo4jConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
        foundation_model: Any = None,
    ):
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(embedding_model)
        self._foundation_model = foundation_model
        self._driver = neo4j.GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        self._driver.verify_connectivity()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the collection-scoped vector + fulltext indexes if absent."""
        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"CREATE VECTOR INDEX `{self._collection_name}__vector` IF NOT EXISTS "
                f"FOR (n:{self._collection_name}) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}}}",
                dim=self._embedding_dimension,
            )
        logger.info("Neo4j schema ready: %s (dim=%d)", self._collection_name, self._embedding_dimension)

    def _ensure_kg_schema(self) -> None:
        """Create the shared KG vector index on plain ``Chunk`` nodes if absent."""
        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"CREATE VECTOR INDEX `{_KG_CHUNK_INDEX}` IF NOT EXISTS "
                f"FOR (n:Chunk) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}}}",
                dim=self._embedding_dimension,
            )

    # ------------------------------------------------------------------
    # Vector workflow
    # ------------------------------------------------------------------

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
        """Embed, deduplicate, and upsert chunks into Neo4j.

        Creates ``Chunk`` and ``Document`` nodes (with the collection label),
        ``CONTAINS`` provenance links, and ``NEXT_CHUNK`` sequential links.
        Also ensures the shared ``Chunk__embedding`` index exists and sets the
        ``collection`` property on each chunk so that ``search(mode="graph")``
        can scope results to this collection without requiring
        :meth:`build_knowledge_graph_from_documents`.

        When *model* is provided via kwargs, entity extraction is performed after
        indexing: chunks are sent in batches to the LLM, and the resulting
        ``__Entity__`` nodes are written to Neo4j with ``FROM_CHUNK``
        relationships so that ``search(mode="graph")`` can expand context via the
        knowledge graph.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and stored.
        **kwargs : Any
            Optional overrides:

            - ``batch_size`` (int) — max chunks per write transaction
              (default :attr:`_BATCH_SIZE`).
            - ``model`` — foundation model used for entity extraction.  When
              omitted, graph search falls back to sequential ``NEXT_CHUNK``
              expansion only.
        """
        if not documents:
            return

        self._ensure_kg_schema()

        embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])
        unique_pairs = list(iter_unique_chunks(documents, embeddings))

        doc_groups: dict[str, list[tuple[AI4RAGChunk, list[float]]]] = {}
        for doc, emb in unique_pairs:
            doc_id = doc.metadata.get("document_id", doc.chunk_id)
            doc_groups.setdefault(doc_id, []).append((doc, emb))

        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda p: p[0].metadata.get("sequence_number", 0))

        batch_size = kwargs.get("batch_size", self._BATCH_SIZE)
        pending: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]] = []
        pending_count = 0

        for doc_id, sorted_pairs in doc_groups.items():
            pending.append((doc_id, sorted_pairs))
            pending_count += len(sorted_pairs)
            if pending_count >= batch_size:
                self._upsert_doc_groups(pending)
                pending = []
                pending_count = 0

        if pending:
            self._upsert_doc_groups(pending)

        if self._foundation_model is not None:
            self._run_kg_pipeline(
                texts=[chunk.text for chunk, _ in unique_pairs],
                model=self._foundation_model,
            )

    def _run_kg_pipeline(self, texts: list[str], model: Any, max_concurrent: int = 8) -> None:
        """Run ``SimpleKGPipeline`` on chunk texts concurrently.

        Explicit entity/relation types are provided so the pipeline uses
        ``SchemaBuilder`` instead of ``SchemaFromTextExtractor`` — the latter
        sends the entire input text in a single LLM call, which exceeds
        context limits for large corpora.

        Chunks are processed concurrently (bounded by ``max_concurrent``) inside
        a single asyncio event loop, giving near-linear speedup over the
        sequential approach since LLM calls are I/O-bound.
        """
        from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

        texts = [t for t in texts if t.strip()]
        if not texts:
            return

        pipeline = SimpleKGPipeline(
            llm=_LLMAdapter(model),
            driver=self._driver,
            embedder=_EmbedderAdapter(self.embedding_model),
            entities=["Person", "Organization", "Place", "Concept", "Event", "Product", "Technology"],
            relations=["RELATED_TO", "PART_OF", "LOCATED_IN", "BELONGS_TO", "CREATED_BY", "MENTIONS"],
            from_pdf=False,
            text_splitter=FixedSizeSplitter(chunk_size=2000, chunk_overlap=200),
            on_error="IGNORE",
            perform_entity_resolution=True,
            neo4j_database=self._config.database,
        )

        async def _run_all() -> None:
            sem = asyncio.Semaphore(max_concurrent)

            async def _run_one(text: str) -> None:
                async with sem:
                    await pipeline.run_async(text=text)

            await asyncio.gather(*(_run_one(t) for t in texts))

        try:
            asyncio.run(_run_all())
        except RuntimeError:
            raise RuntimeError(
                "_run_kg_pipeline cannot be called from within a running event loop. "
                "Install 'nest_asyncio' and call nest_asyncio.apply() beforehand."
            )

        with self._driver.session(database=self._config.database) as session:
            session.run(
                "MATCH (c:Chunk) WHERE c.collection IS NULL SET c.collection = $col",
                col=self._collection_name,
            )

    def _upsert_doc_groups(
        self, doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]]
    ) -> None:
        with self._driver.session(database=self._config.database) as session:
            session.execute_write(self._upsert_batch_tx, doc_groups, self._collection_name)

    @staticmethod
    def _upsert_batch_tx(
        tx: neo4j.Transaction,
        doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]],
        collection_name: str,
    ) -> None:
        for doc_id, sorted_pairs in doc_groups:
            source = sorted_pairs[0][0].metadata.get("source", "")
            tx.run(
                f"MERGE (d:{collection_name}:Document {{id: $doc_id}}) "
                f"SET d.source = $source, d.metadata = $doc_metadata",
                doc_id=doc_id,
                source=source,
                doc_metadata=json.dumps({"source": source}),
            )

            for chunk, embedding in sorted_pairs:
                tx.run(
                    f"MERGE (c:{collection_name}:Chunk {{id: $id}}) "
                    f"SET c.text = $text, c.embedding = $embedding, "
                    f"c.document_id = $document_id, c.sequence_number = $sequence_number, "
                    f"c.metadata = $metadata, c.collection = $collection",
                    id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    document_id=doc_id,
                    sequence_number=chunk.metadata.get("sequence_number", 0),
                    metadata=json.dumps(chunk.metadata),
                    collection=collection_name,
                )
                tx.run(
                    f"MATCH (d:{collection_name}:Document {{id: $doc_id}}) "
                    f"MATCH (c:{collection_name}:Chunk {{id: $chunk_id}}) "
                    f"MERGE (d)-[:CONTAINS]->(c)",
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                )

            for i in range(len(sorted_pairs) - 1):
                chunk_a = sorted_pairs[i][0]
                chunk_b = sorted_pairs[i + 1][0]
                tx.run(
                    f"MATCH (a:{collection_name}:Chunk {{id: $id_a}}) "
                    f"MATCH (b:{collection_name}:Chunk {{id: $id_b}}) "
                    f"MERGE (a)-[:NEXT_CHUNK]->(b)",
                    id_a=chunk_a.chunk_id,
                    id_b=chunk_b.chunk_id,
                )

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        search_mode: str = "vector",
        **kwargs,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Search for chunks relevant to *query*.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.
        include_scores : bool, default=False
            Whether to include similarity scores in the return value.
        search_mode : str, default="vector"
            ``"vector"`` — pure ANN retrieval via :class:`neo4j_graphrag.retrievers.VectorRetriever`
            against the collection-scoped vector index (requires :meth:`add_documents`).

            ``"graph"`` — ANN seed retrieval + entity/sequential context expansion via
            :class:`neo4j_graphrag.retrievers.VectorCypherRetriever` against the shared
            ``Chunk__embedding`` index (requires :meth:`build_knowledge_graph_from_documents`).
        **kwargs : Any
            Graph-mode parameters:

            - ``graph_hops`` (int, default 1) — ``NEXT_CHUNK`` traversal depth.
            - ``include_entity_neighbors`` (bool, default True) — expand via ``__Entity__``.
            - ``entity_neighbor_limit`` (int, default 5) — max entity-linked neighbors per seed.
        """
        _validate_neo4j_search_params(search_mode, **kwargs)

        if search_mode == "graph":
            return self._search_graph(query, k, include_scores, **kwargs)
        return self._search_vector(query, k, include_scores)

    def _search_vector(
        self, query: str, k: int, include_scores: bool
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        from neo4j_graphrag.retrievers import VectorRetriever
        from neo4j_graphrag.types import RetrieverResultItem

        def _fmt(record) -> RetrieverResultItem:
            node = record.get("node")
            text = node.get("text", "") if node else ""
            raw_meta = node.get("metadata") if node else None
            if isinstance(raw_meta, str) and raw_meta:
                try:
                    metadata = json.loads(raw_meta)
                except Exception:
                    metadata = {}
            else:
                metadata = raw_meta or {}
            return RetrieverResultItem(
                content=text,
                metadata={"score": record.get("score", 0.0), "_meta": metadata},
            )

        retriever = VectorRetriever(
            driver=self._driver,
            index_name=f"{self._collection_name}__vector",
            embedder=_EmbedderAdapter(self.embedding_model),
            result_formatter=_fmt,
            neo4j_database=self._config.database,
        )
        result = retriever.search(query_text=query, top_k=k)

        pairs = [
            (
                AI4RAGChunk(text=item.content, metadata=item.metadata.get("_meta", {})),
                float(item.metadata.get("score", 0.0)),
            )
            for item in result.items
            if item.content
        ]
        if include_scores:
            return pairs
        return [chunk for chunk, _ in pairs]

    def _search_graph(
        self,
        query: str,
        k: int,
        include_scores: bool,
        **kwargs,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        from neo4j_graphrag.retrievers import VectorCypherRetriever
        from neo4j_graphrag.types import RetrieverResultItem

        graph_hops = kwargs.get("graph_hops", 1)
        include_entity_neighbors = kwargs.get("include_entity_neighbors", True)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)

        def _fmt(record) -> RetrieverResultItem:
            raw_meta = record.get("metadata")
            if isinstance(raw_meta, str) and raw_meta:
                try:
                    chunk_meta = json.loads(raw_meta)
                except Exception:
                    chunk_meta = {}
            else:
                chunk_meta = raw_meta or {}
            if "document_id" not in chunk_meta and record.get("document_id"):
                chunk_meta["document_id"] = record.get("document_id")
            return RetrieverResultItem(
                content=record.get("text") or "",
                metadata={"score": float(record.get("score", 0.0)), "_meta": chunk_meta},
            )

        retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name=_KG_CHUNK_INDEX,
            retrieval_query=_build_graph_retrieval_query(
                graph_hops, include_entity_neighbors, entity_neighbor_limit
            ),
            embedder=_EmbedderAdapter(self.embedding_model),
            result_formatter=_fmt,
            neo4j_database=self._config.database,
        )
        result = retriever.search(query_text=query, top_k=k, query_params={"col": self._collection_name})

        pairs = [
            (AI4RAGChunk(text=item.content, metadata=item.metadata.get("_meta", {})), float(item.metadata.get("score", 0.0)))
            for item in result.items
            if item.content
        ]
        if include_scores:
            return pairs
        return [chunk for chunk, _ in pairs]

    # ------------------------------------------------------------------
    # Knowledge graph construction
    # ------------------------------------------------------------------

    def build_knowledge_graph_from_documents(
        self,
        documents: list[DoclingDocument],
        model: OpenAIFoundationModel,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        on_error: str = "IGNORE",
        perform_entity_resolution: bool = True,
    ) -> None:
        """Build a knowledge graph from documents using ``SimpleKGPipeline``.

        Delegates the full pipeline — text splitting, embedding, LLM-based entity
        and relation extraction, and Neo4j write — to
        :class:`neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline`.
        The pipeline creates plain ``Chunk`` nodes (with ``embedding`` and ``text``
        properties), ``__Entity__`` nodes, and ``FROM_CHUNK`` / ``NEXT_CHUNK``
        relationships.  After the pipeline, every new ``Chunk`` node is tagged with
        the ``collection`` property so that :meth:`search` (``mode="graph"``) can
        scope results to this collection.

        .. note::
            This method is independent of :meth:`add_documents`.  For graph-RAG
            workloads, call this method instead of ``add_documents``; for pure
            vector search, use ``add_documents``.

        Parameters
        ----------
        documents : list[DoclingDocument]
            Parsed documents to process.
        model : OpenAIFoundationModel
            Foundation model used for entity and relation extraction.
        chunk_size : int, default=2000
            Target chunk size in characters (passed to ``FixedSizeSplitter``).
        chunk_overlap : int, default=200
            Overlap in characters between consecutive chunks.
        on_error : str, default="IGNORE"
            Error handling strategy passed to ``SimpleKGPipeline``
            (``"IGNORE"`` or ``"RAISE"``).
        perform_entity_resolution : bool, default=True
            Whether to merge duplicate entity nodes after extraction.
        """
        from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

        self._ensure_kg_schema()

        text = "\n\n".join(doc.export_to_markdown() for doc in documents)
        if not text.strip():
            logger.info("No text extracted from documents; skipping KG build.")
            return

        embedder = _EmbedderAdapter(self.embedding_model)
        llm = _LLMAdapter(model)

        pipeline = SimpleKGPipeline(
            llm=llm,
            driver=self._driver,
            embedder=embedder,
            from_pdf=False,
            text_splitter=FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
            on_error=on_error,
            perform_entity_resolution=perform_entity_resolution,
            neo4j_database=self._config.database,
        )

        try:
            asyncio.run(pipeline.run_async(text=text))
        except RuntimeError:
            # Already inside a running event loop (e.g. Jupyter).
            # Users can install nest_asyncio and call nest_asyncio.apply() beforehand.
            raise RuntimeError(
                "build_knowledge_graph_from_documents cannot be called from within a running "
                "event loop.  Install 'nest_asyncio' and call nest_asyncio.apply() before "
                "invoking this method in a Jupyter notebook or other async context."
            )

        # Tag newly-created Chunk nodes with the collection name for graph-search scoping.
        with self._driver.session(database=self._config.database) as session:
            session.run(
                "MATCH (c:Chunk) WHERE c.collection IS NULL SET c.collection = $col",
                col=self._collection_name,
            )

        logger.info(
            "Knowledge graph built from %d documents (collection=%s).",
            len(documents),
            self._collection_name,
        )

    def build_knowledge_graph(
        self,
        model: OpenAIFoundationModel,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
        chunk_batch_size: int = 4,
        max_workers: int = 4,
        max_tokens: int = 2048,
    ) -> None:
        """Extract entities and relations from already-indexed chunks.

        Reads all ``Chunk`` nodes in the collection (paginated), sends them in
        batches to the LLM for extraction, and writes ``Entity`` nodes and
        ``MENTIONS`` / ``RELATED_TO`` relationships back to Neo4j.

        Use :meth:`build_knowledge_graph_from_documents` for the recommended
        ``SimpleKGPipeline``-based workflow.  This method is kept for workloads
        where chunks are already loaded via :meth:`add_documents`.

        Parameters
        ----------
        model : OpenAIFoundationModel
            Foundation model used for chat completions.
        entities : list[str] | None, default=None
            Optional entity-type hints for the extraction prompt.
        relations : list[str] | None, default=None
            Optional relation-type hints for the extraction prompt.
        chunk_batch_size : int, default=16
            Number of chunks sent to the LLM per call.
        max_workers : int, default=4
            Number of parallel LLM threads.
        max_tokens : int, default=512
            Maximum output tokens per LLM call.
        """
        all_chunks = self._read_all_chunks()
        if not all_chunks:
            logger.info("No chunks found in collection %s; skipping KG build.", self._collection_name)
            return

        batches = [all_chunks[i : i + chunk_batch_size] for i in range(0, len(all_chunks), chunk_batch_size)]
        entity_hint = f"\nFocus on entity types: {', '.join(entities)}." if entities else ""
        relation_hint = f"\nFocus on relation types: {', '.join(relations)}." if relations else ""
        system_prompt = _KG_BATCH_SYSTEM_PROMPT + entity_hint + relation_hint

        def extract_batch(batch: list[dict]) -> tuple[list[dict], list[dict]]:
            texts = "\n\n".join(f"[{c['id']}] {c['text']}" for c in batch)
            choices = model.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": texts},
                ],
                max_completion_tokens=max_tokens,
            )
            raw = choices[0].message.content or ""
            return _parse_kg_extraction(raw)

        def write_batch_result(chunk_entities: list[dict], relationships: list[dict]) -> None:
            with self._driver.session(database=self._config.database) as session:
                for ent in chunk_entities:
                    chunk_id = ent.get("chunk_id", "")
                    session.execute_write(
                        Neo4jGraphStore._write_kg_result_tx,
                        chunk_id,
                        [ent],
                        [],
                        self._collection_name,
                    )
                if relationships:
                    session.execute_write(
                        Neo4jGraphStore._write_kg_result_tx,
                        "",
                        [],
                        relationships,
                        self._collection_name,
                    )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                try:
                    batch_entities, batch_rels = future.result()
                    if batch_entities or batch_rels:
                        write_batch_result(batch_entities, batch_rels)
                except Exception as exc:
                    logger.warning("KG extraction batch failed: %s", exc)

        logger.info("Knowledge graph built for collection %s.", self._collection_name)

    def _read_all_chunks(self) -> list[dict]:
        """Return all Chunk nodes in this collection, paginated."""
        all_chunks: list[dict] = []
        skip = 0
        page_size = 1000
        with self._driver.session(database=self._config.database) as session:
            while True:
                page = session.run(
                    f"MATCH (c:{self._collection_name}:Chunk) "
                    f"RETURN c.id AS id, c.text AS text "
                    f"SKIP $skip LIMIT $limit",
                    skip=skip,
                    limit=page_size,
                ).data()
                all_chunks.extend(page)
                if len(page) < page_size:
                    break
                skip += page_size
        return all_chunks

    @staticmethod
    def _write_kg_result_tx(
        tx: neo4j.Transaction,
        chunk_id: str,
        entities: list[dict],
        relationships: list[dict],
        collection_name: str,
    ) -> None:
        for ent in entities:
            name = ent.get("name", "")
            if not name:
                continue
            tx.run(
                f"MERGE (e:{collection_name}:Entity {{name: $name, entity_type: $etype}}) "
                f"SET e.description = $desc",
                name=name,
                etype=ent.get("type", "Other"),
                desc=ent.get("description", ""),
            )
            if chunk_id:
                tx.run(
                    f"MATCH (c:{collection_name}:Chunk {{id: $cid}}) "
                    f"MATCH (e:{collection_name}:Entity {{name: $name, entity_type: $etype}}) "
                    f"MERGE (c)-[:MENTIONS]->(e)",
                    cid=chunk_id,
                    name=name,
                    etype=ent.get("type", "Other"),
                )

        entity_type_map = {e["name"]: e.get("type", "Other") for e in entities if e.get("name")}

        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if not src or not tgt:
                continue
            src_type = entity_type_map.get(src, "Other")
            tgt_type = entity_type_map.get(tgt, "Other")
            tx.run(
                f"MERGE (e1:{collection_name}:Entity {{name: $src, entity_type: $stype}}) "
                f"MERGE (e2:{collection_name}:Entity {{name: $tgt, entity_type: $ttype}}) "
                f"MERGE (e1)-[:RELATED_TO {{keywords: $kw, description: $desc}}]->(e2)",
                src=src,
                stype=src_type,
                tgt=tgt,
                ttype=tgt_type,
                kw=rel.get("keywords", ""),
                desc=rel.get("description", ""),
            )

    def resolve_entities(self) -> int:
        """Merge duplicate Entity nodes that share the same name (case-insensitive).

        Returns
        -------
        int
            Number of duplicate nodes removed.
        """
        with self._driver.session(database=self._config.database) as session:
            rows = session.run(
                f"MATCH (e:{self._collection_name}:Entity) "
                f"RETURN e.name AS name, elementId(e) AS eid "
                f"ORDER BY e.name"
            ).data()

        groups: dict[str, list[str]] = {}
        for row in rows:
            key = (row["name"] or "").lower()
            groups.setdefault(key, []).append(row["eid"])

        removed = 0
        with self._driver.session(database=self._config.database) as session:
            for eids in groups.values():
                if len(eids) < 2:
                    continue
                canonical_eid = eids[0]
                for dup_eid in eids[1:]:
                    session.run(
                        "MATCH (c)-[:MENTIONS]->(dup) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (c)-[:MENTIONS]->(canon)",
                        dup=dup_eid, canon=canonical_eid,
                    )
                    session.run(
                        "MATCH (dup)-[:RELATED_TO]->(target) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (canon)-[:RELATED_TO]->(target)",
                        dup=dup_eid, canon=canonical_eid,
                    )
                    session.run(
                        "MATCH (src)-[:RELATED_TO]->(dup) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (src)-[:RELATED_TO]->(canon)",
                        dup=dup_eid, canon=canonical_eid,
                    )
                    session.run(
                        "MATCH (dup) WHERE elementId(dup) = $dup DETACH DELETE dup",
                        dup=dup_eid,
                    )
                    removed += 1

        logger.info("Entity resolver removed %d duplicate nodes from collection %s.", removed, self._collection_name)
        return removed

    def clean_collection(self) -> None:
        """Drop indexes and delete all nodes belonging to this collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(f"DROP INDEX `{self._collection_name}__vector` IF EXISTS")
            session.run(f"DROP INDEX `{self._collection_name}__fulltext` IF EXISTS")
            session.run(f"MATCH (n:{self._collection_name}) DETACH DELETE n")
            # Clean up KG-pipeline chunks tagged with this collection.
            session.run(
                "MATCH (c:Chunk {collection: $col}) DETACH DELETE c",
                col=self._collection_name,
            )
        logger.info("Collection %s cleaned.", self._collection_name)

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()


# ---------------------------------------------------------------------------
# KG extraction helpers (used by build_knowledge_graph / batch mode)
# ---------------------------------------------------------------------------

_KG_BATCH_SYSTEM_PROMPT = """\
You are a top-tier algorithm designed for extracting information in structured \
formats to build a knowledge graph.

You will receive several text chunks, each prefixed with [chunk_id]. \
Extract ALL entities and ALL relationships from every chunk. \
Do not apply any count limit.

---Entity types---
Person, Organization, Location, Concept, Method, Artifact, Event, Data, Content, Other.

---Rules---
- Retain established capitalisation (e.g. "vLLM", "OpenShift").
- Assign a unique string ID (starting from "0") to each node and reuse that ID \
in relationships.
- Each node must carry a "chunk_id" property set to the ID of the chunk it was \
extracted from (the value inside the brackets).
- description: ONE sentence, max 20 words, third person.
- Do not return anything other than the JSON object below.
- Do not wrap the JSON in backticks or markdown fences.

---Output format---
{"nodes": [{"id": "0", "label": "Person", "properties": {"name": "Alice", "chunk_id": "c1", "description": "..."}}],
 "relationships": [{"type": "WORKS_AT", "start_node_id": "0", "end_node_id": "1", "properties": {"description": "..."}}]}
"""


def _normalize_kg_json(content: str) -> str:
    """Normalise LLM output: convert a JSON array response to the expected object format.

    Some models return ``[{...}]`` instead of ``{"nodes": [...], "relationships": [...]}``
    causing the ``neo4j_graphrag`` extractor to raise a ``TypeError``.  This helper
    repairs and normalises the response before it reaches the extractor.
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", content).strip()
        repaired = repair_json(cleaned, skip_json_loads=False, return_objects=False)
        parsed = json.loads(repaired) if isinstance(repaired, str) else repaired
        if isinstance(parsed, list):
            merged: dict = {"nodes": [], "relationships": []}
            for item in parsed:
                if isinstance(item, dict):
                    merged["nodes"].extend(item.get("nodes") or [])
                    merged["relationships"].extend(item.get("relationships") or [])
            return json.dumps(merged)
        return content
    except Exception:
        return content


def _parse_kg_extraction(raw: str) -> tuple[list[dict], list[dict]]:
    """Parse LLM JSON (node-ID format) into ``(entities, relationships)``."""
    try:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        repaired = repair_json(cleaned, skip_json_loads=False, return_objects=False)
        data = json.loads(repaired) if isinstance(repaired, str) else repaired
        if not isinstance(data, dict):
            return [], []

        id_to_entity: dict[str, dict] = {}
        entities: list[dict] = []
        for node in data.get("nodes", []):
            node_id = str(node.get("id", ""))
            props = node.get("properties", {})
            name = props.get("name", "")
            if not name:
                continue
            entity = {
                "name": name,
                "type": node.get("label", "Other"),
                "description": props.get("description", ""),
                "chunk_id": props.get("chunk_id", ""),
            }
            id_to_entity[node_id] = entity
            entities.append(entity)

        relationships: list[dict] = []
        for rel in data.get("relationships", []):
            src = id_to_entity.get(str(rel.get("start_node_id", "")))
            tgt = id_to_entity.get(str(rel.get("end_node_id", "")))
            if not src or not tgt:
                continue
            props = rel.get("properties", {})
            relationships.append({
                "source": src["name"],
                "target": tgt["name"],
                "keywords": rel.get("type", ""),
                "description": props.get("description", ""),
            })

        return entities, relationships
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Graph retrieval query builder
# ---------------------------------------------------------------------------


def _build_graph_retrieval_query(
    graph_hops: int,
    include_entity_neighbors: bool,
    entity_neighbor_limit: int,
) -> str:
    """Build the Cypher retrieval query for :class:`VectorCypherRetriever`.

    The query receives ``node`` (seed ``Chunk``) and ``score`` from the vector
    index call and expands context via:

    - Entity-linked chunks: ``__Entity__`` → ``FROM_CHUNK`` (SimpleKGPipeline schema).
    - Sequential chunks: ``NEXT_CHUNK*1..N`` in both directions.

    Expanded texts are appended to the seed text and returned as a single
    ``text`` value per seed, which is what ``VectorCypherRetriever`` expects.
    """
    # $col is passed via query_params in _search_graph to scope results to one collection.
    collection_filter = "WHERE node.collection = $col "

    if include_entity_neighbors:
        entity_block = (
            f"OPTIONAL MATCH (entity:__Entity__)-[:FROM_CHUNK]->(node) "
            f"WITH node, score, entity "
            f"OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(ent_nb:Chunk) "
            f"WHERE elementId(ent_nb) <> elementId(node) "
            f"WITH node, score, collect(DISTINCT ent_nb.text)[..{entity_neighbor_limit}] AS ent_texts "
        )
    else:
        entity_block = "WITH node, score, [] AS ent_texts "

    return (
        collection_filter
        + entity_block
        + f"OPTIONAL MATCH (node)-[:NEXT_CHUNK*1..{graph_hops}]->(fwd:Chunk) "
        + f"OPTIONAL MATCH (bwd:Chunk)-[:NEXT_CHUNK*1..{graph_hops}]->(node) "
        + "WITH node, score, ent_texts, "
        + "collect(DISTINCT fwd.text) + collect(DISTINCT bwd.text) AS seq_texts "
        + "WITH node, score, [x IN ent_texts + seq_texts WHERE x IS NOT NULL] AS ctx "
        + "RETURN node.text + reduce(s='', t IN ctx | s + '\\n---\\n' + t) AS text, score, "
        + "node.document_id AS document_id, node.metadata AS metadata"
    )


def _validate_neo4j_search_params(search_mode: str, **kwargs: Any) -> None:
    if search_mode not in ("vector", "graph"):
        raise ValueError(f"Invalid search_mode '{search_mode}'. Must be 'vector' or 'graph'.")

    if search_mode == "graph":
        graph_hops = kwargs.get("graph_hops", 1)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        if not isinstance(graph_hops, int) or graph_hops < 1:
            raise ValueError(f"graph_hops must be a positive integer, got {graph_hops!r}.")
        if not isinstance(entity_neighbor_limit, int) or entity_neighbor_limit < 0:
            raise ValueError(f"entity_neighbor_limit must be a non-negative integer, got {entity_neighbor_limit!r}.")
