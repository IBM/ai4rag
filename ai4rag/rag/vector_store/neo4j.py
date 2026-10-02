# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# pylint: disable=too-many-lines
import asyncio
import hashlib
import json
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from typing import Any

import neo4j
from docling_core.types.doc import DoclingDocument
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j_utils import (
    _KG_LEXICAL_GRAPH_CONFIG,
    Neo4jGraphSchema,
    _build_graph_retrieval_query,
    _CanonicalKGWriter,
    _collection_vector_index_name,
    _EmbedderAdapter,
    _kg_pipeline_extraction_options,
    _LLMAdapter,
    _PreChunkedTextSplitter,
    _validate_kg_extraction_config,
    _validate_neo4j_search_params,
)
from ai4rag.rag.vector_store.utils import (
    iter_unique_chunks,
    resolve_embedding_dimension,
)

__all__ = ["Neo4jGraphRetrievalConfig", "Neo4jGraphStore"]

_SCHEMA = Neo4jGraphSchema


@dataclass(frozen=True, kw_only=True)
class Neo4jGraphRetrievalConfig:
    """Controls graph expansion during `Neo4jGraphStore` retrieval."""

    route_k: int | None = None
    include_entity_neighbors: bool = True
    entity_neighbor_limit: int = 5
    entity_pivot_limit: int = 3
    entity_relationship_hops: int = 1
    relationship_neighbor_limit: int = 5

    def __post_init__(self) -> None:
        if self.route_k is not None and (
            not isinstance(self.route_k, int) or isinstance(self.route_k, bool) or self.route_k < 1
        ):
            raise ValueError(f"route_k must be a positive integer or None, got {self.route_k!r}.")
        if not isinstance(self.include_entity_neighbors, bool):
            raise TypeError("include_entity_neighbors must be a boolean.")
        for name, value in self.to_search_kwargs().items():
            if name in {"route_k", "include_entity_neighbors"}:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "Neo4jGraphRetrievalConfig":
        """Create a configuration from graph-search keyword arguments."""
        unexpected = set(values) - set(cls.keys())
        if unexpected:
            raise ValueError(f"Unsupported Neo4j graph retrieval settings: {sorted(unexpected)}.")
        return cls(**dict(values))

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """Return keyword names accepted by :meth:`to_search_kwargs`."""
        return tuple(item.name for item in fields(cls))

    def to_search_kwargs(self) -> dict[str, int | bool]:
        """Return the configuration in :meth:`Neo4jGraphStore.search` keyword form."""
        return {name: value for name, value in asdict(self).items() if value is not None}


class Neo4jGraphStore(BaseVectorStore):
    """Graph-only vector store backed by Neo4j.

    ``add_documents`` stores one Document and Chunk set per collection. When a
    foundation model is supplied, ``SimpleKGPipeline`` extracts entities and
    relations from those chunks, and the graph writer links them back by chunk
    ID. Graph search uses the collection-specific vector index and expands
    context through ``__Entity__`` → ``FROM_CHUNK`` links.

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
        foundation_model: BaseFoundationModel | None = None,
        kg_extraction_config: dict[str, Any] | None = None,
    ):
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(embedding_model)
        self._foundation_model = foundation_model
        self._kg_extraction_config = _validate_kg_extraction_config(kg_extraction_config)
        self._driver = neo4j.GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        self._driver.verify_connectivity()
        self._ensure_kg_schema()

    def _ensure_kg_schema(self) -> None:
        """Create this collection's chunk vector index."""
        index_name = _collection_vector_index_name(self._collection_name)
        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS "
                # Neo4j vector indexes support one node label. The collection
                # label scopes this index; nodes retain their :Chunk label for
                # graph traversal queries.
                f"FOR (n:`{self._collection_name}`) ON (n.{_SCHEMA.EMBEDDING}) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}}}",
                dim=self._embedding_dimension,
            )

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
        """Embed, deduplicate, and upsert chunks into Neo4j.

        Creates ``Chunk`` and ``Document`` nodes (with the collection label),
        ``CONTAINS`` provenance links, and ``NEXT_CHUNK`` sequential links.
        Also ensures a collection-specific vector index exists and sets the
        ``collection`` property on each chunk so graph traversal can scope
        related nodes to this collection without requiring
        :meth:`build_knowledge_graph_from_documents`.

        When *model* is provided via kwargs, entity extraction is performed after
        indexing: chunks are sent to the LLM, and the resulting
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
            - ``model`` — foundation model used for entity extraction. When
              omitted and no model was supplied at construction, graph search
              returns ANN seed chunks without entity-based context expansion.
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

        for pairs in doc_groups.values():
            pairs.sort(key=lambda p: p[0].metadata.get("sequence_number", 0))

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

        model = kwargs.get("model", self._foundation_model)
        if model is not None:
            self._run_kg_pipeline(
                chunks=[chunk for chunk, _ in unique_pairs],
                model=model,
                on_error=kwargs.get("on_error", "IGNORE"),
                perform_entity_resolution=kwargs.get("perform_entity_resolution", True),
            )

    def _run_kg_pipeline(
        self,
        chunks: list[AI4RAGChunk],
        model: Any,
        max_concurrent: int = 8,
        on_error: str = "IGNORE",
        perform_entity_resolution: bool = True,
    ) -> None:
        """Run ``SimpleKGPipeline`` on chunk texts concurrently.

        Constrained extraction supplies entity/relation types; free extraction
        supplies an empty schema. Both use ``SchemaBuilder`` rather than
        ``SchemaFromTextExtractor``, which sends the entire input text in one
        LLM call and can exceed context limits for large corpora.

        Chunks are processed concurrently (bounded by ``max_concurrent``) inside
        a single asyncio event loop, giving near-linear speedup over the
        sequential approach since LLM calls are I/O-bound.
        """
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            return

        kg_writer = _CanonicalKGWriter(self._driver, self._config.database, self._collection_name)
        pipeline = SimpleKGPipeline(
            llm=_LLMAdapter(model),
            driver=self._driver,
            embedder=_EmbedderAdapter(self.embedding_model),
            from_pdf=False,
            text_splitter=_PreChunkedTextSplitter(),
            on_error=on_error,
            # GraphRAG's built-in resolver scans every __Entity__ in the DB.
            # Resolve only this collection after all chunks have been written.
            perform_entity_resolution=False,
            kg_writer=kg_writer,
            lexical_graph_config=_KG_LEXICAL_GRAPH_CONFIG,
            neo4j_database=self._config.database,
            **_kg_pipeline_extraction_options(self._kg_extraction_config),
        )

        run_id = uuid.uuid4().hex

        async def _run_all() -> None:
            sem = asyncio.Semaphore(max_concurrent)

            async def _run_one(chunk: AI4RAGChunk) -> None:
                async with sem:
                    with kg_writer.for_chunk(chunk.chunk_id):
                        await pipeline.run_async(
                            file_path=f"ai4rag://{self._collection_name}/{run_id}/{uuid.uuid4().hex}",
                            text=chunk.text,
                            document_metadata={
                                "ai4rag_chunk_id": chunk.chunk_id,
                                "document_id": chunk.metadata.get("document_id", chunk.chunk_id),
                                "source": chunk.metadata.get("source", ""),
                            },
                        )

            await asyncio.gather(*(_run_one(chunk) for chunk in chunks))

        try:
            asyncio.run(_run_all())
        except RuntimeError as exc:
            raise RuntimeError(
                "_run_kg_pipeline cannot be called from within a running event loop. "
                "Install 'nest_asyncio' and call nest_asyncio.apply() beforehand."
            ) from exc

        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"MATCH (e:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->("
                f":`{self._collection_name}`:{_SCHEMA.CHUNK_LABEL}) "
                f"SET e.{_SCHEMA.KG_COLLECTIONS} = CASE "
                f"WHEN $col IN COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"THEN e.{_SCHEMA.KG_COLLECTIONS} "
                f"ELSE COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, []) + $col END",
                col=self._collection_name,
            )

        if perform_entity_resolution:
            self._resolve_kg_entities()

    def _resolve_kg_entities(self) -> None:
        """Merge same-type, same-name entities exclusive to this collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"MATCH (entity:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->("
                f":`{self._collection_name}`:{_SCHEMA.CHUNK_LABEL}) "
                f"WHERE entity.{_SCHEMA.KG_COLLECTIONS} = [$col] "
                "AND entity.name IS NOT NULL "
                "WITH DISTINCT entity, head([label IN labels(entity) "
                f"WHERE NOT label IN ['{_SCHEMA.ENTITY_LABEL}', '{_SCHEMA.KG_BUILDER_LABEL}']]) AS entity_label "
                "WHERE entity_label IS NOT NULL "
                "WITH entity_label, entity.name AS name, collect(entity) AS entities "
                "WHERE size(entities) > 1 "
                "WITH entities, reduce(owners = [], e IN entities | "
                f"owners + COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, [])) AS owners "
                "CALL apoc.refactor.mergeNodes(entities, {properties: 'discard', mergeRels: false}) YIELD node "
                f"SET node.{_SCHEMA.KG_COLLECTIONS} = reduce(unique = [], owner IN owners | "
                "CASE WHEN owner IN unique THEN unique ELSE unique + owner END) "
                "RETURN count(node) AS merged_groups",
                col=self._collection_name,
            )

    def _upsert_doc_groups(self, doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]]) -> None:
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
                f"MERGE (d:{collection_name}:{_SCHEMA.DOCUMENT_LABEL} {{{_SCHEMA.ID}: $doc_id}}) "
                f"SET d.{_SCHEMA.SOURCE} = $source, d.{_SCHEMA.METADATA} = $doc_metadata",
                doc_id=doc_id,
                source=source,
                doc_metadata=json.dumps({"source": source}),
            )

            for chunk, embedding in sorted_pairs:
                clean_metadata = {
                    **chunk.metadata,
                    "source": chunk.metadata.get("source") or doc_id,
                }
                tx.run(
                    f"MERGE (c:{collection_name}:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.ID}: $id}}) "
                    f"SET c.{_SCHEMA.TEXT} = $text, c.{_SCHEMA.EMBEDDING} = $embedding, "
                    f"c.{_SCHEMA.DOCUMENT_ID} = $document_id, c.{_SCHEMA.SEQUENCE_NUMBER} = $sequence_number, "
                    f"c.{_SCHEMA.METADATA} = $metadata, c.{_SCHEMA.COLLECTION} = $collection",
                    id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    document_id=doc_id,
                    sequence_number=chunk.metadata.get("sequence_number", 0),
                    metadata=json.dumps(clean_metadata),
                    collection=collection_name,
                )
                tx.run(
                    f"MATCH (d:{collection_name}:{_SCHEMA.DOCUMENT_LABEL} {{{_SCHEMA.ID}: $doc_id}}) "
                    f"MATCH (c:{collection_name}:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.ID}: $chunk_id}}) "
                    f"MERGE (d)-[:{_SCHEMA.CONTAINS_RELATIONSHIP}]->(c)",
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                )

            for i in range(len(sorted_pairs) - 1):
                chunk_a = sorted_pairs[i][0]
                chunk_b = sorted_pairs[i + 1][0]
                tx.run(
                    f"MATCH (a:{collection_name}:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.ID}: $id_a}}) "
                    f"MATCH (b:{collection_name}:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.ID}: $id_b}}) "
                    f"MERGE (a)-[:{_SCHEMA.NEXT_CHUNK_RELATIONSHIP}]->(b)",
                    id_a=chunk_a.chunk_id,
                    id_b=chunk_b.chunk_id,
                )

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        search_mode: str = "graph",
        ranker_strategy: str | None = None,
        ranker_k: int | None = None,
        ranker_alpha: float | None = None,
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
        search_mode : str, default="graph"
            ``"graph"`` — ANN seed retrieval + entity-based chunk discovery via
            :class:`neo4j_graphrag.retrievers.VectorCypherRetriever` against the
            collection-specific vector index (requires extracted entities for
            graph-neighbor retrieval). Each result remains a single stored chunk.
        **kwargs : Any
            Graph-mode parameters:

            - ``include_entity_neighbors`` (bool, default True) — expand via ``__Entity__``.
            - ``entity_neighbor_limit`` (int, default 5) — max entity-linked neighbors per seed.
            - ``entity_pivot_limit`` (int, default 3) — max entity pivots for relationship traversal.
            - ``entity_relationship_hops`` (int, default 1) — relationship hops from each pivot.
            - ``relationship_neighbor_limit`` (int, default 5) — max relationship-expanded chunks per seed.
        """
        _validate_neo4j_search_params(
            search_mode,
            ranker_strategy,
            ranker_k,
            ranker_alpha,
            store_class=type(self),
            **kwargs,
        )

        return self._search_graph(query, k, include_scores, **kwargs)

    def _search_graph(
        self,
        query: str,
        k: int,
        include_scores: bool,
        **kwargs,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run direct-vector and local-graph retrieval concurrently."""
        route_k = kwargs.get("route_k", max(k * 2, 10))
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self._search_vector_route, query, route_k)
            local_future = executor.submit(self._search_local_route, query, route_k, **kwargs)
            routes = {
                "vector": vector_future.result(),
                "local": local_future.result(),
            }

        pairs = self._fuse_graph_routes(routes, k)
        if include_scores:
            return pairs
        return [chunk for chunk, _ in pairs]

    def _search_local_route(
        self,
        query: str,
        k: int,
        **kwargs,
    ) -> list[tuple[AI4RAGChunk, float]]:
        """Return individual entity/path-linked chunks using VectorCypherRetriever."""
        include_entity_neighbors = kwargs.get("include_entity_neighbors", True)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 3)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 1)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 5)

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
            if chunk_meta.get("source") is None:
                chunk_meta["source"] = ""
            if chunk_meta.get("document_id") is None:
                chunk_meta["document_id"] = record.get("document_id") or ""
            return RetrieverResultItem(
                content=record.get("text") or "",
                metadata={
                    "score": float(record.get("score", 0.0)),
                    "_meta": chunk_meta,
                },
            )

        retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name=_collection_vector_index_name(self._collection_name),
            retrieval_query=_build_graph_retrieval_query(
                include_entity_neighbors,
                entity_neighbor_limit,
                entity_pivot_limit,
                entity_relationship_hops,
                relationship_neighbor_limit,
            ),
            embedder=_EmbedderAdapter(self.embedding_model),
            result_formatter=_fmt,
            neo4j_database=self._config.database,
        )
        result = retriever.search(query_text=query, top_k=k, query_params={"col": self._collection_name})

        return [
            (
                AI4RAGChunk(text=item.content, metadata=item.metadata.get("_meta", {})),
                float(item.metadata.get("score", 0.0)),
            )
            for item in result.items
            if item.content
        ]

    def _search_vector_route(self, query: str, k: int) -> list[tuple[AI4RAGChunk, float]]:
        """Return raw semantic chunk evidence without graph expansion."""
        return self._search_vector_index(
            query=query,
            k=k,
            index_name=_collection_vector_index_name(self._collection_name),
            text_property=_SCHEMA.TEXT,
            route="vector",
        )

    def _search_vector_index(
        self,
        query: str,
        k: int,
        index_name: str,
        text_property: str,
        route: str,
    ) -> list[tuple[AI4RAGChunk, float]]:
        """Run a Neo4j GraphRAG vector retriever and normalize its evidence."""

        def _format(record) -> RetrieverResultItem:
            # VectorRetriever returns the indexed node under ``node``.  Retain
            # a fallback to top-level properties for compatibility with older
            # GraphRAG versions and mocked retriever records.
            node = record.get("node") or record
            raw_metadata = node.get("metadata")
            if isinstance(raw_metadata, str) and raw_metadata:
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = dict(raw_metadata or {})
            document_id = node.get("document_id")
            if document_id is not None:
                metadata.setdefault("document_id", document_id)
            return RetrieverResultItem(
                content=node.get(text_property) or "",
                metadata={
                    "score": float(record.get("score", 0.0)),
                    "_meta": {**metadata, "route": route},
                },
            )

        retriever = VectorRetriever(
            driver=self._driver,
            index_name=index_name,
            embedder=_EmbedderAdapter(self.embedding_model),
            result_formatter=_format,
            neo4j_database=self._config.database,
        )
        result = retriever.search(query_text=query, top_k=k)
        return [
            (
                AI4RAGChunk(text=item.content, metadata=item.metadata.get("_meta", {})),
                float(item.metadata.get("score", 0.0)),
            )
            for item in result.items
            if item.content
        ]

    @staticmethod
    def _fuse_graph_routes(
        routes: dict[str, list[tuple[AI4RAGChunk, float]]], k: int
    ) -> list[tuple[AI4RAGChunk, float]]:
        """Fuse route rankings with reciprocal-rank fusion and deduplicate text."""
        fused: dict[str, tuple[AI4RAGChunk, float]] = {}
        for route, pairs in routes.items():
            for rank, (chunk, _score) in enumerate(pairs, start=1):
                key = chunk.text.strip()
                if not key:
                    continue
                contribution = 1.0 / (60 + rank)
                if key in fused:
                    previous, score = fused[key]
                    previous.metadata.setdefault("routes", []).append(route)
                    fused[key] = (previous, score + contribution)
                else:
                    chunk.metadata["routes"] = [route]
                    fused[key] = (chunk, contribution)
        return sorted(fused.values(), key=lambda pair: pair[1], reverse=True)[:k]

    def build_knowledge_graph_from_documents(
        self,
        documents: list[DoclingDocument],
        model: BaseFoundationModel,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        on_error: str = "IGNORE",
        perform_entity_resolution: bool = True,
    ) -> None:
        """Chunk Docling documents and build one collection graph.

        Uses ``FixedSizeSplitter`` to create canonical ``AI4RAGChunk`` objects,
        then delegates storage and entity extraction to :meth:`add_documents`.
        The KG writer links extracted entities to those stored chunks by ID;
        it does not persist a second set of pipeline document or chunk nodes.

        Parameters
        ----------
        documents : list[DoclingDocument]
            Parsed documents to process.
        model : BaseFoundationModel
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
        splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        async def _split_documents() -> list[AI4RAGChunk]:
            chunks: list[AI4RAGChunk] = []
            for doc in documents:
                text = doc.export_to_markdown()
                if not text.strip():
                    continue
                name = getattr(doc, "name", None)
                document_id = name if isinstance(name, str) and name else hashlib.sha256(text.encode()).hexdigest()
                result = await splitter.run(text)
                chunks.extend(
                    AI4RAGChunk(
                        text=part.text,
                        metadata={"document_id": document_id, "sequence_number": part.index, "source": document_id},
                    )
                    for part in result.chunks
                )
            return chunks

        try:
            chunks = asyncio.run(_split_documents())
        except RuntimeError as exc:
            raise RuntimeError(
                "build_knowledge_graph_from_documents cannot be called from within a running "
                "event loop.  Install 'nest_asyncio' and call nest_asyncio.apply() before "
                "invoking this method in a Jupyter notebook or other async context."
            ) from exc

        if not chunks:
            logger.info("No text extracted from documents; skipping KG build.")
            return

        self.add_documents(
            chunks,
            model=model,
            on_error=on_error,
            perform_entity_resolution=perform_entity_resolution,
        )

        logger.info(
            "Knowledge graph built from %d documents (collection=%s).",
            len(documents),
            self._collection_name,
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
                        dup=dup_eid,
                        canon=canonical_eid,
                    )
                    session.run(
                        "MATCH (dup)-[:RELATED_TO]->(target) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (canon)-[:RELATED_TO]->(target)",
                        dup=dup_eid,
                        canon=canonical_eid,
                    )
                    session.run(
                        "MATCH (src)-[:RELATED_TO]->(dup) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (src)-[:RELATED_TO]->(canon)",
                        dup=dup_eid,
                        canon=canonical_eid,
                    )
                    session.run(
                        "MATCH (dup) WHERE elementId(dup) = $dup DETACH DELETE dup",
                        dup=dup_eid,
                    )
                    removed += 1

        return removed

    def clean_collection(self) -> None:
        """Delete all nodes, KG entities, and the vector index for this collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(f"DROP INDEX `{_collection_vector_index_name(self._collection_name)}` IF EXISTS")
            session.run(
                f"MATCH (:{_SCHEMA.ENTITY_LABEL})-[r]->(:{_SCHEMA.ENTITY_LABEL}) "
                f"WHERE $col IN COALESCE(r.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"AND size(r.{_SCHEMA.KG_COLLECTIONS}) = 1 DELETE r",
                col=self._collection_name,
            )
            session.run(
                f"MATCH (:{_SCHEMA.ENTITY_LABEL})-[r]->(:{_SCHEMA.ENTITY_LABEL}) "
                f"WHERE $col IN COALESCE(r.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"SET r.{_SCHEMA.KG_COLLECTIONS} = "
                f"[owner IN r.{_SCHEMA.KG_COLLECTIONS} WHERE owner <> $col]",
                col=self._collection_name,
            )
            # Legacy entities have no ownership tag. Only delete those linked
            # exclusively to chunks from this collection, while those links
            # still exist to prove their provenance.
            session.run(
                f"MATCH (e:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(c:{_SCHEMA.CHUNK_LABEL}) "
                f"WHERE (c.{_SCHEMA.COLLECTION} = $col OR $col IN labels(c)) "
                f"AND size(COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, [])) = 0 "
                f"AND NOT EXISTS {{ MATCH (e)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(other:{_SCHEMA.CHUNK_LABEL}) "
                f"WHERE COALESCE(other.{_SCHEMA.COLLECTION}, '') <> $col AND NOT $col IN labels(other) }} "
                "DETACH DELETE e",
                col=self._collection_name,
            )
            # Entity nodes can be shared by several collections. Remove this
            # collection's ownership first and delete only entities no longer
            # owned by any AI4RAG collection.
            session.run(
                f"MATCH (e:{_SCHEMA.ENTITY_LABEL}) WHERE $col IN COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"SET e.{_SCHEMA.KG_COLLECTIONS} = "
                f"[owner IN e.{_SCHEMA.KG_COLLECTIONS} WHERE owner <> $col] "
                f"WITH e WHERE size(e.{_SCHEMA.KG_COLLECTIONS}) = 0 "
                "DETACH DELETE e",
                col=self._collection_name,
            )
            session.run(f"MATCH (n:{self._collection_name}) DETACH DELETE n")
            session.run(
                f"MATCH (c:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.COLLECTION}: $col}}) DETACH DELETE c",
                col=self._collection_name,
            )
        logger.info("Collection %s cleaned.", self._collection_name)

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()
