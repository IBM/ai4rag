# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# pylint: disable=too-many-lines
import asyncio
import hashlib
import json
import re
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import neo4j
from docling_core.types.doc import DoclingDocument
from json_repair import repair_json
from neo4j_graphrag.components.kg_writer import KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph, Neo4jRelationship, TextChunk, TextChunks
from neo4j_graphrag.embeddings.base import Embedder as _NeoEmbedder
from neo4j_graphrag.llm.base import LLMInterface as _NeoLLMInterface
from neo4j_graphrag.neo4j_queries import db_cleaning_query, upsert_relationship_query

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.utils import (
    iter_unique_chunks,
    resolve_embedding_dimension,
    validate_search_params,
)

__all__ = ["Neo4jGraphStore"]

_CONSTRAINED_KG_ENTITIES = ("Person", "Organization", "Place", "Concept", "Event", "Product", "Technology")
_CONSTRAINED_KG_RELATIONS = ("RELATED_TO", "PART_OF", "LOCATED_IN", "BELONGS_TO", "CREATED_BY", "MENTIONS")

# GraphRAG's lexical nodes are temporary: _CanonicalKGWriter discards them in
# favor of the canonical nodes already stored by add_documents. Keep their
# labels distinct from entity types chosen by free extraction (e.g. Document).
_KG_LEXICAL_GRAPH_CONFIG = LexicalGraphConfig(
    document_node_label="__AI4RAG_KG_SOURCE_DOCUMENT__",
    chunk_node_label="__AI4RAG_KG_SOURCE_CHUNK__",
)


class _PreChunkedTextSplitter(TextSplitter):
    """Preserve the canonical chunk passed to the KG pipeline as one text chunk."""

    async def run(self, text: str) -> TextChunks:
        return TextChunks(chunks=[TextChunk(text=text, index=0)])


def _collection_vector_index_name(collection_name: str) -> str:
    """Return the vector-index name dedicated to *collection_name*."""
    return f"{collection_name}__embedding"


def _validate_kg_extraction_config(config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate KG extraction settings and apply the constrained default."""
    if config is None:
        return {"mode": "constrained"}
    if not isinstance(config, dict):
        raise TypeError("kg_extraction_config must be a dictionary or None.")

    mode = config.get("mode", "constrained")
    if mode not in {"constrained", "free"}:
        raise ValueError("kg_extraction_config.mode must be 'constrained' or 'free'.")
    if mode == "constrained":
        return {"mode": mode}

    limits = {
        "max_entities_per_chunk": config.get("max_entities_per_chunk"),
        "max_relationships_per_chunk": config.get("max_relationships_per_chunk"),
    }
    for name, value in limits.items():
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"kg_extraction_config.{name} must be a positive integer for free extraction.")
    return {"mode": mode, **limits}


def _kg_pipeline_extraction_options(config: dict[str, Any]) -> dict[str, Any]:
    """Return ``SimpleKGPipeline`` options for the requested extraction mode."""
    if config["mode"] == "constrained":
        return {"entities": _CONSTRAINED_KG_ENTITIES, "relations": _CONSTRAINED_KG_RELATIONS}

    from neo4j_graphrag.generation.prompts import ERExtractionTemplate

    entity_limit = config["max_entities_per_chunk"]
    relation_limit = config["max_relationships_per_chunk"]
    template = ERExtractionTemplate.DEFAULT_TEMPLATE.replace(
        "Extract the entities (nodes) and specify their type from the following text.",
        "Extract the entities (nodes) and specify their type from the following text. "
        f"Extract at most {entity_limit} entities.",
    ).replace(
        "Also extract the relationships between these nodes.",
        f"Also extract at most {relation_limit} relationships between these nodes.",
    )
    # "FREE" bypasses automatic schema extraction. Omitting the schema would
    # ask the model to generate constraints, which can be invalid or conflicting.
    # The model still determines entity and relationship types from each chunk.
    return {"schema": "FREE", "prompt_template": ERExtractionTemplate(template=template)}


class _EmbedderAdapter(_NeoEmbedder):  # type: ignore[misc]
    """Wraps :class:`BaseEmbeddingModel` to satisfy ``neo4j_graphrag``'s embedder interface."""

    def __init__(self, model: BaseEmbeddingModel) -> None:
        super().__init__()
        self._model = model

    def embed_query(self, text: str) -> list[float]:
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
        input: str,  # pylint: disable=redefined-builtin
        message_history=None,  # pylint: disable=unused-argument
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
        input: str,  # pylint: disable=redefined-builtin
        message_history=None,
        system_instruction: str | None = None,
    ):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.invoke, input, message_history, system_instruction)


class _CollectionKGWriter(Neo4jWriter):
    """Record ownership on precisely the relationships written by the pipeline."""

    def __init__(self, driver: neo4j.Driver, database: str, collection_name: str) -> None:
        super().__init__(driver=driver, neo4j_database=database)
        self._collection_name = collection_name

    def _db_setup(self) -> None:
        self.driver.execute_query(
            "CREATE INDEX __entity__tmp_internal_id IF NOT EXISTS FOR (n:__KGBuilder__) ON (n.__tmp_internal_id)",
            database_=self.neo4j_database,
        )

    def _db_cleaning(self) -> None:
        query = db_cleaning_query(
            support_variable_scope_clause=self.is_version_5_23_or_above,
            batch_size=self.batch_size,
        )
        with self.driver.session(database=self.neo4j_database) as session:
            session.run(query)

    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        return await super().run(graph, lexical_graph_config)

    def _upsert_relationships(self, rels: list[Neo4jRelationship]) -> None:
        query = upsert_relationship_query(support_variable_scope_clause=self.is_version_5_23_or_above)
        if query.count("RETURN elementId(rel)") != 1:
            raise RuntimeError("Neo4j GraphRAG relationship writer query changed; cannot record collection ownership.")
        query = query.replace(
            "RETURN elementId(rel)",
            "SET rel.ai4rag_kg_collections = CASE "
            "WHEN $col IN COALESCE(rel.ai4rag_kg_collections, []) "
            "THEN rel.ai4rag_kg_collections "
            "ELSE COALESCE(rel.ai4rag_kg_collections, []) + $col END "
            "RETURN elementId(rel)",
        )
        self.driver.execute_query(
            query,
            parameters_={"rows": self._relationships_to_rows(rels), "col": self._collection_name},
            database_=self.neo4j_database,
        )


class _CanonicalKGWriter(_CollectionKGWriter):
    """Write extracted entities onto chunks already stored by ``add_documents``."""

    def __init__(self, driver: neo4j.Driver, database: str, collection_name: str) -> None:
        super().__init__(driver, database, collection_name)
        self._links: ContextVar[tuple[str, list[str]] | None] = ContextVar("canonical_kg_links", default=None)
        self._chunk_id: ContextVar[str | None] = ContextVar("canonical_kg_chunk_id", default=None)

    @contextmanager
    def for_chunk(self, chunk_id: str) -> Iterator[None]:
        """Bind the canonical ID to one asynchronous KG pipeline run."""
        token = self._chunk_id.set(chunk_id)
        try:
            yield
        finally:
            self._chunk_id.reset(token)

    async def run(
        self,
        graph: Neo4jGraph | dict[str, Any],
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        # The pipeline serializes component results before passing them to the
        # next component, so pruner.graph arrives as a dictionary.
        if isinstance(graph, dict):
            graph = Neo4jGraph.model_validate(graph)
        documents = [node for node in graph.nodes if node.label == lexical_graph_config.document_node_label]
        entity_nodes = [
            node for node in graph.nodes if node.label not in lexical_graph_config.lexical_graph_node_labels
        ]
        entity_types = dict(sorted(Counter(node.label for node in entity_nodes).items()))
        chunk_id = self._chunk_id.get()
        logger.info(
            "KG extraction for collection '%s', chunk '%s': entity types=%s, source document nodes=%d",
            self._collection_name,
            chunk_id or (documents[0].properties.get("ai4rag_chunk_id") if documents else "unknown"),
            entity_types,
            len(documents),
        )
        if len(documents) > 1:
            raise ValueError("KG extraction produced multiple document nodes for one canonical chunk.")
        if documents:
            document_chunk_id = documents[0].properties.get("ai4rag_chunk_id")
            if chunk_id is not None and document_chunk_id is not None and document_chunk_id != chunk_id:
                raise ValueError("KG extraction document metadata does not match the canonical chunk ID.")
            chunk_id = chunk_id or document_chunk_id
        if not chunk_id:
            raise ValueError("KG extraction requires the canonical chunk ID.")

        entity_ids = {node.id for node in entity_nodes}
        # This pipeline processes one canonical chunk per run. GraphRAG may omit
        # lexical nodes and FROM_CHUNK edges, but all extracted entities still
        # belong to that chunk.
        linked_entity_ids = sorted(entity_ids)
        entity_graph = Neo4jGraph(
            nodes=entity_nodes,
            relationships=[
                rel for rel in graph.relationships if rel.start_node_id in entity_ids and rel.end_node_id in entity_ids
            ],
        )
        token = self._links.set((chunk_id, linked_entity_ids))
        try:
            return await super().run(entity_graph, lexical_graph_config)
        finally:
            self._links.reset(token)

    def _db_cleaning(self) -> None:
        links = self._links.get()
        if links is not None:
            chunk_id, entity_ids = links
            if entity_ids:
                self.driver.execute_query(
                    f"UNWIND $entity_ids AS entity_id "
                    f"MATCH (e:__Entity__ {{__tmp_internal_id: entity_id}}) "
                    f"MATCH (c:`{self._collection_name}`:Chunk {{id: $chunk_id}}) "
                    "MERGE (e)-[:FROM_CHUNK]->(c) "
                    "SET e.ai4rag_kg_collections = CASE "
                    "WHEN $col IN COALESCE(e.ai4rag_kg_collections, []) "
                    "THEN e.ai4rag_kg_collections "
                    "ELSE COALESCE(e.ai4rag_kg_collections, []) + $col END",
                    parameters_={"entity_ids": entity_ids, "chunk_id": chunk_id, "col": self._collection_name},
                    database_=self.neo4j_database,
                )
        super()._db_cleaning()


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
        foundation_model: Any = None,
        kg_extraction_config: dict[str, Any] | None = None,
    ):
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(embedding_model)
        self._foundation_model = foundation_model
        self._kg_extraction_config = _validate_kg_extraction_config(kg_extraction_config)
        self._driver = neo4j.GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
            # MaaS embedding calls can exceed the load balancer's idle Bolt timeout.
            # Recycle pooled connections before they become stale between indexing steps.
            max_connection_lifetime=30.0,
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
                f"FOR (n:`{self._collection_name}`) ON (n.embedding) "
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
        from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

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
                f"MATCH (e:__Entity__)-[:FROM_CHUNK]->(:`{self._collection_name}`:Chunk) "
                "SET e.ai4rag_kg_collections = CASE "
                "WHEN $col IN COALESCE(e.ai4rag_kg_collections, []) "
                "THEN e.ai4rag_kg_collections "
                "ELSE COALESCE(e.ai4rag_kg_collections, []) + $col END",
                col=self._collection_name,
            )

        if perform_entity_resolution:
            self._resolve_kg_entities()

    def _resolve_kg_entities(self) -> None:
        """Merge same-type, same-name entities exclusive to this collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"MATCH (entity:__Entity__)-[:FROM_CHUNK]->(:`{self._collection_name}`:Chunk) "
                "WHERE entity.ai4rag_kg_collections = [$col] "
                "AND entity.name IS NOT NULL "
                "WITH DISTINCT entity, head([label IN labels(entity) "
                "WHERE NOT label IN ['__Entity__', '__KGBuilder__']]) AS entity_label "
                "WHERE entity_label IS NOT NULL "
                "WITH entity_label, entity.name AS name, collect(entity) AS entities "
                "WHERE size(entities) > 1 "
                "WITH entities, reduce(owners = [], e IN entities | "
                "owners + COALESCE(e.ai4rag_kg_collections, [])) AS owners "
                "CALL apoc.refactor.mergeNodes(entities, {properties: 'discard', mergeRels: false}) YIELD node "
                "SET node.ai4rag_kg_collections = reduce(unique = [], owner IN owners | "
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
                f"MERGE (d:{collection_name}:Document {{id: $doc_id}}) "
                f"SET d.source = $source, d.metadata = $doc_metadata",
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
                    f"MERGE (c:{collection_name}:Chunk {{id: $id}}) "
                    f"SET c.text = $text, c.embedding = $embedding, "
                    f"c.document_id = $document_id, c.sequence_number = $sequence_number, "
                    f"c.metadata = $metadata, c.collection = $collection",
                    id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    document_id=doc_id,
                    sequence_number=chunk.metadata.get("sequence_number", 0),
                    metadata=json.dumps(clean_metadata),
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
            - ``entity_pivot_limit`` (int, default 0) — max entity pivots for relationship traversal.
            - ``entity_relationship_hops`` (int, default 0) — relationship hops from each pivot.
            - ``relationship_neighbor_limit`` (int, default 0) — max relationship-expanded chunks per seed.
        """
        _validate_neo4j_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha, **kwargs)

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
        from neo4j_graphrag.retrievers import VectorCypherRetriever
        from neo4j_graphrag.types import RetrieverResultItem

        include_entity_neighbors = kwargs.get("include_entity_neighbors", True)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 0)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 0)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 0)

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
            text_property="text",
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
        from neo4j_graphrag.retrievers import VectorRetriever
        from neo4j_graphrag.types import RetrieverResultItem

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
        """Chunk Docling documents and build one collection graph.

        Uses ``FixedSizeSplitter`` to create canonical ``AI4RAGChunk`` objects,
        then delegates storage and entity extraction to :meth:`add_documents`.
        The KG writer links extracted entities to those stored chunks by ID;
        it does not persist a second set of pipeline document or chunk nodes.

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
        from neo4j_graphrag.components.text_splitters.fixed_size_splitter import (
            FixedSizeSplitter,
        )

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

        logger.info(
            "Entity resolver removed %d duplicate nodes from collection %s.",
            removed,
            self._collection_name,
        )
        return removed

    def clean_collection(self) -> None:
        """Delete all nodes, KG entities, and the vector index for this collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(f"DROP INDEX `{_collection_vector_index_name(self._collection_name)}` IF EXISTS")
            # Clean up indexes left by older Neo4j implementations.
            session.run(f"DROP INDEX `{self._collection_name}__vector` IF EXISTS")
            session.run(f"DROP INDEX `{self._collection_name}__fulltext` IF EXISTS")
            session.run(
                "MATCH (:__Entity__)-[r]->(:__Entity__) "
                "WHERE $col IN COALESCE(r.ai4rag_kg_collections, []) "
                "AND size(r.ai4rag_kg_collections) = 1 DELETE r",
                col=self._collection_name,
            )
            session.run(
                "MATCH (:__Entity__)-[r]->(:__Entity__) "
                "WHERE $col IN COALESCE(r.ai4rag_kg_collections, []) "
                "SET r.ai4rag_kg_collections = "
                "[owner IN r.ai4rag_kg_collections WHERE owner <> $col]",
                col=self._collection_name,
            )
            # Legacy entities have no ownership tag. Only delete those linked
            # exclusively to chunks from this collection, while those links
            # still exist to prove their provenance.
            session.run(
                "MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk) "
                "WHERE (c.collection = $col OR $col IN labels(c)) "
                "AND size(COALESCE(e.ai4rag_kg_collections, [])) = 0 "
                "AND NOT EXISTS { MATCH (e)-[:FROM_CHUNK]->(other:Chunk) "
                "WHERE COALESCE(other.collection, '') <> $col AND NOT $col IN labels(other) } "
                "DETACH DELETE e",
                col=self._collection_name,
            )
            # Entity nodes can be shared by several collections. Remove this
            # collection's ownership first and delete only entities no longer
            # owned by any AI4RAG collection.
            session.run(
                "MATCH (e:__Entity__) WHERE $col IN COALESCE(e.ai4rag_kg_collections, []) "
                "SET e.ai4rag_kg_collections = "
                "[owner IN e.ai4rag_kg_collections WHERE owner <> $col] "
                "WITH e WHERE size(e.ai4rag_kg_collections) = 0 "
                "DETACH DELETE e",
                col=self._collection_name,
            )
            session.run(f"MATCH (n:{self._collection_name}) DETACH DELETE n")
            # Remove pipeline documents from collections created by older releases.
            session.run(
                "MATCH (d:Document {ai4rag_kg_collection: $col}) DETACH DELETE d",
                col=self._collection_name,
            )
            session.run(
                "MATCH (c:Chunk {collection: $col}) DETACH DELETE c",
                col=self._collection_name,
            )
        logger.info("Collection %s cleaned.", self._collection_name)

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()


# ---------------------------------------------------------------------------
# KG extraction response normalization
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Graph retrieval query builder
# ---------------------------------------------------------------------------


def _build_graph_retrieval_query(
    include_entity_neighbors: bool,
    entity_neighbor_limit: int,
    entity_pivot_limit: int = 0,
    entity_relationship_hops: int = 0,
    relationship_neighbor_limit: int = 0,
) -> str:
    """Build the Cypher retrieval query for :class:`VectorCypherRetriever`.

    The query receives seed ``Chunk`` nodes from vector search, ranks their
    graph neighbors by query similarity and graph support, then returns
    distinct chunks as individual results. A seed with no graph neighbors is
    returned as a fallback. Candidate chunks are scored using their own
    embeddings rather than inheriting a seed's ANN score.
    """
    # $col is passed via query_params in _search_graph to scope graph expansion
    # to one collection. The seed index is already collection-specific.
    collection_filter = "WHERE node.collection = $col "

    if include_entity_neighbors and entity_neighbor_limit:
        entity_block = (
            "CALL (node) { "
            "OPTIONAL MATCH (entity:__Entity__)-[:FROM_CHUNK]->(node) "
            "OPTIONAL MATCH (entity)-[:FROM_CHUNK]->(ent_nb:Chunk) "
            "WHERE ent_nb <> node AND ent_nb.collection = $col "
            "WITH ent_nb, count(DISTINCT entity) AS overlap "
            "WHERE ent_nb IS NOT NULL AND ent_nb.embedding IS NOT NULL "
            "WITH ent_nb, overlap, vector.similarity.cosine(ent_nb.embedding, $query_vector) AS similarity "
            "ORDER BY similarity DESC, overlap DESC, ent_nb.id ASC "
            f"LIMIT {entity_neighbor_limit} "
            "RETURN collect({chunk: ent_nb, strength: toFloat(overlap)}) AS ent_hits } "
        )
    else:
        entity_block = "WITH node, [] AS ent_hits "

    if entity_pivot_limit and entity_relationship_hops and relationship_neighbor_limit:
        relationship_block = (
            "CALL (node) { "
            "MATCH (pivot:__Entity__)-[:FROM_CHUNK]->(node) "
            "OPTIONAL MATCH (pivot)-[:FROM_CHUNK]->(pivot_chunk:Chunk) "
            "WHERE pivot_chunk.collection = $col "
            "WITH node, pivot, count(DISTINCT pivot_chunk) AS degree "
            "ORDER BY degree DESC, pivot.name ASC, pivot.id ASC "
            f"LIMIT {entity_pivot_limit} "
            f"MATCH path = (pivot)-[*1..{entity_relationship_hops}]-(related:__Entity__) "
            "WHERE ALL(rel IN relationships(path) WHERE type(rel) <> 'FROM_CHUNK' "
            "AND $col IN COALESCE(rel.ai4rag_kg_collections, [])) "
            "MATCH (related)-[:FROM_CHUNK]->(rel_nb:Chunk) "
            "WHERE rel_nb <> node AND rel_nb.collection = $col AND rel_nb.embedding IS NOT NULL "
            "WITH rel_nb, count(DISTINCT path) AS path_count, min(length(path)) AS hops "
            "WITH rel_nb, path_count, hops, "
            "vector.similarity.cosine(rel_nb.embedding, $query_vector) AS similarity "
            "ORDER BY similarity DESC, path_count DESC, hops ASC, rel_nb.id ASC "
            f"LIMIT {relationship_neighbor_limit} "
            "RETURN collect({chunk: rel_nb, strength: toFloat(path_count) / toFloat(hops)}) AS rel_hits } "
        )
    else:
        relationship_block = "WITH node, ent_hits, [] AS rel_hits "

    # Give independent query similarity most of the weight; bound accumulated
    # graph support so high-degree entities cannot dominate the local route.
    return (
        collection_filter
        + entity_block
        + relationship_block
        + "WITH node, ent_hits + rel_hits AS graph_hits "
        + "WITH CASE WHEN size(graph_hits) = 0 "
        + "THEN [{chunk: node, strength: 0.0}] ELSE graph_hits END AS hits "
        + "UNWIND hits AS hit "
        + "WITH hit.chunk AS candidate, sum(hit.strength) AS graph_strength "
        + "WITH candidate, graph_strength, "
        + "coalesce(vector.similarity.cosine(candidate.embedding, $query_vector), 0.0) AS semantic_score "
        + "WITH candidate, graph_strength, "
        + "0.8 * semantic_score + 0.2 * graph_strength / (1.0 + graph_strength) AS score "
        + "ORDER BY score DESC, graph_strength DESC, candidate.id ASC LIMIT $top_k "
        + "RETURN candidate.text AS text, score, "
        + "candidate.document_id AS document_id, candidate.metadata AS metadata"
    )


def _validate_neo4j_search_params(
    search_mode: str,
    ranker_strategy: str | None = None,
    ranker_k: int | None = None,
    ranker_alpha: float | None = None,
    **kwargs: Any,
) -> None:
    if search_mode != "graph":
        raise ValueError("Neo4jGraphStore supports only search_mode='graph'.")

    validate_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha)

    if search_mode == "graph":
        graph_hops = kwargs.get("graph_hops", 0)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 0)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 0)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 0)
        if not isinstance(graph_hops, int) or graph_hops != 0:
            raise ValueError(
                f"graph_hops must be 0 because Neo4j graph search does not expand NEXT_CHUNK neighbors, "
                f"got {graph_hops!r}."
            )
        if not isinstance(entity_neighbor_limit, int) or entity_neighbor_limit < 0:
            raise ValueError(f"entity_neighbor_limit must be a non-negative integer, got {entity_neighbor_limit!r}.")
        for name, value in {
            "entity_pivot_limit": entity_pivot_limit,
            "entity_relationship_hops": entity_relationship_hops,
            "relationship_neighbor_limit": relationship_neighbor_limit,
        }.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
