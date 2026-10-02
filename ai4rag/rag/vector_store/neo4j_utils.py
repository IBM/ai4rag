# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Private adapters and helpers used by `ai4rag.rag.vector_store.neo4j`."""

import asyncio
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

import neo4j
from neo4j_graphrag.components.kg_writer import KGWriterModel, Neo4jWriter
from neo4j_graphrag.components.text_splitters.base import TextSplitter
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph, Neo4jRelationship, TextChunk, TextChunks
from neo4j_graphrag.embeddings.base import Embedder as _NeoEmbedder
from neo4j_graphrag.generation.prompts import ERExtractionTemplate
from neo4j_graphrag.llm.base import LLMInterface as _NeoLLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.neo4j_queries import db_cleaning_query, upsert_relationship_query

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, MessageTyped
from ai4rag.rag.vector_store.utils import validate_search_params


class Neo4jGraphSchema:
    """Names used by ai4rag's persistent Neo4j graph schema."""

    DOCUMENT_LABEL = "Document"
    CHUNK_LABEL = "Chunk"
    ENTITY_LABEL = "__Entity__"
    KG_BUILDER_LABEL = "__KGBuilder__"

    CONTAINS_RELATIONSHIP = "CONTAINS"
    NEXT_CHUNK_RELATIONSHIP = "NEXT_CHUNK"
    FROM_CHUNK_RELATIONSHIP = "FROM_CHUNK"

    ID = "id"
    TEXT = "text"
    EMBEDDING = "embedding"
    DOCUMENT_ID = "document_id"
    SEQUENCE_NUMBER = "sequence_number"
    METADATA = "metadata"
    COLLECTION = "collection"
    SOURCE = "source"
    KG_COLLECTIONS = "ai4rag_kg_collections"


_SCHEMA = Neo4jGraphSchema


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
    """Wraps :class:`BaseFoundationModel` to satisfy ``neo4j_graphrag``'s LLM interface.

    ``SimpleKGPipeline`` calls ``ainvoke`` (async); we bridge the sync
    :meth:`BaseFoundationModel.chat` via ``run_in_executor``.  Response JSON
    is normalised so that models returning a JSON *array* (``[{...}]``) are
    converted to the expected object format (``{"nodes": [...], "relationships": [...]}``)
    before the extractor parses the output.
    """

    def __init__(self, model: BaseFoundationModel) -> None:
        super().__init__(model_name=model.model_id)
        self._model = model

    def invoke(
        self,
        input: str,  # pylint: disable=redefined-builtin
        message_history=None,  # pylint: disable=unused-argument
        system_instruction: str | None = None,
    ):
        messages: list[MessageTyped] = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": input})
        response = self._model.chat(messages)[0]
        content = (response["content"] if isinstance(response, Mapping) else response.message.content) or ""
        return LLMResponse(content=content)

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
        chunk_id = self._chunk_id.get()
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
                    f"MATCH (e:{_SCHEMA.ENTITY_LABEL} {{__tmp_internal_id: entity_id}}) "
                    f"MATCH (c:`{self._collection_name}`:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.ID}: $chunk_id}}) "
                    f"MERGE (e)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(c) "
                    f"SET e.{_SCHEMA.KG_COLLECTIONS} = CASE "
                    f"WHEN $col IN COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, []) "
                    f"THEN e.{_SCHEMA.KG_COLLECTIONS} "
                    f"ELSE COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, []) + $col END",
                    parameters_={"entity_ids": entity_ids, "chunk_id": chunk_id, "col": self._collection_name},
                    database_=self.neo4j_database,
                )
        super()._db_cleaning()


def _build_graph_retrieval_query(
    include_entity_neighbors: bool,
    entity_neighbor_limit: int,
    entity_pivot_limit: int = 3,
    entity_relationship_hops: int = 1,
    relationship_neighbor_limit: int = 5,
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
    collection_filter = f"WHERE node.{_SCHEMA.COLLECTION} = $col "

    if include_entity_neighbors and entity_neighbor_limit:
        entity_block = (
            "CALL (node) { "
            f"OPTIONAL MATCH (entity:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(node) "
            f"OPTIONAL MATCH (entity)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(ent_nb:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE ent_nb <> node AND ent_nb.{_SCHEMA.COLLECTION} = $col "
            "WITH ent_nb, count(DISTINCT entity) AS overlap "
            f"WHERE ent_nb IS NOT NULL AND ent_nb.{_SCHEMA.EMBEDDING} IS NOT NULL "
            f"WITH ent_nb, overlap, vector.similarity.cosine(ent_nb.{_SCHEMA.EMBEDDING}, $query_vector) AS similarity "
            f"ORDER BY similarity DESC, overlap DESC, ent_nb.{_SCHEMA.ID} ASC "
            f"LIMIT {entity_neighbor_limit} "
            "RETURN collect({chunk: ent_nb, strength: toFloat(overlap)}) AS ent_hits } "
        )
    else:
        entity_block = "WITH node, [] AS ent_hits "

    if entity_pivot_limit and entity_relationship_hops and relationship_neighbor_limit:
        relationship_block = (
            "CALL (node) { "
            f"MATCH (pivot:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(node) "
            f"OPTIONAL MATCH (pivot)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(pivot_chunk:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE pivot_chunk.{_SCHEMA.COLLECTION} = $col "
            "WITH node, pivot, count(DISTINCT pivot_chunk) AS degree "
            f"ORDER BY degree DESC, pivot.name ASC, pivot.{_SCHEMA.ID} ASC "
            f"LIMIT {entity_pivot_limit} "
            f"MATCH path = (pivot)-[*1..{entity_relationship_hops}]-(related:{_SCHEMA.ENTITY_LABEL}) "
            f"WHERE ALL(rel IN relationships(path) WHERE type(rel) <> '{_SCHEMA.FROM_CHUNK_RELATIONSHIP}' "
            f"AND $col IN COALESCE(rel.{_SCHEMA.KG_COLLECTIONS}, [])) "
            f"MATCH (related)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(rel_nb:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE rel_nb <> node AND rel_nb.{_SCHEMA.COLLECTION} = $col "
            f"AND rel_nb.{_SCHEMA.EMBEDDING} IS NOT NULL "
            "WITH rel_nb, count(DISTINCT path) AS path_count, min(length(path)) AS hops "
            "WITH rel_nb, path_count, hops, "
            f"vector.similarity.cosine(rel_nb.{_SCHEMA.EMBEDDING}, $query_vector) AS similarity "
            f"ORDER BY similarity DESC, path_count DESC, hops ASC, rel_nb.{_SCHEMA.ID} ASC "
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
        + f"coalesce(vector.similarity.cosine(candidate.{_SCHEMA.EMBEDDING}, $query_vector), 0.0) AS semantic_score "
        + "WITH candidate, graph_strength, "
        + "0.8 * semantic_score + 0.2 * graph_strength / (1.0 + graph_strength) AS score "
        + f"ORDER BY score DESC, graph_strength DESC, candidate.{_SCHEMA.ID} ASC LIMIT $top_k "
        + f"RETURN candidate.{_SCHEMA.TEXT} AS text, score, "
        + f"candidate.{_SCHEMA.DOCUMENT_ID} AS document_id, candidate.{_SCHEMA.METADATA} AS metadata"
    )


def _validate_neo4j_search_params(
    search_mode: str,
    ranker_strategy: str | None = None,
    ranker_k: int | None = None,
    ranker_alpha: float | None = None,
    store_class: type | None = None,
    **kwargs: Any,
) -> None:
    # pylint: disable=duplicate-code
    validate_search_params(
        search_mode,
        ranker_strategy,
        ranker_k,
        ranker_alpha,
        supported_modes=("graph",),
        store_class=store_class,
    )
    # pylint: enable=duplicate-code

    if search_mode == "graph":
        graph_hops = kwargs.get("graph_hops", 0)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 3)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 1)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 5)
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
