# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import pytest
from neo4j_graphrag.components.types import Neo4jGraph, Neo4jNode, Neo4jRelationship

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j import (
    _KG_LEXICAL_GRAPH_CONFIG,
    Neo4jGraphStore,
    _build_graph_retrieval_query,
    _CanonicalKGWriter,
    _CollectionKGWriter,
    _kg_pipeline_extraction_options,
    _PreChunkedTextSplitter,
    _validate_kg_extraction_config,
    _validate_neo4j_search_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockEmbeddingModel:
    model_id = "test-embedding"
    params = {"embedding_dimension": 128}

    def embed_documents(self, texts):
        return [[float(i % 10) / 10] * 128 for i in range(len(texts))]

    def embed_query(self, query):
        return [0.1] * 128


@pytest.fixture
def mock_embedding():
    return _MockEmbeddingModel()


@pytest.fixture
def neo4j_config():
    return Neo4jConfig(uri="neo4j://localhost:7687", username="neo4j", password="test")


def _make_store(mock_driver_cls, mock_embedding, neo4j_config, collection_name="ai4rag_test"):
    """Construct a Neo4jGraphStore with a fully mocked neo4j driver."""
    store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name=collection_name)
    return store


# ---------------------------------------------------------------------------
# Neo4jConfig
# ---------------------------------------------------------------------------


class TestNeo4jConfig:
    def test_from_env_reads_required_vars(self):
        env = {
            "NEO4J_URI": "neo4j://host:7687",
            "NEO4J_PASSWORD": "secret",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Neo4jConfig.from_env()
        assert cfg.uri == "neo4j://host:7687"
        assert cfg.password == "secret"
        assert cfg.username == "neo4j"
        assert cfg.database == "neo4j"

    def test_from_env_reads_optional_vars(self):
        env = {
            "NEO4J_URI": "neo4j://host:7687",
            "NEO4J_PASSWORD": "pw",
            "NEO4J_USERNAME": "admin",
            "NEO4J_DATABASE": "mydb",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = Neo4jConfig.from_env()
        assert cfg.username == "admin"
        assert cfg.database == "mydb"

    def test_from_env_missing_uri_raises(self):
        env = {"NEO4J_PASSWORD": "pw"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(KeyError):
                # Remove NEO4J_URI if it somehow leaks in
                os.environ.pop("NEO4J_URI", None)
                Neo4jConfig.from_env()

    def test_from_env_missing_password_raises(self):
        env = {"NEO4J_URI": "neo4j://host:7687"}
        with patch.dict(os.environ, env, clear=False):
            with pytest.raises(KeyError):
                os.environ.pop("NEO4J_PASSWORD", None)
                Neo4jConfig.from_env()

    def test_provider_is_neo4j(self):
        cfg = Neo4jConfig(uri="neo4j://h:7687", password="pw")
        assert cfg.provider == "neo4j"


class TestKGExtractionConfig:
    def test_constrained_mode_uses_the_fixed_schema(self):
        options = _kg_pipeline_extraction_options(_validate_kg_extraction_config({"mode": "constrained"}))

        assert "entities" in options
        assert "relations" in options

    def test_free_mode_skips_schema_extraction_and_caps_each_chunk(self):
        options = _kg_pipeline_extraction_options(
            _validate_kg_extraction_config(
                {"mode": "free", "max_entities_per_chunk": 5, "max_relationships_per_chunk": 5}
            )
        )

        assert options["schema"] == "FREE"
        assert "entities" not in options
        assert "relations" not in options
        assert "at most 5 entities" in options["prompt_template"].template
        assert "at most 5 relationships" in options["prompt_template"].template


def test_balanced_graph_query_limits_pivots_hops_and_related_chunks():
    """Balanced graph retrieval must bound relationship traversal per seed."""
    query = _build_graph_retrieval_query(
        include_entity_neighbors=True,
        entity_neighbor_limit=5,
        entity_pivot_limit=3,
        entity_relationship_hops=2,
        relationship_neighbor_limit=5,
    )

    assert "count(DISTINCT entity) AS overlap" in query
    assert "ORDER BY similarity DESC, overlap DESC, ent_nb.id ASC LIMIT 5" in query
    assert "count(DISTINCT pivot_chunk) AS degree" in query
    assert "ORDER BY degree DESC, pivot.name ASC, pivot.id ASC LIMIT 3" in query
    assert "[*1..2]-(related:__Entity__)" in query
    assert "count(DISTINCT path) AS path_count, min(length(path)) AS hops" in query
    assert "ORDER BY similarity DESC, path_count DESC, hops ASC, rel_nb.id ASC LIMIT 5" in query
    assert "type(rel) <> 'FROM_CHUNK'" in query
    assert "$col IN COALESCE(rel.ai4rag_kg_collections, [])" in query
    assert "UNWIND hits AS hit" in query
    assert "sum(hit.strength) AS graph_strength" in query
    assert "vector.similarity.cosine(candidate.embedding, $query_vector)" in query
    assert "0.8 * semantic_score + 0.2 * graph_strength / (1.0 + graph_strength)" in query
    assert "LIMIT $top_k" in query
    assert "ORDER BY score DESC, graph_strength DESC, candidate.id ASC" in query
    assert "RETURN candidate.text AS text" in query
    assert "elementId(candidate)" not in query
    assert "collect(DISTINCT ent_nb)[.." not in query


def test_graph_query_returns_single_chunks_and_falls_back_to_seed():
    query = _build_graph_retrieval_query(include_entity_neighbors=False, entity_neighbor_limit=0)

    assert "[] AS ent_hits" in query
    assert "[] AS rel_hits" in query
    assert "THEN [{chunk: node, strength: 0.0}] ELSE graph_hits END" in query
    assert "RETURN candidate.text AS text" in query
    assert "node.text +" not in query


def test_graph_route_fusion_deduplicates_evidence_and_preserves_routes():
    chunk = AI4RAGChunk(text="shared evidence")
    local_only = AI4RAGChunk(text="local evidence")

    fused = Neo4jGraphStore._fuse_graph_routes(
        {"vector": [(chunk, 0.9)], "local": [(AI4RAGChunk(text="shared evidence"), 0.8), (local_only, 0.7)]},
        k=2,
    )

    assert len(fused) == 2
    assert fused[0][0].text == "shared evidence"
    assert fused[0][0].metadata["routes"] == ["vector", "local"]


# ---------------------------------------------------------------------------
# Neo4jGraphStore.__init__
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestNeo4jGraphStoreInit:
    def test_creates_collection_specific_graph_vector_index(self, mock_driver_cls, mock_embedding, neo4j_config):
        Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        cypher_calls = [str(c) for c in session.run.call_args_list]
        joined = " ".join(cypher_calls)
        assert "VECTOR INDEX" in joined
        assert "ai4rag_col__embedding" in joined
        assert "(n:`ai4rag_col`)" in joined
        assert "(n:`ai4rag_col`:Chunk)" not in joined
        assert "FULLTEXT INDEX" not in joined

    def test_verifies_connectivity(self, mock_driver_cls, mock_embedding, neo4j_config):
        Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        mock_driver_cls.return_value.verify_connectivity.assert_called_once()

    def test_recycles_connections_before_load_balancer_idle_timeout(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        assert mock_driver_cls.call_args.kwargs["max_connection_lifetime"] == 30.0

    def test_collection_name_prefix_guard(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="ai4rag"):
            Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="bad_name")

    def test_reuses_supplied_collection_name(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_my_col")
        assert store.collection_name == "ai4rag_my_col"

    def test_generates_collection_name_when_none(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config)
        assert store.collection_name.startswith("ai4rag_")


# ---------------------------------------------------------------------------
# add_documents
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestAddDocuments:
    def _make_chunks(self, n=3, doc_id="doc1"):
        return [
            AI4RAGChunk(
                text=f"chunk {i}",
                metadata={"document_id": doc_id, "sequence_number": i},
            )
            for i in range(n)
        ]

    def test_empty_list_is_noop(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        mock_driver_cls.reset_mock()
        store.add_documents([])
        mock_driver_cls.return_value.session.assert_not_called()

    def test_merges_document_and_chunk_nodes(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(2)

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(MagicMock(), *args, **kwargs)

        store.add_documents(chunks)
        session.execute_write.assert_called()

    def test_next_chunk_links_are_created(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Consecutive chunks within one document must be linked with NEXT_CHUNK."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunks = self._make_chunks(3)

        tx = MagicMock()
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.add_documents(chunks)

        all_cypher = " ".join(str(c) for c in tx.run.call_args_list)
        assert "NEXT_CHUNK" in all_cypher

    def test_deduplicates_chunks(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunk = AI4RAGChunk(text="same text", metadata={"document_id": "d", "sequence_number": 0})
        duplicates = [chunk, chunk]

        tx = MagicMock()
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.add_documents(duplicates)
        # Only one MERGE for the Chunk node (after dedup)
        merge_chunk_calls = [c for c in tx.run.call_args_list if "Chunk" in str(c) and "MERGE (c:" in str(c)]
        assert len(merge_chunk_calls) == 1

    def test_pipeline_reuses_canonical_chunk_id(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        chunk = self._make_chunks(1)[0]
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        async def run_pipeline(*args, **kwargs):
            assert pipeline_cls.call_args.kwargs["kg_writer"]._chunk_id.get() == chunk.chunk_id
            return MagicMock()

        with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
            with patch("neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline") as pipeline_cls:
                pipeline_cls.return_value.run_async.side_effect = run_pipeline
                store.add_documents([chunk], model=MagicMock())

        assert isinstance(pipeline_cls.call_args.kwargs["kg_writer"], _CanonicalKGWriter)
        assert isinstance(pipeline_cls.call_args.kwargs["text_splitter"], _PreChunkedTextSplitter)
        assert pipeline_cls.call_args.kwargs["perform_entity_resolution"] is False
        lexical_config = pipeline_cls.call_args.kwargs["lexical_graph_config"]
        assert lexical_config.document_node_label != "Document"
        assert lexical_config.chunk_node_label != "Chunk"
        assert (
            pipeline_cls.return_value.run_async.call_args.kwargs["document_metadata"]["ai4rag_chunk_id"]
            == chunk.chunk_id
        )
        assert pipeline_cls.call_args.kwargs["kg_writer"]._chunk_id.get() is None
        assert not any("SET kc:ai4rag_col" in call.args[0] for call in session.run.call_args_list)
        resolution_call = next(
            call for call in session.run.call_args_list if "apoc.refactor.mergeNodes" in call.args[0]
        )
        resolution_query = resolution_call.args[0]
        assert "MATCH (entity:__Entity__)-[:FROM_CHUNK]->(:`ai4rag_col`:Chunk)" in resolution_query
        assert "WHERE entity.ai4rag_kg_collections = [$col]" in resolution_query
        assert "WITH entity_label, entity.name AS name" in resolution_query
        assert "mergeRels: false" in resolution_query
        assert "SET node.ai4rag_kg_collections = reduce(" in resolution_query
        assert resolution_call.kwargs["col"] == "ai4rag_col"

    def test_kg_pipeline_preserves_pre_chunked_text(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        text = "A long canonical chunk. " * 200
        chunk = AI4RAGChunk(text=text, metadata={"document_id": "document.md"})

        async def run_pipeline(*args, **kwargs):
            return MagicMock()

        with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
            with patch("neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline") as pipeline_cls:
                pipeline_cls.return_value.run_async.side_effect = run_pipeline
                store.add_documents([chunk], model=MagicMock())

        splitter = pipeline_cls.call_args.kwargs["text_splitter"]
        result = asyncio.run(splitter.run(text))
        assert len(result.chunks) == 1
        assert result.chunks[0].text == text
        assert result.chunks[0].index == 0
        assert pipeline_cls.return_value.run_async.call_args.kwargs["text"] == text

    def test_pipeline_can_skip_scoped_entity_resolution(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        async def run_pipeline(*args, **kwargs):
            return MagicMock()

        with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
            with patch("neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline") as pipeline_cls:
                pipeline_cls.return_value.run_async.side_effect = run_pipeline
                store._run_kg_pipeline([self._make_chunks(1)[0]], MagicMock(), perform_entity_resolution=False)

        assert pipeline_cls.call_args.kwargs["perform_entity_resolution"] is False
        assert not any("apoc.refactor.mergeNodes" in call.args[0] for call in session.run.call_args_list)


# ---------------------------------------------------------------------------
# search — vector mode
# ---------------------------------------------------------------------------

_CYPHER_RETRIEVER_PATH = "neo4j_graphrag.retrievers.VectorCypherRetriever"
_VECTOR_RETRIEVER_PATH = "neo4j_graphrag.retrievers.VectorRetriever"


@pytest.fixture(autouse=True)
def mock_vector_retriever():
    """Avoid GraphRAG's concrete-driver validation in Neo4j unit tests."""
    with patch(_VECTOR_RETRIEVER_PATH) as retriever:
        retriever.return_value.search.return_value = _make_retriever_result([])
        yield retriever


def _make_retriever_result(items):
    """Build a mock RetrieverResult with the given items."""
    result = MagicMock()
    result.items = items
    return result


def _make_retriever_item(text, score=0.9, meta=None):
    item = MagicMock()
    item.content = text
    item.metadata = {"score": score, "_meta": meta or {}}
    return item


# ---------------------------------------------------------------------------
# search — graph mode
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestSearchGraph:
    def test_uses_cypher_retriever_with_collection_index(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=2, search_mode="graph")

        _, init_kwargs = mock_cr_cls.call_args
        assert init_kwargs["index_name"] == "ai4rag_col__embedding"
        mock_cr_cls.return_value.search.assert_called_once_with(
            query_text="q", top_k=10, query_params={"col": "ai4rag_col"}
        )

    def test_retrieval_query_does_not_expand_sequential_neighbors(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph")

        _, init_kwargs = mock_cr_cls.call_args
        assert "NEXT_CHUNK" not in init_kwargs["retrieval_query"]
        assert "node.collection = $col" in init_kwargs["retrieval_query"]

    def test_graph_search_passes_collection_param(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_kg2")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph")

        mock_cr_cls.return_value.search.assert_called_once_with(
            query_text="q", top_k=10, query_params={"col": "ai4rag_kg2"}
        )

    def test_retrieval_query_contains_entity_expansion(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([])
            store.search("q", k=1, search_mode="graph", include_entity_neighbors=True)

        _, init_kwargs = mock_cr_cls.call_args
        assert "__Entity__" in init_kwargs["retrieval_query"]
        assert "FROM_CHUNK" in init_kwargs["retrieval_query"]
        assert "ent_nb.collection = $col" in init_kwargs["retrieval_query"]
        assert "NEXT_CHUNK" not in init_kwargs["retrieval_query"]

    def test_returns_chunks_from_graph_search(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        item = MagicMock()
        item.content = "seed text with context"
        item.metadata = {"score": 0.88}
        with patch(_CYPHER_RETRIEVER_PATH) as mock_cr_cls:
            mock_cr_cls.return_value.search.return_value = _make_retriever_result([item])
            results = store.search("q", k=1, search_mode="graph")

        assert len(results) == 1
        assert isinstance(results[0], AI4RAGChunk)
        assert results[0].text == "seed text with context"

    def test_vector_route_preserves_chunk_document_metadata(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        with patch(_VECTOR_RETRIEVER_PATH) as mock_vr_cls:
            mock_vr_cls.return_value.search.return_value = _make_retriever_result([])
            store._search_vector_route("q", k=1)

        formatter = mock_vr_cls.call_args.kwargs["result_formatter"]
        item = formatter(
            {
                "node": {
                    "text": "chunk text",
                    "document_id": "datasets/rag/william/william.md",
                    "metadata": json.dumps({"source": "datasets/rag/william/william.md", "sequence_number": 2}),
                },
                "score": 0.9,
            }
        )

        assert item.content == "chunk text"
        assert item.metadata["_meta"] == {
            "document_id": "datasets/rag/william/william.md",
            "source": "datasets/rag/william/william.md",
            "sequence_number": 2,
            "route": "vector",
        }

    def test_nonzero_graph_hops_raises(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", graph_hops=1)

    def test_invalid_entity_neighbor_limit_raises(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="entity_neighbor_limit"):
            _validate_neo4j_search_params("graph", entity_neighbor_limit=-1)


# ---------------------------------------------------------------------------
# build_knowledge_graph_from_documents
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestBuildKnowledgeGraphFromDocuments:
    """Docling input must become canonical chunks before KG extraction."""

    def _make_docling_doc(self, text="Document text content.", name="document.md"):
        doc = MagicMock()
        doc.export_to_markdown.return_value = text
        doc.name = name
        return doc

    def test_uses_one_document_id_and_canonical_chunks_per_input(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        model = MagicMock()
        doc1 = self._make_docling_doc("Text one.", "one.md")
        doc2 = self._make_docling_doc("Text two.", "two.md")

        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(documents=[doc1, doc2], model=model)

        doc1.export_to_markdown.assert_called_once()
        doc2.export_to_markdown.assert_called_once()
        chunks = add_documents.call_args.args[0]
        assert len(chunks) == 2
        assert [chunk.metadata["document_id"] for chunk in chunks] == ["one.md", "two.md"]
        assert all(isinstance(chunk, AI4RAGChunk) for chunk in chunks)
        assert add_documents.call_args.kwargs["model"] is model

    def test_empty_text_skips_pipeline(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        model = MagicMock()
        doc = self._make_docling_doc("")  # export produces empty string

        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(documents=[doc], model=model)

        add_documents.assert_not_called()

    def test_passes_extraction_options_to_shared_indexer(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(
                documents=[self._make_docling_doc()],
                model=MagicMock(),
                chunk_size=500,
                chunk_overlap=20,
                on_error="RAISE",
                perform_entity_resolution=False,
            )

        assert "kg_chunk_size" not in add_documents.call_args.kwargs
        assert "kg_chunk_overlap" not in add_documents.call_args.kwargs
        assert add_documents.call_args.kwargs["on_error"] == "RAISE"
        assert add_documents.call_args.kwargs["perform_entity_resolution"] is False

    def test_repeated_document_produces_the_same_chunk_ids(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        doc = self._make_docling_doc("Alice works at Acme.", "people.md")
        with patch.object(store, "add_documents") as add_documents:
            store.build_knowledge_graph_from_documents(documents=[doc], model=MagicMock())
            first_ids = [chunk.chunk_id for chunk in add_documents.call_args.args[0]]
            store.build_knowledge_graph_from_documents(documents=[doc], model=MagicMock())
            second_ids = [chunk.chunk_id for chunk in add_documents.call_args.args[0]]

        assert first_ids == second_ids

    def test_writes_one_document_and_chunk_set(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        tx = MagicMock()
        session.execute_write.side_effect = lambda fn, *args: fn(tx, *args)

        async def run_pipeline(*args, **kwargs):
            return MagicMock()

        with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
            with patch("neo4j_graphrag.experimental.pipeline.kg_builder.SimpleKGPipeline") as pipeline_cls:
                pipeline_cls.return_value.run_async.side_effect = run_pipeline
                store.build_knowledge_graph_from_documents(
                    documents=[self._make_docling_doc("Alice works at Acme.")], model=MagicMock()
                )

        queries = [call.args[0] for call in tx.run.call_args_list]
        assert sum("MERGE (d:ai4rag_col:Document" in query for query in queries) == 1
        assert sum("MERGE (c:ai4rag_col:Chunk" in query for query in queries) == 1
        assert not any("FROM_DOCUMENT" in query for query in queries)
        assert isinstance(pipeline_cls.call_args.kwargs["kg_writer"], _CanonicalKGWriter)


def test_collection_writer_tags_only_upserted_relationships():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CollectionKGWriter(driver, "neo4j", "ai4rag_col")
    writer._upsert_relationships([Neo4jRelationship(start_node_id="alice", end_node_id="acme", type="WORKS_AT")])

    query = driver.execute_query.call_args.args[0]
    parameters = driver.execute_query.call_args.kwargs["parameters_"]
    assert "CALL apoc.merge.relationship(start, row.type" in query
    assert "SET rel.ai4rag_kg_collections = CASE" in query
    assert "WHEN $col IN COALESCE(rel.ai4rag_kg_collections, [])" in query
    assert parameters["col"] == "ai4rag_col"
    assert len(parameters["rows"]) == 1


def test_collection_writer_setup_and_cleanup_use_configured_database():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CollectionKGWriter(driver, "tenant_graph", "ai4rag_col")

    result = asyncio.run(writer.run(Neo4jGraph()))

    assert result.status == "SUCCESS"
    setup_call = driver.execute_query.call_args
    assert "CREATE INDEX __entity__tmp_internal_id" in setup_call.args[0]
    assert setup_call.kwargs["database_"] == "tenant_graph"
    driver.session.assert_called_once_with(database="tenant_graph")
    cleanup_query = driver.session.return_value.__enter__.return_value.run.call_args.args[0]
    assert "MATCH (n:__KGBuilder__)" in cleanup_query
    assert "IN TRANSACTIONS" in cleanup_query


@pytest.mark.parametrize("serialized", [False, True])
def test_canonical_writer_skips_pipeline_document_and_chunk_nodes(serialized):
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "tenant_graph", "ai4rag_col")

    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(id="pipeline_doc", label="Document", properties={"ai4rag_chunk_id": "canonical_c1"}),
            Neo4jNode(id="pipeline_chunk", label="Chunk", properties={"text": "Alice works at Acme"}),
            Neo4jNode(id="alice", label="Person", properties={"name": "Alice"}),
            Neo4jNode(id="acme", label="Organization", properties={"name": "Acme"}),
        ],
        relationships=[
            Neo4jRelationship(start_node_id="pipeline_chunk", end_node_id="pipeline_doc", type="FROM_DOCUMENT"),
            Neo4jRelationship(start_node_id="alice", end_node_id="pipeline_chunk", type="FROM_CHUNK"),
            Neo4jRelationship(start_node_id="acme", end_node_id="pipeline_chunk", type="FROM_CHUNK"),
            Neo4jRelationship(start_node_id="alice", end_node_id="acme", type="WORKS_AT"),
        ],
    )

    result = asyncio.run(writer.run(graph.model_dump() if serialized else graph))

    assert result.status == "SUCCESS"
    calls = driver.execute_query.call_args_list
    node_rows = next(call.kwargs["parameters_"]["rows"] for call in calls if "CREATE (n:__KGBuilder__" in call.args[0])
    relation_rows = next(
        call.kwargs["parameters_"]["rows"] for call in calls if "apoc.merge.relationship" in call.args[0]
    )
    link_call = next(call for call in calls if "MERGE (e)-[:FROM_CHUNK]->(c)" in call.args[0])
    assert {row["label"] for row in node_rows} == {"Person", "Organization"}
    assert [row["type"] for row in relation_rows] == ["WORKS_AT"]
    assert link_call.kwargs["parameters_"] == {
        "entity_ids": ["acme", "alice"],
        "chunk_id": "canonical_c1",
        "col": "ai4rag_col",
    }
    assert all(call.kwargs["database_"] == "tenant_graph" for call in calls)
    driver.session.assert_called_once_with(database="tenant_graph")


def test_canonical_writer_keeps_entities_named_document_and_chunk():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "tenant_graph", "ai4rag_col")

    lexical_config = _KG_LEXICAL_GRAPH_CONFIG
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(
                id="pipeline_doc",
                label=lexical_config.document_node_label,
                properties={"ai4rag_chunk_id": "canonical_c1"},
            ),
            Neo4jNode(
                id="pipeline_chunk",
                label=lexical_config.chunk_node_label,
                properties={"text": "A document describes a chunk"},
            ),
            Neo4jNode(id="semantic_doc", label="Document", properties={"name": "Report"}),
            Neo4jNode(id="semantic_chunk", label="Chunk", properties={"name": "Chapter"}),
        ],
        relationships=[
            Neo4jRelationship(start_node_id="pipeline_chunk", end_node_id="pipeline_doc", type="FROM_DOCUMENT"),
            Neo4jRelationship(start_node_id="semantic_doc", end_node_id="semantic_chunk", type="CONTAINS"),
        ],
    )

    with patch("ai4rag.rag.vector_store.neo4j.logger") as mock_logger:
        result = asyncio.run(writer.run(graph.model_dump(), lexical_config))

    assert result.status == "SUCCESS"
    mock_logger.info.assert_called_once_with(
        "KG extraction for collection '%s', chunk '%s': entity types=%s, source document nodes=%d",
        "ai4rag_col",
        "canonical_c1",
        {"Chunk": 1, "Document": 1},
        1,
    )
    node_rows = next(
        call.kwargs["parameters_"]["rows"]
        for call in driver.execute_query.call_args_list
        if "CREATE (n:__KGBuilder__" in call.args[0]
    )
    assert {row["label"] for row in node_rows} == {"Document", "Chunk"}
    assert all("__Entity__" in row["labels"] for row in node_rows)
    relationship_rows = next(
        call.kwargs["parameters_"]["rows"]
        for call in driver.execute_query.call_args_list
        if "apoc.merge.relationship" in call.args[0]
    )
    assert [row["type"] for row in relationship_rows] == ["CONTAINS"]
    link_call = next(
        call for call in driver.execute_query.call_args_list if "MERGE (e)-[:FROM_CHUNK]->(c)" in call.args[0]
    )
    assert link_call.kwargs["parameters_"]["entity_ids"] == ["semantic_chunk", "semantic_doc"]


def test_canonical_writer_logs_entity_types_before_rejecting_multiple_source_documents():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "tenant_graph", "ai4rag_col")

    lexical_config = _KG_LEXICAL_GRAPH_CONFIG
    graph = Neo4jGraph(
        nodes=[
            Neo4jNode(id="source_1", label=lexical_config.document_node_label),
            Neo4jNode(id="source_2", label=lexical_config.document_node_label),
            Neo4jNode(id="report", label="Document", properties={"name": "Report"}),
        ]
    )

    with writer.for_chunk("canonical_c1"), patch("ai4rag.rag.vector_store.neo4j.logger") as mock_logger:
        with pytest.raises(ValueError, match="multiple document nodes"):
            asyncio.run(writer.run(graph, lexical_config))

    mock_logger.info.assert_called_once_with(
        "KG extraction for collection '%s', chunk '%s': entity types=%s, source document nodes=%d",
        "ai4rag_col",
        "canonical_c1",
        {"Document": 1},
        2,
    )
    driver.execute_query.assert_not_called()


def test_canonical_writer_requires_chunk_identity():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "neo4j", "ai4rag_col")
    graph = Neo4jGraph(nodes=[Neo4jNode(id="pipeline_doc", label="Document")])
    driver.execute_query.reset_mock()

    with pytest.raises(ValueError, match="canonical chunk ID"):
        asyncio.run(writer.run(graph))
    driver.execute_query.assert_not_called()


def test_canonical_writer_uses_bound_chunk_id_without_lexical_graph():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "tenant_graph", "ai4rag_col")
    graph = Neo4jGraph(nodes=[Neo4jNode(id="alice", label="Person", properties={"name": "Alice"})])

    with writer.for_chunk("canonical_c1"):
        result = asyncio.run(writer.run(graph))

    assert result.status == "SUCCESS"
    link_call = next(
        call for call in driver.execute_query.call_args_list if "MERGE (e)-[:FROM_CHUNK]->(c)" in call.args[0]
    )
    assert link_call.kwargs["parameters_"]["entity_ids"] == ["alice"]
    assert link_call.kwargs["parameters_"]["chunk_id"] == "canonical_c1"
    assert writer._chunk_id.get() is None


def test_canonical_writer_allows_empty_extraction_with_bound_chunk_id():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "tenant_graph", "ai4rag_col")

    with writer.for_chunk("canonical_c1"):
        result = asyncio.run(writer.run(Neo4jGraph()))

    assert result.status == "SUCCESS"
    assert not any("MERGE (e)-[:FROM_CHUNK]->(c)" in call.args[0] for call in driver.execute_query.call_args_list)


def test_canonical_writer_rejects_mismatched_document_chunk_id():
    driver = MagicMock()
    with patch("neo4j_graphrag.components.kg_writer.get_version", return_value=((5, 26, 0), False, False)):
        writer = _CanonicalKGWriter(driver, "neo4j", "ai4rag_col")
    graph = Neo4jGraph(nodes=[Neo4jNode(id="doc", label="Document", properties={"ai4rag_chunk_id": "other"})])
    driver.execute_query.reset_mock()

    with writer.for_chunk("canonical_c1"):
        with pytest.raises(ValueError, match="does not match"):
            asyncio.run(writer.run(graph))
    driver.execute_query.assert_not_called()


# ---------------------------------------------------------------------------
# clean_collection / close
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestCleanAndClose:
    def test_clean_collection_drops_legacy_indexes_without_dropping_shared_index(
        self, mock_driver_cls, mock_embedding, neo4j_config
    ):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.run.reset_mock()

        store.clean_collection()

        cypher_calls = " ".join(str(c) for c in session.run.call_args_list)
        assert "ai4rag_col__vector" in cypher_calls
        assert "ai4rag_col__fulltext" in cypher_calls
        assert "ai4rag_col__embedding" in cypher_calls
        assert "ai4rag_kg_collection" in cypher_calls
        assert "__Entity__" in cypher_calls
        assert "ai4rag_kg_collections" in cypher_calls
        assert "WHERE owner <> $col" in cypher_calls
        assert "MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)" in cypher_calls
        assert "COALESCE(other.collection, '') <> $col" in cypher_calls
        assert "WHERE NOT (e)-[:FROM_CHUNK]-(:Chunk)" not in cypher_calls
        assert "DETACH DELETE" in cypher_calls

        queries = [call.args[0] for call in session.run.call_args_list]
        legacy_cleanup = next(i for i, query in enumerate(queries) if "MATCH (e:__Entity__)-[:FROM_CHUNK]" in query)
        chunk_cleanup = next(i for i, query in enumerate(queries) if "MATCH (n:ai4rag_col)" in query)
        assert legacy_cleanup < chunk_cleanup
        assert any("size(r.ai4rag_kg_collections) = 1 DELETE r" in query for query in queries)

    def test_close_closes_driver(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        store.close()
        mock_driver_cls.return_value.close.assert_called_once()

    def test_context_manager_calls_close(self, mock_driver_cls, mock_embedding, neo4j_config):
        with Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col"):
            pass
        mock_driver_cls.return_value.close.assert_called_once()


# ---------------------------------------------------------------------------
# _validate_neo4j_search_params
# ---------------------------------------------------------------------------


class TestValidateNeo4jSearchParams:
    @pytest.mark.parametrize("search_mode", ["vector", "hybrid"])
    def test_non_graph_modes_rejected(self, search_mode):
        with pytest.raises(ValueError, match="only search_mode='graph'"):
            _validate_neo4j_search_params(search_mode)

    def test_graph_mode_valid(self):
        _validate_neo4j_search_params("graph")

    def test_graph_mode_valid_with_zero_hops(self):
        _validate_neo4j_search_params("graph", graph_hops=0)

    def test_graph_hops_nonzero_raises(self):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", graph_hops=1)

    def test_graph_hops_non_int_raises(self):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", graph_hops=1.5)

    def test_entity_neighbor_limit_negative_raises(self):
        with pytest.raises(ValueError, match="entity_neighbor_limit"):
            _validate_neo4j_search_params("graph", entity_neighbor_limit=-1)

    def test_entity_neighbor_limit_zero_is_valid(self):
        _validate_neo4j_search_params("graph", entity_neighbor_limit=0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="search_mode"):
            _validate_neo4j_search_params("unknown")
