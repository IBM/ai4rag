# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import os
from unittest.mock import MagicMock, call, patch

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j import Neo4jGraphStore, _parse_kg_extraction, _validate_neo4j_search_params


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


# ---------------------------------------------------------------------------
# Neo4jGraphStore.__init__
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestNeo4jGraphStoreInit:
    def test_creates_vector_index(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        cypher_calls = [str(c) for c in session.run.call_args_list]
        joined = " ".join(cypher_calls)
        assert "VECTOR INDEX" in joined
        assert "FULLTEXT INDEX" not in joined

    def test_verifies_connectivity(self, mock_driver_cls, mock_embedding, neo4j_config):
        Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        mock_driver_cls.return_value.verify_connectivity.assert_called_once()

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
            AI4RAGChunk(text=f"chunk {i}", metadata={"document_id": doc_id, "sequence_number": i})
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
        merge_chunk_calls = [
            c for c in tx.run.call_args_list if "Chunk" in str(c) and "MERGE (c:" in str(c)
        ]
        assert len(merge_chunk_calls) == 1


# ---------------------------------------------------------------------------
# search — vector mode
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestSearchVector:
    def _mock_vector_result(self, session, rows):
        result = MagicMock()
        result.data.return_value = rows
        session.run.return_value = result

    def _make_row(self, text="hello", score=0.9):
        return {"id": "abc", "text": text, "metadata": json.dumps({"seq": 1}), "score": score}

    def test_calls_vector_index_query(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        self._mock_vector_result(session, [self._make_row()])

        store.search("query", k=1)

        cypher_calls = " ".join(str(c) for c in session.run.call_args_list)
        assert "db.index.vector.queryNodes" in cypher_calls

    def test_returns_chunks_without_scores(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        self._mock_vector_result(session, [self._make_row("text A")])

        results = store.search("query", k=1)

        assert isinstance(results, list)
        assert all(isinstance(r, AI4RAGChunk) for r in results)

    def test_returns_chunks_with_scores(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        self._mock_vector_result(session, [self._make_row(score=0.85)])

        results = store.search("query", k=1, include_scores=True)

        assert isinstance(results[0], tuple)
        chunk, score = results[0]
        assert isinstance(chunk, AI4RAGChunk)
        assert abs(score - 0.85) < 1e-6

    def test_graph_mode_rejected_by_milvus(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Milvus must raise ValueError for search_mode='graph' — validated here via the shared check."""
        from ai4rag.rag.vector_store.milvus import MilvusVectorStore

        with pytest.raises(ValueError, match="not supported by MilvusVectorStore"):
            with patch("ai4rag.rag.vector_store.milvus.MilvusClient"):
                from ai4rag.rag.vector_store.config import MilvusConfig
                milvus_cfg = MilvusConfig(uri="http://localhost:19530")
                store = MilvusVectorStore(mock_embedding, milvus_cfg, collection_name="ai4rag_col")
                store.search("q", k=1, search_mode="graph")

    def test_graph_mode_rejected_by_pgvector(self, mock_driver_cls, mock_embedding, neo4j_config):
        """PGVector must raise ValueError for search_mode='graph'."""
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        with pytest.raises(ValueError, match="not supported by PGVectorStore"):
            with patch("ai4rag.rag.vector_store.pgvector.ConnectionPool"):
                from ai4rag.rag.vector_store.config import PGVectorConfig
                pg_cfg = PGVectorConfig(host="localhost")
                store = PGVectorStore(mock_embedding, pg_cfg, collection_name="ai4rag_col")
                store.search("q", k=1, search_mode="graph")


# ---------------------------------------------------------------------------
# search — graph mode
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestSearchGraph:
    def _make_row(self, text="seed", score=0.9, chunk_id=None):
        meta = json.dumps({"seq": 0})
        return {"id": chunk_id or "seed_id", "text": text, "metadata": meta, "score": score}

    def test_issues_sequential_neighbor_query(self, mock_driver_cls, mock_embedding, neo4j_config):
        seed_row = self._make_row("seed text", 0.9, "seed_id")

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        schema_r = MagicMock()
        vector_r = MagicMock()
        vector_r.data.return_value = [seed_row]
        neighbor_r = MagicMock()
        neighbor_r.data.return_value = []
        entity_r = MagicMock()
        entity_r.data.return_value = []

        session.run.side_effect = [schema_r, vector_r, neighbor_r, entity_r]

        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        store.search("q", k=1, search_mode="graph")

        cypher_calls = " ".join(str(c) for c in session.run.call_args_list)
        assert "NEXT_CHUNK" in cypher_calls

    def test_returns_seeds_when_no_neighbors(self, mock_driver_cls, mock_embedding, neo4j_config):
        seed_row = self._make_row("seed", 0.9, "seed_id")

        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        schema_r = MagicMock()
        vector_r = MagicMock()
        vector_r.data.return_value = [seed_row]
        empty_r = MagicMock()
        empty_r.data.return_value = []

        session.run.side_effect = [schema_r, vector_r, empty_r, empty_r]

        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        results = store.search("q", k=5, search_mode="graph")

        assert len(results) == 1
        assert results[0].text == "seed"

    def test_invalid_graph_hops_raises(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", None, None, None, graph_hops=0)

    def test_invalid_entity_neighbor_limit_raises(self, mock_driver_cls, mock_embedding, neo4j_config):
        with pytest.raises(ValueError, match="entity_neighbor_limit"):
            _validate_neo4j_search_params("graph", None, None, None, entity_neighbor_limit=-1)


# ---------------------------------------------------------------------------
# build_knowledge_graph
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestBuildKnowledgeGraph:
    def _make_model(self, response_json):
        model = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps(response_json)
        model.chat.return_value = [choice]
        return model

    def _node_id_payload(self):
        return _make_extraction_payload(
            nodes=[
                {"id": "0", "label": "Person", "properties": {"name": "Alice", "chunk_id": "c1", "description": ""}},
                {"id": "1", "label": "Organization", "properties": {"name": "Acme", "chunk_id": "c1", "description": ""}},
            ],
            relationships=[
                {"type": "WORKS_AT", "start_node_id": "0", "end_node_id": "1", "properties": {"description": ""}},
            ],
        )

    def test_calls_llm_for_each_batch(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        chunk_data = [{"id": "c1", "text": "Alice works at Acme."}, {"id": "c2", "text": "Bob is CEO."}]
        page_result = MagicMock()
        page_result.data.return_value = chunk_data
        empty_page = MagicMock()
        empty_page.data.return_value = []
        session.run.side_effect = [page_result, empty_page]

        model = self._make_model(self._node_id_payload())
        tx = MagicMock()
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.build_knowledge_graph(model, chunk_batch_size=16)

        model.chat.assert_called_once()
        call_kwargs = model.chat.call_args.kwargs
        assert "max_completion_tokens" in call_kwargs

    def test_writes_entity_and_mentions_nodes(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        chunk_data = [{"id": "c1", "text": "Alice works at Acme."}]
        page_result = MagicMock()
        page_result.data.return_value = chunk_data
        empty_page = MagicMock()
        empty_page.data.return_value = []
        session.run.side_effect = [page_result, empty_page]

        model = self._make_model(self._node_id_payload())
        tx = MagicMock()
        session.execute_write.side_effect = lambda fn, *args, **kwargs: fn(tx, *args, **kwargs)

        store.build_knowledge_graph(model, chunk_batch_size=16)

        all_cypher = " ".join(str(c) for c in tx.run.call_args_list)
        assert "Entity" in all_cypher
        assert "MENTIONS" in all_cypher

    def test_no_chunks_skips_llm(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        empty_page = MagicMock()
        empty_page.data.return_value = []
        session.run.return_value = empty_page

        model = MagicMock()
        store.build_knowledge_graph(model)
        model.chat.assert_not_called()

    def test_invalid_json_response_is_skipped(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value

        chunk_data = [{"id": "c1", "text": "some text"}]
        page_result = MagicMock()
        page_result.data.return_value = chunk_data
        empty_page = MagicMock()
        empty_page.data.return_value = []
        session.run.side_effect = [page_result, empty_page]

        model = MagicMock()
        choice = MagicMock()
        choice.message.content = "not valid json !!!"
        model.chat.return_value = [choice]

        # Must not raise
        store.build_knowledge_graph(model)


# ---------------------------------------------------------------------------
# _parse_kg_extraction
# ---------------------------------------------------------------------------


def _make_extraction_payload(nodes, relationships=None):
    """Build a node-ID-format extraction payload for tests."""
    return {"nodes": nodes, "relationships": relationships or []}


class TestParseKgExtraction:
    def test_valid_json(self):
        raw = json.dumps(_make_extraction_payload(
            nodes=[
                {"id": "0", "label": "Person", "properties": {"name": "Alice", "description": "A researcher."}},
                {"id": "1", "label": "Organization", "properties": {"name": "Acme", "description": "A company."}},
            ],
            relationships=[
                {"type": "WORKS_AT", "start_node_id": "0", "end_node_id": "1", "properties": {"description": "Alice works at Acme."}},
            ],
        ))
        ents, rels = _parse_kg_extraction(raw)
        assert len(ents) == 2
        assert ents[0]["name"] == "Alice"
        assert len(rels) == 1
        assert rels[0]["source"] == "Alice"
        assert rels[0]["target"] == "Acme"
        assert rels[0]["keywords"] == "WORKS_AT"

    def test_strips_markdown_fences(self):
        raw = "```json\n" + json.dumps({"nodes": [], "relationships": []}) + "\n```"
        ents, rels = _parse_kg_extraction(raw)
        assert ents == []
        assert rels == []

    def test_invalid_json_repaired(self):
        # json_repair should recover a truncated but partially valid object.
        ents, rels = _parse_kg_extraction('{"nodes": [], "relationships": []')
        assert ents == []
        assert rels == []

    def test_garbage_returns_empty(self):
        ents, rels = _parse_kg_extraction("not json at all!!! ???")
        assert ents == []
        assert rels == []

    def test_nodes_without_name_are_skipped(self):
        raw = json.dumps(_make_extraction_payload(
            nodes=[{"id": "0", "label": "Person", "properties": {}}],
        ))
        ents, _ = _parse_kg_extraction(raw)
        assert ents == []

    def test_relationships_with_unknown_node_id_are_skipped(self):
        raw = json.dumps(_make_extraction_payload(
            nodes=[{"id": "0", "label": "Person", "properties": {"name": "Alice"}}],
            relationships=[{"type": "KNOWS", "start_node_id": "0", "end_node_id": "99"}],
        ))
        _, rels = _parse_kg_extraction(raw)
        assert rels == []

    def test_missing_optional_fields_default(self):
        raw = json.dumps(_make_extraction_payload(
            nodes=[{"id": "0", "label": "Person", "properties": {"name": "Alice"}}],
        ))
        ents, _ = _parse_kg_extraction(raw)
        assert ents[0]["type"] == "Person"
        assert ents[0]["description"] == ""


# ---------------------------------------------------------------------------
# build_knowledge_graph_from_documents
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestBuildKnowledgeGraphFromDocuments:
    def _make_model(self, payload):
        model = MagicMock()
        choice = MagicMock()
        choice.message.content = json.dumps(payload)
        model.chat.return_value = [choice]
        return model

    def _make_docling_doc(self, name="doc1"):
        doc = MagicMock()
        doc.name = name
        return doc

    def _make_chunks(self, n=1):
        return [
            AI4RAGChunk(text=f"Chunk {i} text.", metadata={"document_id": "d1", "sequence_number": i})
            for i in range(n)
        ]

    def test_calls_add_documents_then_llm(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *a, **kw: fn(MagicMock(), *a, **kw)

        chunks = self._make_chunks(1)
        payload = _make_extraction_payload(
            nodes=[{"id": "0", "label": "Person", "properties": {"name": "Alice", "description": "A person."}}],
        )
        model = self._make_model(payload)

        with patch("ai4rag.rag.vector_store.neo4j.DoclingChunker") as mock_chunker_cls:
            mock_chunker_cls.return_value.split_documents.return_value = chunks
            store.build_knowledge_graph_from_documents(
                documents=[self._make_docling_doc()],
                model=model,
            )

        model.chat.assert_called_once()
        messages = model.chat.call_args.args[0]
        roles = [m["role"] for m in messages]
        assert "system" in roles

    def test_uses_provided_chunker(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *a, **kw: fn(MagicMock(), *a, **kw)

        chunks = self._make_chunks(1)
        model = self._make_model(_make_extraction_payload([]))
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = chunks

        store.build_knowledge_graph_from_documents(
            documents=[self._make_docling_doc()],
            model=model,
            chunker=mock_chunker,
        )

        mock_chunker.split_documents.assert_called_once()

    def test_empty_chunks_skips_llm(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        model = MagicMock()
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = []

        store.build_knowledge_graph_from_documents(
            documents=[self._make_docling_doc()],
            model=model,
            chunker=mock_chunker,
        )

        model.chat.assert_not_called()

    def test_bad_llm_response_does_not_raise(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *a, **kw: fn(MagicMock(), *a, **kw)

        chunks = self._make_chunks(1)
        model = MagicMock()
        choice = MagicMock()
        choice.message.content = "not json"
        model.chat.return_value = [choice]
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = chunks

        # Must not raise
        store.build_knowledge_graph_from_documents(
            documents=[self._make_docling_doc()],
            model=model,
            chunker=mock_chunker,
        )

    def test_no_db_readback(self, mock_driver_cls, mock_embedding, neo4j_config):
        """Extraction must not issue paginated MATCH queries against the DB."""
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.execute_write.side_effect = lambda fn, *a, **kw: fn(MagicMock(), *a, **kw)

        chunks = self._make_chunks(1)
        model = self._make_model(_make_extraction_payload([]))
        mock_chunker = MagicMock()
        mock_chunker.split_documents.return_value = chunks

        store.build_knowledge_graph_from_documents(
            documents=[self._make_docling_doc()],
            model=model,
            chunker=mock_chunker,
        )

        skip_calls = [c for c in session.run.call_args_list if "SKIP" in str(c)]
        assert skip_calls == []


# ---------------------------------------------------------------------------
# clean_collection / close
# ---------------------------------------------------------------------------


@patch("ai4rag.rag.vector_store.neo4j.neo4j.GraphDatabase.driver")
class TestCleanAndClose:
    def test_clean_collection_drops_indexes_and_deletes_nodes(self, mock_driver_cls, mock_embedding, neo4j_config):
        store = Neo4jGraphStore(mock_embedding, neo4j_config, collection_name="ai4rag_col")
        session = mock_driver_cls.return_value.session.return_value.__enter__.return_value
        session.run.reset_mock()

        store.clean_collection()

        cypher_calls = " ".join(str(c) for c in session.run.call_args_list)
        assert "DROP INDEX" in cypher_calls
        assert "DETACH DELETE" in cypher_calls

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
    def test_vector_mode_valid(self):
        _validate_neo4j_search_params("vector", None, None, None)

    def test_vector_mode_rejects_ranker_strategy(self):
        with pytest.raises(ValueError, match="ranker parameters"):
            _validate_neo4j_search_params("vector", "rrf", None, None)

    def test_vector_mode_rejects_ranker_k(self):
        with pytest.raises(ValueError, match="ranker parameters"):
            _validate_neo4j_search_params("vector", None, 60, None)

    def test_hybrid_mode_rejected(self):
        with pytest.raises(ValueError, match="search_mode"):
            _validate_neo4j_search_params("hybrid", "rrf", None, None)

    def test_graph_mode_valid(self):
        _validate_neo4j_search_params("graph", None, None, None)

    def test_graph_mode_valid_with_rrf(self):
        _validate_neo4j_search_params("graph", "rrf", 60, None, graph_hops=2)

    def test_graph_mode_invalid_ranker_strategy_raises(self):
        with pytest.raises(ValueError, match="ranker_strategy"):
            _validate_neo4j_search_params("graph", "bad_strategy", None, None)

    def test_graph_mode_ranker_k_without_rrf_raises(self):
        with pytest.raises(ValueError, match="ranker_k"):
            _validate_neo4j_search_params("graph", "weighted", 60, None)

    def test_graph_mode_ranker_alpha_without_weighted_raises(self):
        with pytest.raises(ValueError, match="ranker_alpha"):
            _validate_neo4j_search_params("graph", "rrf", None, 0.5)

    def test_graph_hops_zero_raises(self):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", None, None, None, graph_hops=0)

    def test_graph_hops_non_int_raises(self):
        with pytest.raises(ValueError, match="graph_hops"):
            _validate_neo4j_search_params("graph", None, None, None, graph_hops=1.5)

    def test_entity_neighbor_limit_negative_raises(self):
        with pytest.raises(ValueError, match="entity_neighbor_limit"):
            _validate_neo4j_search_params("graph", None, None, None, entity_neighbor_limit=-1)

    def test_entity_neighbor_limit_zero_is_valid(self):
        _validate_neo4j_search_params("graph", None, None, None, entity_neighbor_limit=0)

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="search_mode"):
            _validate_neo4j_search_params("unknown", None, None, None)
