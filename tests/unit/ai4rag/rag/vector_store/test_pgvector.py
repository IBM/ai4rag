# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock, call, patch

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import PGVectorConfig


class _MockEmbeddingModel:
    """Minimal mock for BaseEmbeddingModel."""

    model_id = "test-embedding"
    params = {"embedding_dimension": 128}

    def embed_documents(self, texts):
        return [[0.1] * 128 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 128


@pytest.fixture
def mock_embedding():
    return _MockEmbeddingModel()


@pytest.fixture
def pgvector_config():
    return PGVectorConfig(host="localhost", port=5432, dbname="testdb", user="testuser")


@patch("ai4rag.rag.vector_store.pgvector.register_vector")
@patch("ai4rag.rag.vector_store.pgvector.psycopg.connect")
class TestPGVectorStoreInit:

    def test_creates_table_without_indexes_at_init(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        conn = mock_connect.return_value
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config)

        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "CREATE EXTENSION" in executed
        assert "CREATE TABLE" in executed
        # Indexes are deferred until the first search (build-after-load); they must NOT
        # be created at connection time, otherwise every insert pays HNSW maintenance.
        assert "hnsw" not in executed
        assert "USING gin" not in executed
        assert store.collection_name.startswith("ai4rag_")

    def test_reuses_collection_name(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        conn = mock_connect.return_value
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_my_col")
        assert store.collection_name == "ai4rag_my_col"

    def test_invalid_distance_metric_raises(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        with pytest.raises(ValueError, match="Unsupported distance metric"):
            PGVectorStore(mock_embedding, pgvector_config, distance_metric="hamming")

    def test_embedding_dimension_over_limit_raises_before_connect(self, mock_connect, mock_reg, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        class _HighDimEmbedding:
            model_id = "big-embedding"
            params = {"embedding_dimension": 3072}

            def embed_documents(self, texts):
                return [[0.1] * 3072 for _ in texts]

            def embed_query(self, query):
                return [0.1] * 3072

        with pytest.raises(ValueError, match="exceeds pgvector's"):
            PGVectorStore(_HighDimEmbedding(), pgvector_config)

        # The guard must fire before any expensive work: no connection is opened.
        mock_connect.assert_not_called()

    def test_embedding_dimension_at_limit_allowed(self, mock_connect, mock_reg, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        class _LimitDimEmbedding:
            model_id = "limit-embedding"
            params = {"embedding_dimension": 2000}

            def embed_documents(self, texts):
                return [[0.1] * 2000 for _ in texts]

            def embed_query(self, query):
                return [0.1] * 2000

        PGVectorStore(_LimitDimEmbedding(), pgvector_config)

        # Exactly at the limit is valid: the table is created with a 2000-dim column.
        executed = " ".join(str(c) for c in mock_connect.return_value.execute.call_args_list)
        assert "vector(2000)" in executed

    def test_password_passed_to_connect(self, mock_connect, mock_reg, mock_embedding):
        cfg = PGVectorConfig(host="h", port=5432, dbname="d", user="u", password="secret")
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        PGVectorStore(mock_embedding, cfg, collection_name="ai4rag_c")
        connect_kwargs = mock_connect.call_args[1]
        assert connect_kwargs["password"] == "secret"

    def test_no_password_skips_kwarg(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        connect_kwargs = mock_connect.call_args[1]
        assert "password" not in connect_kwargs


@patch("ai4rag.rag.vector_store.pgvector.register_vector")
@patch("ai4rag.rag.vector_store.pgvector.psycopg.connect")
class TestPGVectorStoreSearch:

    def _make_store(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        mock_connect.return_value.closed = False
        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_indexes_built_lazily_on_first_search(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value
        conn.execute.return_value.fetchall.return_value = []

        # No index DDL has run yet after construction.
        assert "hnsw" not in " ".join(str(c) for c in conn.execute.call_args_list)

        store.search("query", k=1)

        after_search = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "hnsw" in after_search
        assert "USING gin" in after_search

        # A second search must not re-issue the index DDL (guarded by _indexes_built).
        conn.execute.reset_mock()
        conn.execute.return_value.fetchall.return_value = []
        store.search("query", k=1)
        assert "hnsw" not in " ".join(str(c) for c in conn.execute.call_args_list)

    def test_vector_search(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value

        conn.execute.return_value.fetchall.return_value = [
            ({"content": "hello", "metadata": {}, "chunk_id": "c1"}, 0.5),
        ]

        results = store.search("query", k=1)
        assert len(results) == 1
        assert isinstance(results[0], AI4RAGChunk)
        assert results[0].text == "hello"

    def test_vector_search_with_scores(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value

        conn.execute.return_value.fetchall.return_value = [
            ({"content": "hello", "metadata": {}, "chunk_id": "c1"}, 0.5),
        ]

        results = store.search("query", k=1, include_scores=True)
        assert len(results) == 1
        chunk, score = results[0]
        assert chunk.text == "hello"
        assert score == pytest.approx(2.0)  # 1/0.5


@patch("ai4rag.rag.vector_store.pgvector.register_vector")
@patch("ai4rag.rag.vector_store.pgvector.psycopg.connect")
class TestPGVectorStoreValidation:

    def _make_store(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        mock_connect.return_value.closed = False
        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_invalid_search_mode(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="Invalid search_mode"):
            store.search("q", k=1, search_mode="full_text")

    def test_ranker_strategy_on_vector_mode(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="only valid when search_mode='hybrid'"):
            store.search("q", k=1, search_mode="vector", ranker_strategy="rrf")

    def test_hybrid_without_strategy(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="ranker_strategy must be set"):
            store.search("q", k=1, search_mode="hybrid")


@patch("ai4rag.rag.vector_store.pgvector.register_vector")
@patch("ai4rag.rag.vector_store.pgvector.psycopg.connect")
class TestPGVectorStoreAddDocuments:

    def _make_store(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        mock_connect.return_value.closed = False
        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_add_documents(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value

        docs = [AI4RAGChunk(text="hello", metadata={"document_id": "d1"})]
        store.add_documents(docs)

        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.executemany.assert_called_once()

    def test_add_empty_documents(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value

        store.add_documents([])
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.executemany.assert_not_called()

    def test_deduplicates_by_chunk_id(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value

        chunk = AI4RAGChunk(text="same text", metadata={"document_id": "d1"})
        store.add_documents([chunk, chunk])

        cursor = conn.cursor.return_value.__enter__.return_value
        call_args = cursor.executemany.call_args
        assert len(call_args[0][1]) == 1

    def test_insert_reconnects_and_retries_on_operational_error(
        self, mock_connect, mock_reg, mock_embedding, pgvector_config
    ):
        import psycopg

        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value
        cursor = conn.cursor.return_value.__enter__.return_value
        # First attempt hits a dropped backend, second (post-reconnect) succeeds.
        cursor.executemany.side_effect = [psycopg.OperationalError("server closed the connection"), None]

        connect_calls_before = mock_connect.call_count
        store.add_documents([AI4RAGChunk(text="hello", metadata={"document_id": "d1"})])

        assert cursor.executemany.call_count == 2  # failed once, retried once
        assert mock_connect.call_count == connect_calls_before + 1  # reconnected exactly once

    def test_custom_batch_size_splits_inserts(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        store = self._make_store(mock_connect, mock_reg, mock_embedding, pgvector_config)
        conn = mock_connect.return_value

        docs = [AI4RAGChunk(text=f"doc {i}", metadata={"document_id": f"d{i}"}) for i in range(5)]
        store.add_documents(docs, batch_size=2)

        cursor = conn.cursor.return_value.__enter__.return_value
        assert cursor.executemany.call_count == 3  # 2 + 2 + 1


@patch("ai4rag.rag.vector_store.pgvector.register_vector")
@patch("ai4rag.rag.vector_store.pgvector.psycopg.connect")
class TestPGVectorStoreCleanAndClose:

    def test_clean_collection(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_to_drop")
        conn = mock_connect.return_value

        store.clean_collection()
        drop_calls = [c for c in conn.execute.call_args_list if "DROP TABLE" in str(c)]
        assert len(drop_calls) == 1

    def test_close(self, mock_connect, mock_reg, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        conn = mock_connect.return_value
        conn.closed = False

        store.close()
        conn.close.assert_called_once()
