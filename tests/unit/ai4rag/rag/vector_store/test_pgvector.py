# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import threading
import time
from dataclasses import replace
from unittest.mock import MagicMock, patch

import psycopg
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


def _conn_from(mock_pool_cls):
    """Return the connection mock yielded by ``with self._pool.connection() as conn:``.

    ``pool.connection()`` is a fixed mock, so every call within a test — across
    however many ``with`` blocks the store opens — resolves to this same object,
    letting tests assert against one accumulated ``conn.execute.call_args_list``.
    """
    pool = mock_pool_cls.return_value
    return pool.connection.return_value.__enter__.return_value


@patch("ai4rag.rag.vector_store.pgvector.ConnectionPool")
class TestPGVectorStoreInit:

    def test_creates_table_without_indexes_at_init(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config)

        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "CREATE TABLE" in executed
        # Indexes are deferred until the first search (build-after-load); they must NOT
        # be created at connection time, otherwise every insert pays HNSW maintenance.
        assert "hnsw" not in executed
        assert "USING gin" not in executed
        assert store.collection_name.startswith("ai4rag_")

    def test_reuses_collection_name(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_my_col")
        assert store.collection_name == "ai4rag_my_col"

    def test_invalid_distance_metric_raises(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        with pytest.raises(ValueError, match="Unsupported distance metric"):
            PGVectorStore(mock_embedding, pgvector_config, distance_metric="hamming")

    def test_embedding_dimension_over_limit_raises_before_pool_opens(self, mock_pool_cls, pgvector_config):
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

        # The guard must fire before any expensive work: no pool is opened.
        mock_pool_cls.assert_not_called()

    def test_embedding_dimension_at_limit_allowed(self, mock_pool_cls, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        class _LimitDimEmbedding:
            model_id = "limit-embedding"
            params = {"embedding_dimension": 2000}

            def embed_documents(self, texts):
                return [[0.1] * 2000 for _ in texts]

            def embed_query(self, query):
                return [0.1] * 2000

        conn = _conn_from(mock_pool_cls)
        PGVectorStore(_LimitDimEmbedding(), pgvector_config)

        # Exactly at the limit is valid: the table is created with a 2000-dim column.
        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "vector(2000)" in executed

    def test_password_passed_to_pool_kwargs(self, mock_pool_cls, mock_embedding):
        cfg = PGVectorConfig(host="h", port=5432, dbname="d", user="u", password="secret")
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        PGVectorStore(mock_embedding, cfg, collection_name="ai4rag_c")
        connect_kwargs = mock_pool_cls.call_args.kwargs["kwargs"]
        assert connect_kwargs["password"] == "secret"

    def test_no_password_skips_kwarg(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        connect_kwargs = mock_pool_cls.call_args.kwargs["kwargs"]
        assert "password" not in connect_kwargs

    def test_pool_sized_for_concurrent_search(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        pool_kwargs = mock_pool_cls.call_args.kwargs
        assert pool_kwargs["min_size"] == PGVectorStore._MIN_POOL_SIZE
        assert pool_kwargs["max_size"] == pgvector_config.pool_max_size
        assert pool_kwargs["configure"] == PGVectorStore._configure_connection

    def test_pool_max_size_follows_config(self, mock_pool_cls, mock_embedding, pgvector_config):
        """A caller-supplied ``pool_max_size`` — e.g. sized to its own query concurrency —
        must reach ``ConnectionPool`` verbatim, not the class's historical default."""
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        cfg = replace(pgvector_config, pool_max_size=25)
        PGVectorStore(mock_embedding, cfg, collection_name="ai4rag_c")
        pool_kwargs = mock_pool_cls.call_args.kwargs
        assert pool_kwargs["max_size"] == 25


class TestConfigureConnection:
    """Unit tests for the pool's per-connection setup, in isolation from ConnectionPool itself.

    ConnectionPool is a third-party dependency responsible for actually invoking
    ``configure`` on each connection it creates; that behaviour is its own
    library's concern, not ours to re-verify. These tests instead call
    ``_configure_connection`` directly to check that *our* setup logic is correct.
    """

    @patch("ai4rag.rag.vector_store.pgvector.register_vector")
    def test_registers_vector_adapter_and_ensures_extension(self, mock_register_vector):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        conn_execute_calls = []

        class _Conn:
            def execute(self, sql):
                conn_execute_calls.append(sql)

        conn = _Conn()
        PGVectorStore._configure_connection(conn)

        mock_register_vector.assert_called_once_with(conn)
        assert any("CREATE EXTENSION" in sql for sql in conn_execute_calls)


@patch("ai4rag.rag.vector_store.pgvector.ConnectionPool")
class TestPGVectorStoreSearch:

    def _make_store(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_indexes_built_lazily_on_first_search(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
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

    def test_ensure_indexes_is_thread_safe_under_concurrent_search(
        self, mock_pool_cls, mock_embedding, pgvector_config
    ):
        """search() is called from multiple threads at once (see query_rag's ThreadPoolExecutor).

        Regression test: without a lock around the ``_indexes_built`` check-then-act,
        concurrent threads all observe it as False and all race to run
        ``CREATE INDEX IF NOT EXISTS``, which PostgreSQL does not make atomic across
        sessions — the loser raises a real UniqueViolation instead of a silent no-op.
        """
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        def _slow_execute(*_args, **_kwargs):
            # A real CREATE INDEX/SELECT round-trip takes real time, during which the
            # GIL is released and other threads run. Sleeping here reproduces that
            # window against a mocked connection, which otherwise returns instantly and
            # never gives concurrent threads a chance to interleave on the race.
            time.sleep(0.02)
            result = MagicMock()
            result.fetchall.return_value = []
            return result

        conn.execute.side_effect = _slow_execute

        barrier = threading.Barrier(8)
        errors = []

        def _search():
            try:
                barrier.wait(timeout=5)
                store.search("query", k=1)
            except Exception as exc:  # noqa: BLE001 - capture for the main thread to assert on
                errors.append(exc)

        threads = [threading.Thread(target=_search) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert not errors, f"search() raised under concurrency: {errors}"

        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        # The DDL must have run exactly once in total, not once per thread.
        assert executed.count("CREATE INDEX IF NOT EXISTS") == 2  # one HNSW + one GIN statement

    def test_create_index_ignoring_race_swallows_unique_violation(self, mock_pool_cls, mock_embedding, pgvector_config):
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        conn = MagicMock()
        conn.execute.side_effect = psycopg.errors.UniqueViolation("already exists")

        store._create_index_ignoring_race(conn, "CREATE INDEX IF NOT EXISTS idx ...")  # must not raise

    def test_create_index_ignoring_race_reraises_other_errors(self, mock_pool_cls, mock_embedding, pgvector_config):
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        conn = MagicMock()
        conn.execute.side_effect = psycopg.OperationalError("connection dropped")

        with pytest.raises(psycopg.OperationalError):
            store._create_index_ignoring_race(conn, "CREATE INDEX IF NOT EXISTS idx ...")

    def test_vector_search(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        conn.execute.return_value.fetchall.return_value = [
            ("hello", {}, 0.5),
        ]

        results = store.search("query", k=1)
        assert len(results) == 1
        assert isinstance(results[0], AI4RAGChunk)
        assert results[0].text == "hello"

    def test_vector_search_with_scores(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        conn.execute.return_value.fetchall.return_value = [
            ("hello", {}, 0.5),
        ]

        results = store.search("query", k=1, include_scores=True)
        assert len(results) == 1
        chunk, score = results[0]
        assert chunk.text == "hello"
        assert score == pytest.approx(2.0)  # 1/0.5

    def test_vector_search_metadata_as_json_string_is_parsed(self, mock_pool_cls, mock_embedding, pgvector_config):
        """Defensive fallback: some drivers/paths may hand back the JSONB column as raw text."""
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        conn.execute.return_value.fetchall.return_value = [
            ("hello", '{"document_id": "d1"}', 0.5),
        ]

        results = store.search("query", k=1)
        assert results[0].metadata == {"document_id": "d1"}

    def test_vector_search_null_metadata_defaults_to_empty_dict(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        conn.execute.return_value.fetchall.return_value = [("hello", None, 0.5)]

        results = store.search("query", k=1)
        assert results[0].metadata == {}

    def test_inner_product_scores_preserve_ranking(self, mock_pool_cls, mock_embedding):
        """The ``<#>`` operator returns the *negative* inner product (a signed value):
        more similar rows have a more negative distance. A ``1 / distance`` transform
        would be non-monotonic and invert that ranking, so the store must negate the
        distance instead, restoring "higher score = more relevant".
        """
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        cfg = PGVectorConfig(host="localhost", port=5432, dbname="testdb", user="testuser")
        conn = _conn_from(mock_pool_cls)
        store = PGVectorStore(mock_embedding, cfg, distance_metric="inner_product", collection_name="ai4rag_ip")

        # Rows arrive already ordered by distance ASC: -0.9 (most similar) before -0.2.
        conn.execute.return_value.fetchall.return_value = [
            ("closer", {}, -0.9),
            ("farther", {}, -0.2),
        ]

        results = store.search("query", k=2, include_scores=True)
        assert [chunk.text for chunk, _ in results] == ["closer", "farther"]
        # Negating recovers the plain inner product: 0.9 > 0.2, ranking preserved.
        assert results[0][1] == pytest.approx(0.9)
        assert results[1][1] == pytest.approx(0.2)
        assert results[0][1] > results[1][1]


@patch("ai4rag.rag.vector_store.pgvector.ConnectionPool")
class TestPGVectorStoreValidation:

    def _make_store(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_invalid_search_mode(self, mock_pool_cls, mock_embedding, pgvector_config):
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="Invalid search_mode"):
            store.search("q", k=1, search_mode="full_text")

    def test_ranker_strategy_on_vector_mode(self, mock_pool_cls, mock_embedding, pgvector_config):
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="only valid when search_mode='hybrid'"):
            store.search("q", k=1, search_mode="vector", ranker_strategy="rrf")

    def test_hybrid_without_strategy(self, mock_pool_cls, mock_embedding, pgvector_config):
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="ranker_strategy must be set"):
            store.search("q", k=1, search_mode="hybrid")


@patch("ai4rag.rag.vector_store.pgvector.ConnectionPool")
class TestPGVectorStoreAddDocuments:

    def _make_store(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_add_documents(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        docs = [AI4RAGChunk(text="hello", metadata={"document_id": "d1"})]
        store.add_documents(docs)

        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.executemany.assert_called_once()
        batch = cursor.executemany.call_args[0][1]
        chunk_id, metadata_json, embedding, content_text, tokenize_text = batch[0]
        assert chunk_id == docs[0].chunk_id
        assert content_text == "hello"
        assert tokenize_text == "hello"
        assert json.loads(metadata_json) == {"document_id": "d1"}

    def test_add_empty_documents(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        store.add_documents([])
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.executemany.assert_not_called()

    def test_deduplicates_by_chunk_id(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        chunk = AI4RAGChunk(text="same text", metadata={"document_id": "d1"})
        store.add_documents([chunk, chunk])

        cursor = conn.cursor.return_value.__enter__.return_value
        call_args = cursor.executemany.call_args
        assert len(call_args[0][1]) == 1

    def test_insert_retries_once_on_operational_error(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        cursor = conn.cursor.return_value.__enter__.return_value
        # First attempt hits a dropped backend; the pool hands out a working
        # connection (still the same mock here) on the retry.
        cursor.executemany.side_effect = [psycopg.OperationalError("server closed the connection"), None]

        store.add_documents([AI4RAGChunk(text="hello", metadata={"document_id": "d1"})])

        assert cursor.executemany.call_count == 2  # failed once, retried once

    def test_insert_does_not_mask_deterministic_failure(self, mock_pool_cls, mock_embedding, pgvector_config):
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)
        conn = _conn_from(mock_pool_cls)
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.executemany.side_effect = psycopg.OperationalError("always fails")

        with pytest.raises(psycopg.OperationalError):
            store.add_documents([AI4RAGChunk(text="hello", metadata={"document_id": "d1"})])

        assert cursor.executemany.call_count == 2  # one retry, then the error surfaces

    def test_custom_batch_size_splits_inserts(self, mock_pool_cls, mock_embedding, pgvector_config):
        conn = _conn_from(mock_pool_cls)
        store = self._make_store(mock_pool_cls, mock_embedding, pgvector_config)

        docs = [AI4RAGChunk(text=f"doc {i}", metadata={"document_id": f"d{i}"}) for i in range(5)]
        store.add_documents(docs, batch_size=2)

        cursor = conn.cursor.return_value.__enter__.return_value
        assert cursor.executemany.call_count == 3  # 2 + 2 + 1


@patch("ai4rag.rag.vector_store.pgvector.ConnectionPool")
class TestPGVectorStoreCleanAndClose:

    def test_clean_collection(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        conn = _conn_from(mock_pool_cls)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_to_drop")

        store.clean_collection()
        drop_calls = [c for c in conn.execute.call_args_list if "DROP TABLE" in str(c)]
        assert len(drop_calls) == 1

    def test_close(self, mock_pool_cls, mock_embedding, pgvector_config):
        from ai4rag.rag.vector_store.pgvector import PGVectorStore

        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        pool = mock_pool_cls.return_value

        store.close()
        pool.close.assert_called_once()
