# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import asyncio
import threading
from dataclasses import replace
from unittest.mock import AsyncMock, MagicMock, patch

import asyncpg
import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import PGVectorConfig
from ai4rag.rag.vector_store.pgvector import PGVectorStore


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


class _HighDimEmbedding:
    """Mock embedding model above pgvector's 2000-dimension HNSW index limit."""

    model_id = "big-embedding"
    params = {"embedding_dimension": 3072}

    def embed_documents(self, texts):
        return [[0.1] * 3072 for _ in texts]

    def embed_query(self, query):
        return [0.1] * 3072


@pytest.fixture
def pgvector_config():
    return PGVectorConfig(host="localhost", port=5432, dbname="testdb", user="testuser")


@pytest.fixture(autouse=True)
def _close_stores_after_test():
    """Track every PGVectorStore created in a test and close() it on teardown.

    Unlike the old psycopg_pool-based store — entirely replaced by the
    ``mock_pool_cls`` patch below, so opening it never touched real resources —
    this store opens a *real* background thread and event loop the moment a
    test triggers its first DB access (``asyncpg.create_pool`` is mocked, but
    the loop/thread machinery around it lives in ``PGVectorStore`` itself, not
    in asyncpg). Without closing each instance, every test that touches the DB
    would leak one daemon thread for the rest of the process.
    """
    created: list[PGVectorStore] = []
    original_init = PGVectorStore.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    with patch.object(PGVectorStore, "__init__", _tracking_init):
        yield
    for store in created:
        store.close()


def _conn_from(mock_create_pool):
    """Return the connection mock yielded by ``async with pool.acquire() as conn:``.

    ``pool`` (``mock_create_pool.return_value``) is itself an ``AsyncMock`` —
    every descendant attribute of an ``AsyncMock`` defaults to ``AsyncMock`` too,
    so ``pool.acquire`` would otherwise be async and ``pool.acquire()`` would
    return a bare coroutine instead of the awaitable *and* async-context-manager
    object the real ``asyncpg.Pool.acquire()`` returns. Resetting it to a plain
    ``MagicMock`` restores that: ``__aenter__``/``__aexit__`` are auto-async on
    any ``MagicMock`` regardless, which is all ``async with pool.acquire() as
    conn:`` needs. ``pool.acquire()`` is then a fixed mock, so every call within
    a test — across however many ``async with`` blocks the store opens —
    resolves to this same ``conn``, letting tests assert against one
    accumulated ``conn.execute.call_args_list``. ``execute``/``fetch``/
    ``executemany`` are likewise reset to plain ``AsyncMock`` (not inherited,
    ordinary named attributes) so calling them returns an awaitable coroutine.
    """
    pool = mock_create_pool.return_value
    pool.acquire = MagicMock()
    conn = pool.acquire.return_value.__aenter__.return_value
    conn.execute = AsyncMock()
    conn.fetch = AsyncMock()
    conn.executemany = AsyncMock()
    return conn


@patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool", new_callable=AsyncMock)
class TestPGVectorStoreInit:

    def test_creates_table_on_first_db_access_not_at_init(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config)

        # Pool and table creation are deferred until the first DB operation so
        # that embed_documents() runs without any idle connections open.
        executed_at_init = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "CREATE TABLE" not in executed_at_init
        assert mock_create_pool.call_count == 0

        # First DB access (e.g. add_documents) triggers pool open + table creation.
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        executed_after = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "CREATE TABLE" in executed_after
        # Indexes are deferred until the first search (build-after-load); they must NOT
        # be created at connection time, otherwise every insert pays HNSW maintenance.
        assert "hnsw" not in executed_after
        assert "USING gin" not in executed_after
        assert store.collection_name.startswith("ai4rag_")

    def test_reuses_collection_name(self, mock_create_pool, mock_embedding, pgvector_config):
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_my_col")
        assert store.collection_name == "ai4rag_my_col"

    def test_invalid_distance_metric_raises(self, mock_create_pool, mock_embedding, pgvector_config):
        with pytest.raises(ValueError, match="Unsupported distance metric"):
            PGVectorStore(mock_embedding, pgvector_config, distance_metric="hamming")

    def test_embedding_dimension_over_limit_logs_warning_and_proceeds(self, mock_create_pool, pgvector_config, caplog):
        conn = _conn_from(mock_create_pool)

        with caplog.at_level("WARNING"):
            store = PGVectorStore(_HighDimEmbedding(), pgvector_config)

        # Warning is emitted at construction time; pool + table are still deferred.
        assert store.collection_name.startswith("ai4rag_")
        assert any("exceeds pgvector's" in record.message for record in caplog.records)
        mock_create_pool.assert_not_called()

        # Trigger DB access; the table DDL must use the actual dimension.
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        mock_create_pool.assert_called_once()
        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "vector(3072)" in executed

    def test_embedding_dimension_at_limit_allowed(self, mock_create_pool, pgvector_config):
        class _LimitDimEmbedding:
            model_id = "limit-embedding"
            params = {"embedding_dimension": 2000}

            def embed_documents(self, texts):
                return [[0.1] * 2000 for _ in texts]

            def embed_query(self, query):
                return [0.1] * 2000

        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(_LimitDimEmbedding(), pgvector_config)
        # Trigger DB access to materialise the table DDL.
        store.add_documents([AI4RAGChunk(text="x", metadata={})])

        # Exactly at the limit is valid: the table is created with a 2000-dim column.
        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "vector(2000)" in executed

    def test_password_passed_to_pool_kwargs(self, mock_create_pool, mock_embedding):
        cfg = PGVectorConfig(host="h", port=5432, dbname="d", user="u", password="secret")
        _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, cfg, collection_name="ai4rag_c")
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        connect_kwargs = mock_create_pool.call_args.kwargs
        assert connect_kwargs["password"] == "secret"

    def test_no_password_skips_kwarg(self, mock_create_pool, mock_embedding, pgvector_config):
        _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        connect_kwargs = mock_create_pool.call_args.kwargs
        assert "password" not in connect_kwargs

    def test_pool_sized_for_concurrent_search(self, mock_create_pool, mock_embedding, pgvector_config):
        _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        pool_kwargs = mock_create_pool.call_args.kwargs
        assert pool_kwargs["min_size"] == PGVectorStore._MIN_POOL_SIZE
        assert pool_kwargs["max_size"] == pgvector_config.pool_max_size
        assert pool_kwargs["init"] == PGVectorStore._configure_connection

    def test_pool_max_size_follows_config(self, mock_create_pool, mock_embedding, pgvector_config):
        """A caller-supplied ``pool_max_size`` — e.g. sized to its own query concurrency —
        must reach ``create_pool`` verbatim, not the class's historical default."""
        _conn_from(mock_create_pool)
        cfg = replace(pgvector_config, pool_max_size=25)
        store = PGVectorStore(mock_embedding, cfg, collection_name="ai4rag_c")
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        pool_kwargs = mock_create_pool.call_args.kwargs
        assert pool_kwargs["max_size"] == 25


class TestConfigureConnection:
    """Unit tests for the pool's per-connection setup, in isolation from asyncpg's pool itself.

    asyncpg's ``Pool`` is a third-party dependency responsible for actually invoking
    ``init`` on each connection it creates; that behaviour is its own library's
    concern, not ours to re-verify. These tests instead call
    ``_configure_connection`` directly to check that *our* setup logic is correct.
    """

    @patch("ai4rag.rag.vector_store.pgvector.register_vector", new_callable=AsyncMock)
    def test_registers_vector_adapter_and_ensures_extension(self, mock_register_vector):
        conn = MagicMock()
        conn.execute = AsyncMock()
        conn.set_type_codec = AsyncMock()

        asyncio.run(PGVectorStore._configure_connection(conn))

        # The extension must be created before register_vector() looks up its OID.
        create_extension_call = conn.execute.call_args_list[0]
        assert "CREATE EXTENSION" in create_extension_call.args[0]
        mock_register_vector.assert_called_once_with(conn)
        conn.set_type_codec.assert_called_once()
        assert conn.set_type_codec.call_args.args[0] == "jsonb"


@patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool", new_callable=AsyncMock)
class TestPGVectorStoreSearch:

    def _make_store(self, mock_create_pool, mock_embedding, pgvector_config):
        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_indexes_built_lazily_on_first_search(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        conn.fetch.return_value = []

        # No index DDL has run yet after construction.
        assert "hnsw" not in " ".join(str(c) for c in conn.execute.call_args_list)

        store.search("query", k=1)

        after_search = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "hnsw" in after_search
        assert "USING gin" in after_search

        # A second search must not re-issue the index DDL (guarded by _indexes_built).
        conn.execute.reset_mock()
        store.search("query", k=1)
        assert "hnsw" not in " ".join(str(c) for c in conn.execute.call_args_list)

    def test_high_dimension_search_skips_hnsw_builds_gin(self, mock_create_pool, pgvector_config):
        """Above pgvector's 2000-dim limit, HNSW is skipped but GIN and search still work."""
        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(_HighDimEmbedding(), pgvector_config, collection_name="ai4rag_highdim")
        conn.fetch.return_value = [("hello", {}, 0.5)]

        results = store.search("query", k=1)

        executed = " ".join(str(c) for c in conn.execute.call_args_list)
        assert "hnsw" not in executed
        assert "USING gin" in executed
        assert results[0].text == "hello"

    def test_high_dimension_hybrid_search_still_fuses(self, mock_create_pool, pgvector_config):
        """Hybrid fusion is independent of the embedding dimension/index tier."""
        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(_HighDimEmbedding(), pgvector_config, collection_name="ai4rag_highdim_hybrid")
        conn.fetch.return_value = [("hello", {}, 0.5)]

        results = store.search("query", k=1, search_mode="hybrid", ranker_strategy="rrf")

        assert len(results) == 1
        assert results[0].text == "hello"

    def test_ensure_indexes_is_thread_safe_under_concurrent_search(
        self, mock_create_pool, mock_embedding, pgvector_config
    ):
        """search() is called from multiple threads at once (see query_rag's ThreadPoolExecutor).

        Regression test: without a lock around the ``_indexes_built`` check-then-act,
        concurrent threads all observe it as False and all race to run
        ``CREATE INDEX IF NOT EXISTS``, which PostgreSQL does not make atomic across
        sessions — the loser raises a real UniqueViolationError instead of a silent
        no-op.
        """
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        async def _slow_execute(*_args, **_kwargs):
            # A real CREATE INDEX/SELECT round-trip takes real time, during which
            # other coroutines on the shared loop run. Sleeping here reproduces that
            # window against a mocked connection, which otherwise returns instantly and
            # never gives concurrent threads a chance to interleave on the race.
            await asyncio.sleep(0.02)
            return "CREATE INDEX"

        conn.execute.side_effect = _slow_execute
        conn.fetch.return_value = []

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

    def test_create_index_ignoring_race_swallows_unique_violation(
        self, mock_create_pool, mock_embedding, pgvector_config
    ):
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=asyncpg.exceptions.UniqueViolationError("already exists"))

        asyncio.run(store._create_index_ignoring_race(conn, "CREATE INDEX IF NOT EXISTS idx ..."))  # must not raise

    def test_create_index_ignoring_race_reraises_other_errors(self, mock_create_pool, mock_embedding, pgvector_config):
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        conn = MagicMock()
        conn.execute = AsyncMock(side_effect=asyncpg.exceptions.PostgresConnectionError("connection dropped"))

        with pytest.raises(asyncpg.exceptions.PostgresConnectionError):
            asyncio.run(store._create_index_ignoring_race(conn, "CREATE INDEX IF NOT EXISTS idx ..."))

    def test_vector_search(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        conn.fetch.return_value = [("hello", {}, 0.5)]

        results = store.search("query", k=1)
        assert len(results) == 1
        assert isinstance(results[0], AI4RAGChunk)
        assert results[0].text == "hello"

    def test_vector_search_with_scores(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        conn.fetch.return_value = [("hello", {}, 0.5)]

        results = store.search("query", k=1, include_scores=True)
        assert len(results) == 1
        chunk, score = results[0]
        assert chunk.text == "hello"
        assert score == pytest.approx(2.0)  # 1/0.5

    def test_vector_search_null_metadata_defaults_to_empty_dict(
        self, mock_create_pool, mock_embedding, pgvector_config
    ):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        conn.fetch.return_value = [("hello", None, 0.5)]

        results = store.search("query", k=1)
        assert results[0].metadata == {}

    def test_inner_product_scores_preserve_ranking(self, mock_create_pool, mock_embedding):
        """The ``<#>`` operator returns the *negative* inner product (a signed value):
        more similar rows have a more negative distance. A ``1 / distance`` transform
        would be non-monotonic and invert that ranking, so the store must negate the
        distance instead, restoring "higher score = more relevant".
        """
        cfg = PGVectorConfig(host="localhost", port=5432, dbname="testdb", user="testuser")
        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, cfg, distance_metric="inner_product", collection_name="ai4rag_ip")

        # Rows arrive already ordered by distance ASC: -0.9 (most similar) before -0.2.
        conn.fetch.return_value = [
            ("closer", {}, -0.9),
            ("farther", {}, -0.2),
        ]

        results = store.search("query", k=2, include_scores=True)
        assert [chunk.text for chunk, _ in results] == ["closer", "farther"]
        # Negating recovers the plain inner product: 0.9 > 0.2, ranking preserved.
        assert results[0][1] == pytest.approx(0.9)
        assert results[1][1] == pytest.approx(0.2)
        assert results[0][1] > results[1][1]


@patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool", new_callable=AsyncMock)
class TestPGVectorStoreValidation:

    def _make_store(self, mock_create_pool, mock_embedding, pgvector_config):
        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_invalid_search_mode(self, mock_create_pool, mock_embedding, pgvector_config):
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="Invalid search_mode"):
            store.search("q", k=1, search_mode="full_text")

    def test_ranker_strategy_on_vector_mode(self, mock_create_pool, mock_embedding, pgvector_config):
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="only valid when search_mode='hybrid'"):
            store.search("q", k=1, search_mode="vector", ranker_strategy="rrf")

    def test_hybrid_without_strategy(self, mock_create_pool, mock_embedding, pgvector_config):
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        with pytest.raises(ValueError, match="ranker_strategy must be set"):
            store.search("q", k=1, search_mode="hybrid")


@patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool", new_callable=AsyncMock)
class TestPGVectorStoreAddDocuments:

    def _make_store(self, mock_create_pool, mock_embedding, pgvector_config):
        return PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_test_col")

    def test_add_documents(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        docs = [AI4RAGChunk(text="hello", metadata={"document_id": "d1"})]
        store.add_documents(docs)

        conn.executemany.assert_called_once()
        batch = conn.executemany.call_args.args[1]
        chunk_id, metadata, embedding, content_text, tokenize_text = batch[0]
        assert chunk_id == docs[0].chunk_id
        assert content_text == "hello"
        assert tokenize_text == "hello"
        assert metadata == {"document_id": "d1"}

    def test_add_empty_documents(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        store.add_documents([])
        conn.executemany.assert_not_called()

    def test_deduplicates_by_chunk_id(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        chunk = AI4RAGChunk(text="same text", metadata={"document_id": "d1"})
        store.add_documents([chunk, chunk])

        call_args = conn.executemany.call_args
        assert len(call_args.args[1]) == 1

    def test_insert_retries_once_on_operational_error(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        # First attempt hits a dropped backend; the pool hands out a working
        # connection (still the same mock here) on the retry.
        conn.executemany.side_effect = [
            asyncpg.exceptions.PostgresConnectionError("server closed the connection"),
            None,
        ]

        store.add_documents([AI4RAGChunk(text="hello", metadata={"document_id": "d1"})])

        assert conn.executemany.call_count == 2  # failed once, retried once

    def test_insert_does_not_mask_deterministic_failure(self, mock_create_pool, mock_embedding, pgvector_config):
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)
        conn = _conn_from(mock_create_pool)
        conn.executemany.side_effect = asyncpg.exceptions.PostgresConnectionError("always fails")

        with pytest.raises(asyncpg.exceptions.PostgresConnectionError):
            store.add_documents([AI4RAGChunk(text="hello", metadata={"document_id": "d1"})])

        assert conn.executemany.call_count == 2  # one retry, then the error surfaces

    def test_custom_batch_size_splits_inserts(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = self._make_store(mock_create_pool, mock_embedding, pgvector_config)

        docs = [AI4RAGChunk(text=f"doc {i}", metadata={"document_id": f"d{i}"}) for i in range(5)]
        store.add_documents(docs, batch_size=2)

        assert conn.executemany.call_count == 3  # 2 + 2 + 1


@patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool", new_callable=AsyncMock)
class TestPGVectorStoreCleanAndClose:

    def test_clean_collection(self, mock_create_pool, mock_embedding, pgvector_config):
        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_to_drop")

        store.clean_collection()
        all_sql = " ".join(str(c) for c in conn.execute.call_args_list)
        # clean_collection uses _ensure_pool (not _ensure_db), so it must NOT
        # issue a spurious CREATE TABLE before the DROP.
        assert "CREATE TABLE" not in all_sql
        drop_calls = [c for c in conn.execute.call_args_list if "DROP TABLE" in str(c)]
        assert len(drop_calls) == 1

    def test_close(self, mock_create_pool, mock_embedding, pgvector_config):
        _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        pool = mock_create_pool.return_value
        pool.close = AsyncMock()

        store.close()
        pool.close.assert_called_once()
        assert store._db is None

    def test_close_before_any_db_access_is_safe(self, mock_create_pool, mock_embedding, pgvector_config):
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        store.close()  # pool was never opened — must not raise
        mock_create_pool.return_value.close.assert_not_called()

    def test_close_is_idempotent(self, mock_create_pool, mock_embedding, pgvector_config):
        """A second close() must not hang trying to dispatch onto the already-stopped loop."""
        _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        store.add_documents([AI4RAGChunk(text="x", metadata={})])
        pool = mock_create_pool.return_value
        pool.close = AsyncMock()

        store.close()
        store.close()  # must return immediately, not raise or hang
        pool.close.assert_called_once()


@patch("ai4rag.rag.vector_store.pgvector.asyncpg.create_pool", new_callable=AsyncMock)
class TestPGVectorStoreConcurrentInit:
    """Verify that concurrent first DB accesses initialise pool and table exactly once."""

    def test_pool_opened_exactly_once_under_concurrent_access(self, mock_create_pool, mock_embedding, pgvector_config):
        _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def _trigger() -> None:
            try:
                barrier.wait()
                store._ensure_db()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_trigger) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent _ensure_db raised: {errors}"
        assert mock_create_pool.call_count == 1, "Pool must be opened exactly once"

    def test_table_created_exactly_once_under_concurrent_access(
        self, mock_create_pool, mock_embedding, pgvector_config
    ):
        conn = _conn_from(mock_create_pool)
        store = PGVectorStore(mock_embedding, pgvector_config, collection_name="ai4rag_c")
        barrier = threading.Barrier(8)
        errors: list[Exception] = []

        def _trigger() -> None:
            try:
                barrier.wait()
                store._ensure_db()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_trigger) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        create_calls = [c for c in conn.execute.call_args_list if "CREATE TABLE" in str(c)]
        assert len(create_calls) == 1, "Table must be created exactly once"
