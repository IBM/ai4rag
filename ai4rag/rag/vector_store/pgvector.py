# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import asyncio
import heapq
import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import asyncpg
from pgvector.asyncpg import register_vector

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import PGVectorConfig
from ai4rag.rag.vector_store.reranker import WeightedInMemoryAggregator
from ai4rag.rag.vector_store.utils import iter_unique_chunks, resolve_embedding_dimension, validate_search_params

__all__ = ["PGVectorStore"]


@dataclass(frozen=True)
class _LoopBoundPool:
    """A pool bundled with the background event loop/thread it is bound to.

    asyncpg's ``Pool``/``Connection`` objects are only usable from the event
    loop that created them, so these three always come into being together (in
    :meth:`PGVectorStore._open_pool_sync`) and are torn down together (in
    :meth:`PGVectorStore.close`); grouping them keeps that invariant obvious
    and avoids three separately-nullable attributes on the store.
    """

    pool: asyncpg.Pool
    loop: asyncio.AbstractEventLoop
    thread: threading.Thread


class _InFlightTracker:
    """Tracks calls dispatched via :meth:`PGVectorStore._run`, so :meth:`PGVectorStore.close`
    can wait for them to drain before tearing down the event loop.

    ``cond`` and ``count`` are always read and mutated together — one under the
    other — so, like :class:`_LoopBoundPool`, they are bundled into a single
    object rather than two separate attributes on the store.
    """

    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.count = 0


class PGVectorStore(BaseVectorStore):
    """Vector store backed by PostgreSQL with the ``pgvector`` extension.

    Supports pure vector search and hybrid search (dense vector + tsvector
    full-text) with RRF or weighted reranking via in-memory fusion.

    Driven by ``asyncpg`` rather than ``psycopg``: asyncpg speaks the Postgres
    wire protocol itself instead of wrapping the ``libpq`` C library, so it
    ships as a normal self-contained wheel with no system ``libpq`` dependency
    and none of ``psycopg[binary]``'s bundled-OpenSSL conflicts. Every public
    method here stays synchronous (matching :class:`BaseVectorStore` and every
    other backend) by dispatching onto one dedicated background event loop —
    see :meth:`_run` — so callers never need to know asyncpg is involved.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    config : PGVectorConfig
        Connection parameters for the PostgreSQL server.
    distance_metric : str
        Distance metric (default ``"cosine"``). One of ``"cosine"``,
        ``"l2"``, ``"l1"``, ``"inner_product"``.
    collection_name : str | None
        Existing collection to reuse; must start with the ``ai4rag`` prefix. The
        name is used verbatim as the PostgreSQL table name. When omitted, a new
        compliant name is generated (see
        :func:`ai4rag.rag.vector_store.utils.resolve_collection_name`).
    """

    _BATCH_SIZE = 1024
    _CONNECT_TIMEOUT = 10

    # asyncpg has no equivalent of libpq's `keepalives_*` socket options (see the
    # note on `_open_pool`), so a connection silently killed by an idle middlebox
    # would otherwise surface as a hang bounded only by the OS's TCP retransmission
    # timeout (minutes, not seconds) instead of a prompt, retryable error. Applied
    # to both the acquire and the query on each hot-path call (see
    # `_fetch_vector_rows`, `_fetch_keyword_rows`, `_insert_batch_async`) — never to
    # the one-time DDL in `_create_table`/`_build_indexes`/`_drop_table`, which can
    # legitimately take longer on a large collection and must not be cut short.
    # Passing it to `pool.acquire()` too (not just the query itself) matters: asyncpg
    # reuses that same budget for the connection's *release-time* cleanup (resetting
    # session state, confirming a cancelled query) — without it, that cleanup step
    # has no timeout of its own and could hang indefinitely on a truly dead
    # connection, undoing the whole point of bounding the query.
    _COMMAND_TIMEOUT = 90.0

    # Shorter than asyncpg's own 300s default: proactively recycling idle pooled
    # connections more often shrinks (but, absent a real keepalive, cannot fully
    # close) the window in which a connection can be silently dropped while sitting
    # idle in the pool between calls.
    _MAX_INACTIVE_CONNECTION_LIFETIME = 60.0

    # search() and add_documents() may be called concurrently across threads (e.g.
    # one worker per benchmark question in query_rag(), or one per concurrent
    # request in a deployed service), and a single shared connection is not safe
    # for concurrent use. The pool starts at _MIN_POOL_SIZE and grows lazily up to
    # the caller-supplied config.pool_max_size, so a fully concurrent caller never
    # queues for a slot as long as pool_max_size covers its own concurrency (see
    # PGVectorConfig.pool_max_size). asyncpg's pool lives on one event loop shared
    # by every caller thread (see _run), which is exactly what lets those threads'
    # DB calls interleave concurrently instead of serializing on the loop.
    _MIN_POOL_SIZE = 1

    # pgvector caps HNSW (and IVFFlat) indexes on the ``vector`` type at 2000
    # dimensions (https://github.com/pgvector/pgvector#hnsw). Higher-dimensional
    # vectors still store and query correctly — only the ANN index cannot be built.
    # Rather than rejecting oversized models outright, ``_ensure_indexes`` skips
    # building the HNSW index above this threshold and lets PostgreSQL fall back to
    # an exact sequential scan: slower per query, but still fully correct (in fact
    # exact rather than HNSW's approximate result), and every other code path
    # (storage, keyword search, hybrid fusion) is unaffected by the dimension.
    _MAX_INDEXABLE_DIMENSION = 2000

    _DISTANCE_METRIC_TO_OPERATOR: dict[str, str] = {
        "cosine": "<=>",
        "l2": "<->",
        "l1": "<+>",
        "inner_product": "<#>",
    }

    _DISTANCE_METRIC_TO_INDEX_OPS: dict[str, str] = {
        "cosine": "vector_cosine_ops",
        "l2": "vector_l2_ops",
        "l1": "vector_l1_ops",
        "inner_product": "vector_ip_ops",
    }

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: PGVectorConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
    ):
        """Validate parameters and prepare the store for use.

        Resolves the distance metric to its pgvector operator and index opclass.
        The connection pool and backing table are created lazily on the first DB
        access (see :meth:`_ensure_db`) so that :meth:`add_documents` can embed
        documents — the slow step — before any idle connection is opened.
        HNSW and GIN indexes are deferred further, to the first search (see
        :meth:`_ensure_indexes`), avoiding per-row HNSW maintenance during bulk
        inserts.

        Parameters
        ----------
        embedding_model : BaseEmbeddingModel
            Model used to embed documents and queries.
        config : PGVectorConfig
            Connection parameters for the PostgreSQL server.
        distance_metric : str, default="cosine"
            Distance metric. One of ``"cosine"``, ``"l2"``, ``"l1"``,
            ``"inner_product"``.
        collection_name : str | None, default=None
            Existing collection to reuse; must start with the ``ai4rag`` prefix
            and is used verbatim as the table name. When omitted, a new compliant
            name is generated.

        Raises
        ------
        ValueError
            If ``distance_metric`` is not one of the supported metrics.
        """
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(self.embedding_model)
        if self._embedding_dimension > self._MAX_INDEXABLE_DIMENSION:
            logger.warning(
                "Embedding dimension %d exceeds pgvector's %d-dimension limit for HNSW "
                "indexes; searches on collection %r will use an exact sequential scan "
                "instead of an approximate nearest-neighbor index.",
                self._embedding_dimension,
                self._MAX_INDEXABLE_DIMENSION,
                self._collection_name,
            )

        distance_key = distance_metric.lower()
        if distance_key not in self._DISTANCE_METRIC_TO_OPERATOR:
            raise ValueError(
                f"Unsupported distance metric '{distance_metric}'. "
                f"Must be one of {list(self._DISTANCE_METRIC_TO_OPERATOR)}."
            )
        self._distance_key = distance_key
        self._distance_operator = self._DISTANCE_METRIC_TO_OPERATOR[distance_key]
        self._index_ops = self._DISTANCE_METRIC_TO_INDEX_OPS[distance_key]

        # Indexes are built lazily after documents are loaded (see ``_ensure_indexes``),
        # not at connection time: maintaining an HNSW graph on every insert is the
        # memory-heavy path that can trigger the server-side OOM killer on large batches.
        # search() is called concurrently across threads (see config.pool_max_size above),
        # so the flag guarding this one-time DDL needs a lock, not just a bare check.
        self._indexes_built = False
        self._indexes_lock = threading.Lock()

        # Pool and table are created lazily on first DB access so that
        # add_documents() can embed documents (the slow step) before any
        # connection is opened. _db_lock guards one-time loop/pool/table init.
        self._db: _LoopBoundPool | None = None
        self._table_created: bool = False
        self._db_lock = threading.Lock()

        # Guards the handoff between _run() (reader) and close() (writer): close()
        # must not stop/close the loop while another thread's _run() call is still
        # dispatched on it. See _run() and close() for the drain protocol this
        # implements.
        self._inflight = _InFlightTracker()

    def _run(self, coro: Any) -> Any:
        """Run *coro* on the store's dedicated event loop and block for its result.

        asyncpg's ``Pool``/``Connection`` objects are bound to the event loop that
        created them and are not safe to drive from arbitrary threads directly, so
        every DB-touching coroutine is dispatched here instead of via
        ``asyncio.run()`` (which would tear the pool down again after each call).
        Multiple caller threads (e.g. one per question in ``query_rag()``) can call
        this concurrently: each dispatches onto the same loop, where asyncpg's pool
        multiplexes them across its connections exactly as it did across threads
        with ``psycopg_pool.ConnectionPool``.

        Must only be called from a thread other than the loop's own — calling it
        from a coroutine already running on the loop would deadlock, since that
        thread would be blocked waiting for a result the loop can never produce
        while it is blocked. Every async helper method in this class receives an
        already-opened ``pool``/``conn`` as an argument rather than calling
        :meth:`_ensure_db`/:meth:`_ensure_pool` itself, so that lookup — which
        calls :meth:`_run` — always happens on the caller's thread.

        Safe to call concurrently with :meth:`close`: this method and ``close()``
        coordinate over :attr:`_inflight` so that a store closed mid-call raises
        a clear :class:`RuntimeError` (never an ``AttributeError`` from a
        half-torn-down ``self._db``), and ``close()`` blocks until every
        in-flight :meth:`_run` call it can see has finished before stopping the
        loop out from under it.

        Parameters
        ----------
        coro : Any
            The coroutine to run.

        Returns
        -------
        Any
            The coroutine's result.

        Raises
        ------
        RuntimeError
            If the store has already been closed via :meth:`close`.
        """
        with self._inflight.cond:
            db = self._db
            if db is None:
                coro.close()  # avoid a "coroutine was never awaited" warning
                raise RuntimeError(f"PGVectorStore for collection {self._collection_name!r} is closed.")
            self._inflight.count += 1
        try:
            return self._dispatch(db.loop, coro)
        finally:
            with self._inflight.cond:
                self._inflight.count -= 1
                if self._inflight.count == 0:
                    self._inflight.cond.notify_all()

    def _run_with_retry(self, make_coro: Callable[[], Any]) -> Any:
        """Run a coroutine on the store's event loop, retrying once on a dropped connection.

        *make_coro* is a zero-argument callable rather than a coroutine because a
        coroutine object can only be awaited once: a retry needs a fresh one. The
        pool discards a connection it finds broken and hands out a fresh one on
        the next borrow, so the retry itself needs no explicit reconnect. This
        recovers from a *transient* drop (recycled backend, dropped idle
        connection); it deliberately does not mask a deterministic failure — an
        operation that always kills the backend still surfaces after one retry.

        Parameters
        ----------
        make_coro : Callable[[], Any]
            Builds a fresh coroutine to run; called once, and again on retry.

        Returns
        -------
        Any
            The coroutine's result.
        """
        try:
            return self._run(make_coro())
        except asyncpg.exceptions.PostgresConnectionError as exc:
            logger.warning("PGVector operation failed (%s); retrying once.", exc)
            return self._run(make_coro())

    @staticmethod
    def _dispatch(loop: asyncio.AbstractEventLoop, coro: Any) -> Any:
        """Run *coro* on *loop* from another thread and block for its result.

        Bare ``asyncio.run_coroutine_threadsafe(...).result()`` factored out so
        :meth:`close` can dispatch the pool's shutdown coroutine onto a loop it
        already holds a direct reference to, after :attr:`_db` has been cleared
        and :meth:`_run` is no longer usable.
        """
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def _ensure_pool(self) -> asyncpg.Pool:
        """Return the pool, opening it (and its background event loop) on the first call.

        Use this when only a connection is needed and the table is not required
        (e.g. :meth:`clean_collection`, which drops the table unconditionally).
        All other callers should use :meth:`_ensure_db`.

        Thread-safe via double-checked locking.
        """
        if self._db is not None:
            return self._db.pool
        with self._db_lock:
            if self._db is None:
                self._db = self._open_pool_sync()
        return self._db.pool

    def _open_pool_sync(self) -> _LoopBoundPool:
        """Start the background event loop and open the pool on it.

        asyncpg's ``Pool``/``Connection`` objects are bound to the loop that
        creates them, so this store keeps one dedicated event loop alive on a
        background thread for its whole lifetime (see :meth:`_run`), rather than
        opening a new loop per call. If opening the pool fails (e.g. bad
        credentials or an unreachable host), the loop and thread just started
        for this attempt are torn down before the error is raised, so a retried
        call — after the caller fixes the problem — does not leak the failed
        attempt's thread.

        Returns
        -------
        _LoopBoundPool
            The opened pool bundled with the event loop/thread it is bound to.
        """
        loop = asyncio.new_event_loop()
        thread = threading.Thread(target=loop.run_forever, daemon=True, name=f"pgvector-{self._collection_name}")
        thread.start()
        try:
            pool = asyncio.run_coroutine_threadsafe(self._open_pool(), loop).result()
        except BaseException:
            loop.call_soon_threadsafe(loop.stop)
            thread.join()
            loop.close()
            raise
        return _LoopBoundPool(pool=pool, loop=loop, thread=thread)

    async def _open_pool(self) -> asyncpg.Pool:
        """Open the asyncpg pool backing this store.

        Every physical connection the pool creates — at startup, to grow the
        pool under concurrent load, or to replace one it has closed — is
        configured identically via *init*: the ``vector``/``jsonb`` type codecs
        are registered and the ``pgvector`` extension is ensured, so no caller
        ever sees an unconfigured connection regardless of pool churn.

        Returns
        -------
        asyncpg.Pool
            The opened pool, sized between :attr:`_MIN_POOL_SIZE` and
            ``self._config.pool_max_size``.
        """
        connect_kwargs: dict[str, Any] = {
            "host": self._config.host,
            "port": self._config.port,
            "database": self._config.dbname,
            "user": self._config.user,
            "timeout": self._CONNECT_TIMEOUT,
        }
        if self._config.password:
            connect_kwargs["password"] = self._config.password

        return await asyncpg.create_pool(
            min_size=self._MIN_POOL_SIZE,
            max_size=self._config.pool_max_size,
            max_inactive_connection_lifetime=self._MAX_INACTIVE_CONNECTION_LIFETIME,
            init=self._configure_connection,
            **connect_kwargs,
        )

    def _ensure_db(self) -> asyncpg.Pool:
        """Return the pool, opening it and creating the table on the first call.

        Calls :meth:`_ensure_pool` for the pool, then creates the backing table
        once under :attr:`_db_lock`. Concurrent callers all return the same
        pool; the table is created exactly once.

        Lock ordering: :attr:`_db_lock` only (no nesting with
        :attr:`_indexes_lock` from this path — see :meth:`_ensure_indexes` for
        the callers that hold :attr:`_indexes_lock` → :attr:`_db_lock`).
        """
        pool = self._ensure_pool()
        if not self._table_created:
            with self._db_lock:
                if not self._table_created:
                    self._run(self._create_table(pool))
                    self._table_created = True
        return pool

    @staticmethod
    async def _configure_connection(conn: asyncpg.Connection) -> None:
        """Prepare one physical connection for use.

        Passed to :func:`asyncpg.create_pool` as its ``init`` callback, so it
        runs on every connection the pool creates — at startup, when growing the
        pool, or when replacing one it closed — rather than this store calling it
        once itself.

        The extension must be created *before* :func:`register_vector`: the
        latter looks up the ``vector`` type's OID, which does not exist until
        the extension has been created at least once. ``jsonb`` gets its own
        codec so ``metadata`` round-trips as a plain ``dict`` — asyncpg, unlike
        psycopg, does not decode ``jsonb`` to ``dict`` by default.

        Parameters
        ----------
        conn : asyncpg.Connection
            A newly opened, not-yet-pooled connection.
        """
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await register_vector(conn)
        await conn.set_type_codec("jsonb", schema="pg_catalog", encoder=json.dumps, decoder=json.loads, format="text")

    async def _create_table(self, pool: asyncpg.Pool) -> None:
        """Create the backing table if it does not already exist.

        The table maps one-to-one to the collection name and holds the chunk id,
        the dense ``embedding`` vector, the plain ``content_text``, its
        ``tokenized_content`` ``tsvector`` column feeding full-text (keyword)
        search, and a ``metadata`` JSONB column for the chunk's arbitrary
        metadata. ``content_text`` is the sole source of truth for chunk text —
        it is not also duplicated inside ``metadata``.
        """
        async with pool.acquire() as conn:
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._quoted_table()} (
                    id TEXT PRIMARY KEY,
                    metadata JSONB,
                    embedding vector({self._embedding_dimension}),
                    content_text TEXT,
                    tokenized_content TSVECTOR
                )
                """)
        logger.info("PGVector table ready: %s (dim=%d)", self._collection_name, self._embedding_dimension)

    def _ensure_indexes(self) -> None:
        """Build the HNSW and GIN indexes once, after documents have been loaded.

        Called lazily on the first search rather than at table creation. Bulk-inserting
        into an already-indexed table forces per-row HNSW graph maintenance — the slow,
        memory-hungry path that can crash the backend under large batches. Building the
        indexes once the data is in place is both faster and far lighter on server memory.
        ``IF NOT EXISTS`` keeps this a no-op for reused collections whose indexes already
        exist, and the in-memory flag avoids re-issuing the DDL on every subsequent search.

        ``search()`` runs concurrently across threads (see
        :attr:`PGVectorConfig.pool_max_size <ai4rag.rag.vector_store.config.PGVectorConfig.pool_max_size>`),
        so the flag check is guarded by :attr:`_indexes_lock` with the standard double-checked
        pattern: without it, two threads can both see ``False``, and both race to run the
        DDL. PostgreSQL's ``IF NOT EXISTS`` is not atomic across concurrent sessions — the
        loser doesn't silently no-op, it raises a real ``UniqueViolationError`` on the system
        catalog. The lock prevents that race for this instance; the ``UniqueViolationError``
        catch below is a second line of defense for a collection shared across instances
        (e.g. reused by another trial), where no Python-level lock can help.

        The HNSW index is only built when :attr:`_embedding_dimension` is within
        pgvector's :attr:`_MAX_INDEXABLE_DIMENSION` limit; above it, this step is
        skipped entirely (search then falls back to an exact sequential scan) while
        the GIN full-text index is still built unconditionally, since it does not
        depend on the embedding column at all.
        """
        if self._indexes_built:
            return

        # Lock ordering: _indexes_lock -> _db_lock (via _ensure_db inside).
        # Never acquire _indexes_lock while holding _db_lock.
        with self._indexes_lock:
            if self._indexes_built:
                return

            pool = self._ensure_db()
            hnsw_idx = f"idx_{self._collection_name}_hnsw"
            gin_idx = f"idx_{self._collection_name}_gin"
            self._run(self._build_indexes(pool, hnsw_idx, gin_idx))

            self._indexes_built = True
            logger.info("PGVector indexes ready: %s", self._collection_name)

    async def _build_indexes(self, pool: asyncpg.Pool, hnsw_idx: str, gin_idx: str) -> None:
        """Issue the HNSW (if indexable) and GIN index DDL on one connection.

        Parameters
        ----------
        pool : asyncpg.Pool
            Pool to borrow the connection from.
        hnsw_idx : str
            Name for the HNSW index.
        gin_idx : str
            Name for the GIN full-text index.
        """
        async with pool.acquire() as conn:
            # Each statement is guarded independently, not by one shared try/except:
            # each is its own implicit transaction, so a race lost on one index must
            # not skip creating the other.
            if self._embedding_dimension <= self._MAX_INDEXABLE_DIMENSION:
                await self._create_index_ignoring_race(
                    conn,
                    f"""
                    CREATE INDEX IF NOT EXISTS {hnsw_idx}
                    ON {self._quoted_table()} USING hnsw (embedding {self._index_ops})
                    """,
                )
            else:
                logger.info(
                    "Skipping HNSW index for %s: embedding dimension %d exceeds "
                    "pgvector's %d-dimension limit; search will use an exact "
                    "sequential scan.",
                    self._collection_name,
                    self._embedding_dimension,
                    self._MAX_INDEXABLE_DIMENSION,
                )
            await self._create_index_ignoring_race(
                conn,
                f"""
                CREATE INDEX IF NOT EXISTS {gin_idx}
                ON {self._quoted_table()} USING gin (tokenized_content)
                """,
            )

    async def _create_index_ignoring_race(self, conn: asyncpg.Connection, index_sql: str) -> None:
        """Run a ``CREATE INDEX IF NOT EXISTS`` statement, tolerating a concurrent creator.

        PostgreSQL's ``IF NOT EXISTS`` is not atomic across concurrent sessions: two
        sessions that both see the index absent can both attempt to create it, and the
        loser gets a real ``UniqueViolationError`` on the system catalog rather than a
        silent no-op. That outcome means the index now exists (created by the winner),
        which is exactly what this method is trying to achieve, so it is swallowed
        rather than raised.

        Parameters
        ----------
        conn : asyncpg.Connection
            Connection to execute *index_sql* on.
        index_sql : str
            A ``CREATE INDEX IF NOT EXISTS ...`` statement.
        """
        try:
            await conn.execute(index_sql)
        except asyncpg.exceptions.UniqueViolationError:
            logger.info(
                "PGVector index for %s was created concurrently elsewhere; continuing.",
                self._collection_name,
            )

    def _quoted_table(self) -> str:
        """Return the collection name as a safely double-quoted SQL identifier.

        Any embedded double quotes are escaped by doubling them, making the value
        safe to interpolate directly into table references.

        Returns
        -------
        str
            The double-quoted, escaped table identifier.
        """
        return '"' + self._collection_name.replace('"', '""') + '"'

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        search_mode: str = "vector",
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
            Whether to include similarity scores.
        search_mode : str, default="vector"
            ``"vector"`` for dense-only search or ``"hybrid"`` for dense +
            full-text search.
        ranker_strategy : str | None, default=None
            Hybrid ranker: ``"rrf"``, ``"weighted"``, or ``"normalized"``.
        ranker_k : int | None, default=None
            RRF smoothing constant (``k``).
        ranker_alpha : float | None, default=None
            Weighted blend factor (``0`` = keyword, ``1`` = vector).
        **kwargs : Any
            Accepted for interface compatibility; ignored by this backend.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Matched chunks, optionally paired with their scores.
        """
        validate_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha)
        self._ensure_indexes()

        if search_mode == "hybrid":
            return self._search_hybrid(query, k, include_scores, ranker_strategy, ranker_k, ranker_alpha)
        return self._search_vector(query, k, include_scores)

    def _distance_to_score(self, distance: float) -> float:
        """Convert a raw pgvector distance into a "higher = more relevant" score.

        pgvector's operators return two different kinds of value, so a single
        ``1 / distance`` rule does not fit all of them:

        * ``cosine`` (``<=>``), ``l2`` (``<->``), ``l1`` (``<+>``) return a
          non-negative *distance* — smaller means more similar. ``1 / distance``
          maps that into a monotonically decreasing score (``inf`` at an exact
          ``0`` distance), preserving the operator's ``ORDER BY distance ASC``
          ranking.
        * ``inner_product`` (``<#>``) returns the *negative* inner product, a
          signed value where a more negative result means more similar. Here
          ``1 / distance`` would be non-monotonic (it flips sign around zero and
          diverges at the boundary), inverting the ranking. Negating restores the
          plain inner product, for which higher already means more relevant and
          the ordering matches ``ORDER BY distance ASC``.

        Parameters
        ----------
        distance : float
            Raw value returned by the configured distance operator.

        Returns
        -------
        float
            Score where larger values indicate greater relevance.
        """
        if self._distance_key == "inner_product":
            return -distance
        return 1.0 / distance if distance != 0 else float("inf")

    def _search_vector(
        self, query: str, k: int, include_scores: bool
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run a pure dense-vector similarity search.

        Rows are ordered by the configured distance operator, and each distance
        is converted to a "higher = more relevant" score via
        :meth:`_distance_to_score` (which accounts for the signed value the
        ``inner_product`` operator returns).

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.
        include_scores : bool
            Whether to pair each returned chunk with its score.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Matched chunks, optionally paired with their scores.
        """
        # embed_query() is a synchronous, network-bound call: it runs here, on the
        # caller's own thread, before anything is dispatched to the shared event
        # loop. Doing it inside the coroutine instead would block that one loop
        # for every other concurrent search()/add_documents() on this store.
        embedding = self.embedding_model.embed_query(query)
        pool = self._ensure_db()
        rows = self._run_with_retry(lambda: self._fetch_vector_rows(pool, embedding, k))
        results = self._vector_rows_to_results(rows)
        if include_scores:
            return results
        return [chunk for chunk, _ in results]

    def _vector_rows_to_results(self, rows: list[asyncpg.Record]) -> list[tuple[AI4RAGChunk, float]]:
        """Convert raw dense-search rows into chunks paired with relevance scores."""
        results: list[tuple[AI4RAGChunk, float]] = []
        for content_text, metadata, distance in rows:
            score = self._distance_to_score(float(distance))
            chunk = AI4RAGChunk(text=content_text, metadata=self._parse_metadata(metadata))
            results.append((chunk, score))
        return results

    async def _fetch_vector_rows(self, pool: asyncpg.Pool, embedding: list[float], k: int) -> list[asyncpg.Record]:
        """Run the dense similarity query and return the raw matching rows."""
        # `acquire(timeout=...)` matters beyond bounding the wait for a free slot: asyncpg
        # records it as the connection holder's budget for the *release-time* cleanup this
        # `async with` triggers on exit (resetting session state, awaiting confirmation of a
        # cancelled query) — see PoolConnectionHolder.release() in asyncpg/pool.py. Without it,
        # that cleanup step defaults to no timeout at all, so a query that times out on a truly
        # dead connection could still hang indefinitely at release, right where _COMMAND_TIMEOUT
        # was meant to prevent exactly that.
        async with pool.acquire(timeout=self._COMMAND_TIMEOUT) as conn:
            return await conn.fetch(
                f"""
                SELECT content_text, metadata, embedding {self._distance_operator} $1::vector AS distance
                FROM {self._quoted_table()}
                ORDER BY distance
                LIMIT $2
                """,
                embedding,
                k,
                timeout=self._COMMAND_TIMEOUT,
            )

    def _search_keyword(self, query: str, k: int) -> list[tuple[AI4RAGChunk, float]]:
        """Run a PostgreSQL full-text (keyword) search.

        Ranks rows whose ``tokenized_content`` matches the ``plainto_tsquery`` of
        *query* by ``ts_rank``, highest first.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.

        Returns
        -------
        list[tuple[AI4RAGChunk, float]]
            Matched chunks paired with their ``ts_rank`` scores.
        """
        pool = self._ensure_db()
        rows = self._run_with_retry(lambda: self._fetch_keyword_rows(pool, query, k))
        return self._keyword_rows_to_results(rows)

    def _keyword_rows_to_results(self, rows: list[asyncpg.Record]) -> list[tuple[AI4RAGChunk, float]]:
        """Convert raw keyword-search rows into chunks paired with ``ts_rank`` scores."""
        results: list[tuple[AI4RAGChunk, float]] = []
        for content_text, metadata, score in rows:
            chunk = AI4RAGChunk(text=content_text, metadata=self._parse_metadata(metadata))
            results.append((chunk, float(score)))
        return results

    async def _fetch_keyword_rows(self, pool: asyncpg.Pool, query: str, k: int) -> list[asyncpg.Record]:
        """Run the full-text query and return the raw matching rows."""
        # See the matching comment in _fetch_vector_rows: the acquire-time timeout also
        # bounds this connection's release-time cleanup, not just the wait for a free slot.
        async with pool.acquire(timeout=self._COMMAND_TIMEOUT) as conn:
            return await conn.fetch(
                f"""
                SELECT content_text, metadata, ts_rank(tokenized_content, plainto_tsquery('english', $1)) AS score
                FROM {self._quoted_table()}
                WHERE tokenized_content @@ plainto_tsquery('english', $1)
                ORDER BY score DESC
                LIMIT $2
                """,
                query,
                k,
                timeout=self._COMMAND_TIMEOUT,
            )

    async def _fetch_hybrid_rows(
        self, pool: asyncpg.Pool, embedding: list[float], query: str, k: int
    ) -> tuple[list[asyncpg.Record], list[asyncpg.Record]]:
        """Run the dense and keyword row fetches concurrently and return both raw row sets.

        The two queries are independent (different WHERE/ORDER BY clauses, no
        shared state) and each borrows its own connection from *pool*, so
        :func:`asyncio.gather` runs them concurrently on the shared event loop
        instead of one waiting on the other's round trip — this coroutine is
        dispatched as a single :meth:`_run` call for exactly that reason; two
        separate ``_run`` calls from the caller thread would serialize them
        again by blocking on the first's result before issuing the second.
        """
        return await asyncio.gather(
            self._fetch_vector_rows(pool, embedding, k),
            self._fetch_keyword_rows(pool, query, k),
        )

    @staticmethod
    def _parse_metadata(metadata: dict | None) -> dict:
        """Normalize a ``metadata`` column value into a plain dict.

        The ``jsonb`` codec registered in :meth:`_configure_connection` decodes
        the column to a ``dict`` (or ``None`` for SQL ``NULL``) on every
        connection this store's pool hands out.

        Parameters
        ----------
        metadata : dict | None
            Raw value read from the ``metadata`` column.

        Returns
        -------
        dict
            The chunk's metadata, or ``{}`` when none was stored.
        """
        return metadata or {}

    def _search_hybrid(
        self,
        query: str,
        k: int,
        include_scores: bool,
        ranker_strategy: str | None,
        ranker_k: int | None,
        ranker_alpha: float | None,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run a hybrid dense + full-text search with in-memory fusion.

        Runs the dense and keyword *database* queries concurrently — as a single
        :func:`asyncio.gather` dispatched through one :meth:`_run` call, since the
        keyword lookup has no dependency on the vector lookup or the embedding
        step — then fuses their per-chunk score maps with
        :class:`WeightedInMemoryAggregator` and keeps the top ``k`` results.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.
        include_scores : bool
            Whether to pair each returned chunk with its fused score.
        ranker_strategy : str | None
            Fusion strategy: ``"rrf"``, ``"weighted"``, or ``"normalized"``.
            Defaults to RRF when ``None``.
        ranker_k : int | None
            RRF smoothing constant; applied only for the ``"rrf"`` strategy.
        ranker_alpha : float | None
            Weighted blend factor; applied only for the ``"weighted"`` strategy.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Fused chunks, optionally paired with their scores.
        """
        # embed_query() runs here, on the caller's own thread, for the same reason
        # given in _search_vector — and because the vector fetch below needs the
        # embedding before it can be dispatched at all.
        embedding = self.embedding_model.embed_query(query)
        pool = self._ensure_db()
        vector_rows, keyword_rows = self._run_with_retry(lambda: self._fetch_hybrid_rows(pool, embedding, query, k))
        chunk_map, combined_scores = self._fuse_results(
            self._vector_rows_to_results(vector_rows),
            self._keyword_rows_to_results(keyword_rows),
            ranker_strategy,
            ranker_k,
            ranker_alpha,
        )

        top_k_items = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])

        if include_scores:
            return [(chunk_map[doc_id], score) for doc_id, score in top_k_items if doc_id in chunk_map]
        return [chunk_map[doc_id] for doc_id, _ in top_k_items if doc_id in chunk_map]

    @staticmethod
    def _fuse_results(
        vector_results: list[tuple[AI4RAGChunk, float]],
        keyword_results: list[tuple[AI4RAGChunk, float]],
        ranker_strategy: str | None,
        ranker_k: int | None,
        ranker_alpha: float | None,
    ) -> tuple[dict[str, AI4RAGChunk], dict[str, float]]:
        """Fuse dense and keyword result sets into one combined score map.

        Builds per-chunk score maps for each modality (the dense results seed the
        shared chunk lookup, so a chunk found by both searches is stored once),
        selects the single reranker parameter matching *ranker_strategy*, and
        delegates the blend to :class:`WeightedInMemoryAggregator`.

        Parameters
        ----------
        vector_results : list[tuple[AI4RAGChunk, float]]
            Dense-search hits paired with their similarity scores.
        keyword_results : list[tuple[AI4RAGChunk, float]]
            Full-text-search hits paired with their ``ts_rank`` scores.
        ranker_strategy : str | None
            Fusion strategy: ``"rrf"`` consumes ``ranker_k``, ``"weighted"``
            consumes ``ranker_alpha``; defaults to RRF when ``None``.
        ranker_k : int | None
            RRF smoothing constant; applied only for the ``"rrf"`` strategy.
        ranker_alpha : float | None
            Weighted blend factor; applied only for the ``"weighted"`` strategy.

        Returns
        -------
        tuple[dict[str, AI4RAGChunk], dict[str, float]]
            The ``chunk_id`` → chunk lookup and the fused ``chunk_id`` → score map.
        """
        vector_scores: dict[str, float] = {}
        keyword_scores: dict[str, float] = {}
        chunk_map: dict[str, AI4RAGChunk] = {}

        for chunk, score in vector_results:
            vector_scores[chunk.chunk_id] = score
            chunk_map[chunk.chunk_id] = chunk

        for chunk, score in keyword_results:
            keyword_scores[chunk.chunk_id] = score
            chunk_map.setdefault(chunk.chunk_id, chunk)

        reranker_params: dict[str, Any] = {}
        if ranker_strategy == "rrf" and ranker_k is not None and ranker_k > 0:
            reranker_params["k"] = ranker_k
        if ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
            reranker_params["alpha"] = ranker_alpha

        combined_scores = WeightedInMemoryAggregator.combine_search_results(
            vector_scores, keyword_scores, ranker_strategy or "rrf", reranker_params
        )
        return chunk_map, combined_scores

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
        """Embed, deduplicate, and upsert chunks into PGVector.

        Duplicate ``chunk_id`` values within *documents* are skipped (first
        occurrence wins) and logged. Rows are upserted in batches, each with a
        one-shot retry on a dropped connection.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and stored.
        **kwargs : Any
            Optional overrides. ``batch_size`` (int) sets the insert batch size
            (default :attr:`_BATCH_SIZE`).
        """
        if not documents:
            return

        # embed_documents() is a synchronous, network-bound call: it runs here, on
        # the caller's own thread, before anything is dispatched to the shared
        # event loop (see the matching note in _search_vector).
        embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])
        pool = self._ensure_db()

        values: list[tuple[str, dict, list[float], str, str]] = []
        for doc, embedding in iter_unique_chunks(documents, embeddings):
            values.append((doc.chunk_id, doc.metadata, embedding, doc.text, doc.text))

        batch_size = kwargs.get("batch_size", self._BATCH_SIZE)
        for idx in range(0, len(values), batch_size):
            self._insert_batch_with_retry(pool, values[idx : idx + batch_size])

    async def _insert_batch_async(
        self, pool: asyncpg.Pool, batch: list[tuple[str, dict, list[float], str, str]]
    ) -> None:
        """Upsert a single batch of rows into the table.

        The trailing text of each row feeds ``to_tsvector`` for the full-text
        column, and existing ids are updated in place via ``ON CONFLICT``.

        Parameters
        ----------
        pool : asyncpg.Pool
            Pool to borrow the connection from.
        batch : list[tuple[str, dict, list[float], str, str]]
            Rows to upsert, each as ``(id, metadata, embedding, content text,
            text to tokenize)``.
        """
        # See the matching comment in _fetch_vector_rows: the acquire-time timeout also
        # bounds this connection's release-time cleanup, not just the wait for a free slot.
        async with pool.acquire(timeout=self._COMMAND_TIMEOUT) as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self._quoted_table()} (id, metadata, embedding, content_text, tokenized_content)
                VALUES ($1, $2::jsonb, $3::vector, $4, to_tsvector('english', $5))
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    content_text = EXCLUDED.content_text,
                    tokenized_content = EXCLUDED.tokenized_content
                """,
                batch,
                timeout=self._COMMAND_TIMEOUT,
            )

    def _insert_batch_with_retry(
        self, pool: asyncpg.Pool, batch: list[tuple[str, dict, list[float], str, str]]
    ) -> None:
        """Insert one batch, retrying once on a dropped connection.

        The ``ON CONFLICT`` upsert makes the retry idempotent even when the first attempt
        committed some rows before the connection died. The pool discards a connection it
        finds broken and hands out a fresh one on the next borrow, so the retry itself needs
        no explicit reconnect. This recovers from *transient* drops (recycled backend,
        middlebox); it deliberately does not mask a deterministic failure — a batch that
        always kills the backend still surfaces after one retry.

        Kept distinct from the generic :meth:`_run_with_retry` so the retry log line can
        report the batch size; behaviorally identical otherwise.
        """
        try:
            self._run(self._insert_batch_async(pool, batch))
        except asyncpg.exceptions.PostgresConnectionError as exc:
            logger.warning("PGVector insert failed (%s); retrying batch of %d rows.", exc, len(batch))
            self._run(self._insert_batch_async(pool, batch))

    def clean_collection(self) -> None:
        """Drop the PostgreSQL table."""
        pool = self._ensure_pool()
        self._run_with_retry(lambda: self._drop_table(pool))

    async def _drop_table(self, pool: asyncpg.Pool) -> None:
        async with pool.acquire() as conn:
            await conn.execute(f"DROP TABLE IF EXISTS {self._quoted_table()} CASCADE")

    def close(self) -> None:
        """Close the connection pool and stop the store's background event loop.

        Idempotent: a second call sees :attr:`_db` already ``None`` and returns
        immediately, rather than dispatching onto a loop that has already
        stopped — which would hang forever waiting for a callback the stopped
        loop can never run.

        Safe to call while another thread is mid-:meth:`_run`: :attr:`_db` is
        cleared up front (under :attr:`_inflight`) so no *new* call can start,
        but the loop and pool are only stopped once every call that already
        started has finished — see :meth:`_run`. A call that started before
        this method clears :attr:`_db` still completes normally; one that
        starts after gets a clear ``RuntimeError`` instead of racing the teardown.
        """
        with self._inflight.cond:
            if self._db is None:
                return
            db = self._db
            self._db = None
            while self._inflight.count > 0:
                self._inflight.cond.wait()
        self._dispatch(db.loop, db.pool.close())
        db.loop.call_soon_threadsafe(db.loop.stop)
        db.thread.join()
        db.loop.close()
