# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import heapq
import json
import threading
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import PGVectorConfig
from ai4rag.rag.vector_store.reranker import WeightedInMemoryAggregator
from ai4rag.rag.vector_store.utils import iter_unique_chunks, resolve_embedding_dimension, validate_search_params

__all__ = ["PGVectorStore"]


class PGVectorStore(BaseVectorStore):
    """Vector store backed by PostgreSQL with the ``pgvector`` extension.

    Supports pure vector search and hybrid search (dense vector + tsvector
    full-text) with RRF or weighted reranking via in-memory fusion.

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

    # search() and add_documents() may be called concurrently across threads (e.g.
    # one worker per benchmark question in query_rag(), or one per concurrent
    # request in a deployed service), and a single shared psycopg connection is
    # not safe for concurrent use. The pool starts at _MIN_POOL_SIZE and grows
    # lazily up to the caller-supplied config.pool_max_size, so a fully concurrent
    # caller never queues for a slot as long as pool_max_size covers its own
    # concurrency (see PGVectorConfig.pool_max_size).
    #
    # min_size=0 (no proactive connections): when _create_table() borrows the one
    # pre-warmed connection, a min_size=1 pool would immediately open a second
    # connection to restore the minimum, leaving *two* idle connections during the
    # subsequent embed_documents() call.  With a large corpus (e.g. 556 hybrid
    # chunks), that call can take >80 s — long enough for the OS TCP keepalive
    # exhaustion (keepalives_idle + count*interval = 80 s) to declare both sockets
    # dead.  The insert then fails on the first stale connection and the single
    # retry exhausts the second stale one, raising IndexingError.  With min_size=0
    # the pool never proactively opens a second connection, so only one socket
    # accumulates during embedding; the existing one-retry policy picks up a fresh
    # connection and the insert succeeds.
    _MIN_POOL_SIZE = 0

    # pgvector caps HNSW (and IVFFlat) indexes on the ``vector`` type at 2000
    # dimensions (https://github.com/pgvector/pgvector#hnsw). Higher-dimensional
    # vectors still store and query correctly, but the index cannot be built. Since
    # indexes are created lazily on the first search (see ``_ensure_indexes``), an
    # oversized model would otherwise crash on the first query — after a full, and
    # potentially costly, embed-and-insert cycle. Rejecting it here fails fast,
    # before a connection is opened or a single document is embedded.
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
        """Initialize the store, open a connection pool, and ensure the table.

        Resolves the distance metric to its pgvector operator and index opclass,
        opens a connection pool (registering the vector adapter and ensuring the
        ``vector`` extension on every pooled connection), and creates the backing
        table when absent. HNSW and GIN indexes are built lazily on the first
        search (see :meth:`_ensure_indexes`), not here.

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
            If ``distance_metric`` is not one of the supported metrics, or if the
            model's embedding dimension exceeds pgvector's HNSW index limit of
            :attr:`_MAX_INDEXABLE_DIMENSION` dimensions.
        """
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(self.embedding_model)
        if self._embedding_dimension > self._MAX_INDEXABLE_DIMENSION:
            raise ValueError(
                f"Embedding dimension {self._embedding_dimension} exceeds pgvector's "
                f"{self._MAX_INDEXABLE_DIMENSION}-dimension limit for HNSW indexes. "
                f"Use an embedding model with at most {self._MAX_INDEXABLE_DIMENSION} "
                "dimensions, or a backend that supports higher-dimensional indexing "
                "(e.g. Milvus)."
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

        self._pool = self._open_pool()

        # The collection name IS the physical table name: the base class has
        # already validated (ai4rag prefix) and sanitized it into a safe SQL
        # identifier, so no separate table name or prefix is needed.
        self._create_table()

    def _open_pool(self) -> ConnectionPool:
        """Open the connection pool backing this store.

        Every physical connection the pool creates — at startup, to grow the
        pool under concurrent load, or to replace one the pool has detected as
        broken — is configured identically via *configure*: the ``vector`` type
        adapter is registered and the ``pgvector`` extension is ensured, so no
        caller ever sees an unconfigured connection regardless of pool churn.

        Returns
        -------
        ConnectionPool
            The opened pool, sized between :attr:`_MIN_POOL_SIZE` and
            ``self._config.pool_max_size``. :meth:`_create_table`, called right
            after this in :meth:`__init__`, borrows the first connection and so
            blocks (up to :attr:`_CONNECT_TIMEOUT`) until one is ready — a
            misconfigured connection still fails fast, during construction.
        """
        connect_kwargs: dict[str, Any] = {
            "host": self._config.host,
            "port": self._config.port,
            "dbname": self._config.dbname,
            "user": self._config.user,
            "autocommit": True,
            "connect_timeout": self._CONNECT_TIMEOUT,
            # Keep idle connections alive through NAT/firewall middleboxes so a long
            # embed-then-insert cycle is not silently dropped mid-batch.
            "keepalives": 1,
            "keepalives_idle": 30,
            "keepalives_interval": 10,
            "keepalives_count": 5,
        }
        if self._config.password:
            connect_kwargs["password"] = self._config.password

        return ConnectionPool(
            kwargs=connect_kwargs,
            min_size=self._MIN_POOL_SIZE,
            max_size=self._config.pool_max_size,
            configure=self._configure_connection,
            timeout=self._CONNECT_TIMEOUT,
            open=True,
        )

    @staticmethod
    def _configure_connection(conn: psycopg.Connection) -> None:
        """Prepare one physical connection for use: register the vector adapter and ensure the extension.

        Passed to :class:`ConnectionPool` as its ``configure`` callback, so the
        pool invokes it on every connection it creates — at startup, when
        growing the pool, or when replacing one it found broken — rather than
        this store calling it once itself.

        Parameters
        ----------
        conn : psycopg.Connection
            A newly opened, not-yet-pooled connection.
        """
        register_vector(conn)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def _create_table(self) -> None:
        """Create the backing table if it does not already exist.

        The table maps one-to-one to the collection name and holds the chunk id,
        the dense ``embedding`` vector, the plain ``content_text``, its
        ``tokenized_content`` ``tsvector`` column feeding full-text (keyword)
        search, and a ``metadata`` JSONB column for the chunk's arbitrary
        metadata. ``content_text`` is the sole source of truth for chunk text —
        it is not also duplicated inside ``metadata``.
        """
        with self._pool.connection() as conn:
            conn.execute(f"""
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
        loser doesn't silently no-op, it raises a real ``UniqueViolation`` on the system
        catalog. The lock prevents that race for this instance; the ``UniqueViolation``
        catch below is a second line of defense for a collection shared across instances
        (e.g. reused by another trial), where no Python-level lock can help.
        """
        if self._indexes_built:
            return

        with self._indexes_lock:
            if self._indexes_built:
                return

            hnsw_idx = f"idx_{self._collection_name}_hnsw"
            gin_idx = f"idx_{self._collection_name}_gin"
            with self._pool.connection() as conn:
                # Each statement is guarded independently, not by one shared try/except:
                # under autocommit there is no transaction spanning them, so a race lost on
                # one index must not skip creating the other.
                self._create_index_ignoring_race(
                    conn,
                    f"""
                    CREATE INDEX IF NOT EXISTS {hnsw_idx}
                    ON {self._quoted_table()} USING hnsw (embedding {self._index_ops})
                    """,
                )
                self._create_index_ignoring_race(
                    conn,
                    f"""
                    CREATE INDEX IF NOT EXISTS {gin_idx}
                    ON {self._quoted_table()} USING gin (tokenized_content)
                    """,
                )

            self._indexes_built = True
            logger.info("PGVector indexes ready: %s", self._collection_name)

    def _create_index_ignoring_race(self, conn: psycopg.Connection, index_sql: str) -> None:
        """Run a ``CREATE INDEX IF NOT EXISTS`` statement, tolerating a concurrent creator.

        PostgreSQL's ``IF NOT EXISTS`` is not atomic across concurrent sessions: two
        sessions that both see the index absent can both attempt to create it, and the
        loser gets a real ``UniqueViolation`` on the system catalog rather than a silent
        no-op. That outcome means the index now exists (created by the winner), which is
        exactly what this method is trying to achieve, so it is swallowed rather than
        raised.

        Parameters
        ----------
        conn : psycopg.Connection
            Connection to execute *index_sql* on.
        index_sql : str
            A ``CREATE INDEX IF NOT EXISTS ...`` statement.
        """
        try:
            conn.execute(index_sql)
        except psycopg.errors.UniqueViolation:
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
        embedding = self.embedding_model.embed_query(query)

        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT content_text, metadata, embedding {self._distance_operator} %s::vector AS distance
                FROM {self._quoted_table()}
                ORDER BY distance
                LIMIT %s
                """,
                (embedding, k),
            ).fetchall()

        results: list[tuple[AI4RAGChunk, float]] = []
        for content_text, metadata, distance in rows:
            score = self._distance_to_score(float(distance))
            chunk = AI4RAGChunk(text=content_text, metadata=self._parse_metadata(metadata))
            results.append((chunk, score))
        if include_scores:
            return results
        return [chunk for chunk, _ in results]

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
        with self._pool.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT content_text, metadata, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score
                FROM {self._quoted_table()}
                WHERE tokenized_content @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, k),
            ).fetchall()

        results: list[tuple[AI4RAGChunk, float]] = []
        for content_text, metadata, score in rows:
            chunk = AI4RAGChunk(text=content_text, metadata=self._parse_metadata(metadata))
            results.append((chunk, float(score)))
        return results

    @staticmethod
    def _parse_metadata(metadata: dict | str | None) -> dict:
        """Normalize a ``metadata`` column value into a plain dict.

        psycopg auto-decodes ``JSONB`` into a ``dict`` on most drivers, but a
        defensive ``json.loads`` fallback keeps this correct if a connection
        ever returns the raw JSON string instead.

        Parameters
        ----------
        metadata : dict | str | None
            Raw value read from the ``metadata`` column.

        Returns
        -------
        dict
            The chunk's metadata, or ``{}`` when none was stored.
        """
        if isinstance(metadata, dict):
            return metadata
        return json.loads(metadata) if metadata else {}

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

        Runs the dense and keyword searches independently, fuses their per-chunk
        score maps with :class:`WeightedInMemoryAggregator`, and keeps the top
        ``k`` results.

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
        vector_results = self._search_vector(query, k, include_scores=True)
        keyword_results = self._search_keyword(query, k)
        chunk_map, combined_scores = self._fuse_results(
            vector_results, keyword_results, ranker_strategy, ranker_k, ranker_alpha
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

        embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])

        values: list[tuple[str, str, list[float], str, str]] = []
        for doc, embedding in iter_unique_chunks(documents, embeddings):
            metadata_json = json.dumps(doc.metadata)
            values.append((doc.chunk_id, metadata_json, embedding, doc.text, doc.text))

        batch_size = kwargs.get("batch_size", self._BATCH_SIZE)
        for idx in range(0, len(values), batch_size):
            self._insert_batch_with_retry(values[idx : idx + batch_size])

    def _insert_batch(self, batch: list[tuple[str, str, list[float], str, str]]) -> None:
        """Upsert a single batch of rows into the table.

        The trailing text of each row feeds ``to_tsvector`` for the full-text
        column, and existing ids are updated in place via ``ON CONFLICT``.

        Parameters
        ----------
        batch : list[tuple[str, str, list[float], str, str]]
            Rows to upsert, each as ``(id, metadata JSON, embedding, content
            text, text to tokenize)``.
        """
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._quoted_table()} (id, metadata, embedding, content_text, tokenized_content)
                VALUES (%s, %s::jsonb, %s::vector, %s, to_tsvector('english', %s))
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata,
                    content_text = EXCLUDED.content_text,
                    tokenized_content = EXCLUDED.tokenized_content
                """,
                batch,
            )

    def _insert_batch_with_retry(self, batch: list[tuple[str, str, list[float], str, str]]) -> None:
        """Insert one batch, retrying once on a dropped connection.

        The ``ON CONFLICT`` upsert makes the retry idempotent even when the first attempt
        committed some rows before the connection died. The pool discards a connection it
        finds broken and hands out a fresh one on the next borrow, so the retry itself needs
        no explicit reconnect. This recovers from *transient* drops (recycled backend,
        middlebox); it deliberately does not mask a deterministic failure — a batch that
        always kills the backend still surfaces after one retry.
        """
        try:
            self._insert_batch(batch)
        except psycopg.OperationalError as exc:
            logger.warning("PGVector insert failed (%s); retrying batch of %d rows.", exc, len(batch))
            self._insert_batch(batch)

    def clean_collection(self) -> None:
        """Drop the PostgreSQL table."""
        with self._pool.connection() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {self._quoted_table()} CASCADE")

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()
