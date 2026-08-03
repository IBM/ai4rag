# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""PGVectorStore — direct ``psycopg`` + ``pgvector`` wrapper for vector and hybrid search."""

from __future__ import annotations

import heapq
import json
from typing import Any

import psycopg
from pgvector.psycopg import register_vector

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
        """Initialize the store, connect to PostgreSQL, and ensure the table.

        Resolves the distance metric to its pgvector operator and index opclass,
        opens a connection (registering the vector adapter and ensuring the
        ``vector`` extension), and creates the backing table when absent. HNSW
        and GIN indexes are built lazily on the first search (see
        :meth:`_ensure_indexes`), not here.

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
        self._distance_operator = self._DISTANCE_METRIC_TO_OPERATOR[distance_key]
        self._index_ops = self._DISTANCE_METRIC_TO_INDEX_OPS[distance_key]

        # Indexes are built lazily after documents are loaded (see ``_ensure_indexes``),
        # not at connection time: maintaining an HNSW graph on every insert is the
        # memory-heavy path that can trigger the server-side OOM killer on large batches.
        self._indexes_built = False

        self._conn: psycopg.Connection | None = None
        self._connect()

        # The collection name IS the physical table name: the base class has
        # already validated (ai4rag prefix) and sanitized it into a safe SQL
        # identifier, so no separate table name or prefix is needed.
        self._create_table()

    def _connect(self) -> None:
        """Open a new connection, register the vector adapter, and ensure the extension.

        Extracted from :meth:`__init__` so :meth:`_ensure_connection` and the insert
        retry path can transparently re-establish a connection that the server has
        dropped (e.g. a recycled backend or a middlebox closing an idle socket).
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

        self._conn = psycopg.connect(**connect_kwargs)
        register_vector(self._conn)
        self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    def _ensure_connection(self) -> None:
        """Reconnect if the connection was never opened or has since been closed."""
        if self._conn is None or self._conn.closed:
            logger.warning("PGVector connection is down; reconnecting to %s:%s", self._config.host, self._config.port)
            self._connect()

    def _reconnect(self) -> None:
        """Force-close a broken connection and open a fresh one."""
        try:
            if self._conn is not None and not self._conn.closed:
                self._conn.close()
        except Exception:  # pragma: no cover - best-effort teardown of an already-broken socket
            pass
        self._connect()

    def _create_table(self) -> None:
        """Create the backing table if it does not already exist.

        The table maps one-to-one to the collection name and holds the chunk id,
        the raw chunk JSON (``document``), the dense ``embedding`` vector, the
        plain ``content_text``, and a ``tokenized_content`` ``tsvector`` column
        feeding full-text (keyword) search.
        """
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._quoted_table()} (
                id TEXT PRIMARY KEY,
                document JSONB,
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
        """
        if self._indexes_built:
            return

        self._ensure_connection()

        hnsw_idx = f"idx_{self._collection_name}_hnsw"
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {hnsw_idx}
            ON {self._quoted_table()} USING hnsw (embedding {self._index_ops})
            """)

        gin_idx = f"idx_{self._collection_name}_gin"
        self._conn.execute(f"""
            CREATE INDEX IF NOT EXISTS {gin_idx}
            ON {self._quoted_table()} USING gin (tokenized_content)
            """)

        self._indexes_built = True
        logger.info("PGVector indexes ready: %s", self._collection_name)

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
        self._ensure_connection()
        self._ensure_indexes()

        if search_mode == "hybrid":
            return self._search_hybrid(query, k, include_scores, ranker_strategy, ranker_k, ranker_alpha)
        return self._search_vector(query, k, include_scores)

    def _search_vector(
        self, query: str, k: int, include_scores: bool
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run a pure dense-vector similarity search.

        Rows are ordered by the configured distance operator, and each distance
        is converted to a "higher = more relevant" score as ``1 / distance``
        (``inf`` for an exact match at distance ``0``).

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

        rows = self._conn.execute(
            f"""
            SELECT document, embedding {self._distance_operator} %s::vector AS distance
            FROM {self._quoted_table()}
            ORDER BY distance
            LIMIT %s
            """,
            (embedding, k),
        ).fetchall()

        results: list[tuple[AI4RAGChunk, float]] = []
        for row in rows:
            doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            dist = float(row[1])
            score = 1.0 / dist if dist != 0 else float("inf")
            chunk = AI4RAGChunk(text=doc["content"], metadata=doc.get("metadata", {}))
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
        rows = self._conn.execute(
            f"""
            SELECT document, ts_rank(tokenized_content, plainto_tsquery('english', %s)) AS score
            FROM {self._quoted_table()}
            WHERE tokenized_content @@ plainto_tsquery('english', %s)
            ORDER BY score DESC
            LIMIT %s
            """,
            (query, query, k),
        ).fetchall()

        results: list[tuple[AI4RAGChunk, float]] = []
        for row in rows:
            doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            score = float(row[1])
            chunk = AI4RAGChunk(text=doc["content"], metadata=doc.get("metadata", {}))
            results.append((chunk, score))
        return results

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
        one-shot reconnect-and-retry on a dropped connection.

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
            doc_json = json.dumps({"content": doc.text, "metadata": doc.metadata, "chunk_id": doc.chunk_id})
            values.append((doc.chunk_id, doc_json, embedding, doc.text, doc.text))

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
            Rows to upsert, each as ``(id, document JSON, embedding, content
            text, text to tokenize)``.
        """
        with self._conn.cursor() as cur:
            cur.executemany(
                f"""
                INSERT INTO {self._quoted_table()} (id, document, embedding, content_text, tokenized_content)
                VALUES (%s, %s::jsonb, %s::vector, %s, to_tsvector('english', %s))
                ON CONFLICT (id) DO UPDATE SET
                    embedding = EXCLUDED.embedding,
                    document = EXCLUDED.document,
                    content_text = EXCLUDED.content_text,
                    tokenized_content = EXCLUDED.tokenized_content
                """,
                batch,
            )

    def _insert_batch_with_retry(self, batch: list[tuple[str, str, list[float], str, str]]) -> None:
        """Insert one batch, reconnecting once and retrying if the connection was dropped.

        The ``ON CONFLICT`` upsert makes the retry idempotent even when the first attempt
        committed some rows before the connection died. This recovers from *transient*
        drops (recycled backend, middlebox); it deliberately does not mask a deterministic
        failure — a batch that always kills the backend still surfaces after one retry.
        """
        self._ensure_connection()
        try:
            self._insert_batch(batch)
        except psycopg.OperationalError as exc:
            logger.warning("PGVector insert failed (%s); reconnecting and retrying batch of %d rows.", exc, len(batch))
            self._reconnect()
            self._insert_batch(batch)

    def clean_collection(self) -> None:
        """Drop the PostgreSQL table."""
        self._conn.execute(f"DROP TABLE IF EXISTS {self._quoted_table()} CASCADE")

    def close(self) -> None:
        """Close the database connection."""
        if self._conn and not self._conn.closed:
            self._conn.close()
