# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Integration tests for :class:`PGVectorStore` against a live PostgreSQL server.

These tests exercise the full table lifecycle — create, add embeddings, search,
drop — against a real PostgreSQL instance with the ``pgvector`` extension, and
are skipped unless ``PGVECTOR_HOST`` is set. Connection settings are read via
:meth:`PGVectorConfig.from_env` (``PGVECTOR_HOST``, ``PGVECTOR_PORT``,
``PGVECTOR_DB``, ``PGVECTOR_USER``, ``PGVECTOR_PASSWORD``); see
``tests/integration/conftest.py`` for how ``.env.local`` is loaded.
"""

import os

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import PGVectorConfig
from ai4rag.rag.vector_store.pgvector import PGVectorStore
from tests.integration.vector_store.conftest import DeterministicEmbeddingModel

pytestmark = pytest.mark.skipif(
    os.environ.get("PGVECTOR_HOST") is None,
    reason="PGVECTOR_HOST is not set; skipping live pgvector integration tests.",
)


def _table_exists(store: PGVectorStore) -> bool:
    """Return whether the store's backing table currently exists in the database."""

    # ``to_regclass`` resolves a relation name to its OID, or NULL if it does not
    # exist — a non-throwing existence check. The collection name IS the table
    # name; ``_quoted_table`` supplies the safely-quoted identifier. Routed through
    # store._run() since the pool's connections belong to the store's own event loop.
    async def _check() -> object:
        async with store._db.pool.acquire() as conn:
            return await conn.fetchval("SELECT to_regclass($1)", store._quoted_table())

    return store._run(_check()) is not None


@pytest.mark.pgvector
class TestPGVectorIntegration:
    """Full create → add → search → drop lifecycle against a live pgvector server.

    A single class-scoped ``vector_store`` fixture owns the lifecycle: it creates
    and populates the table on setup and drops it (and closes the connection) on
    teardown, so the remaining test methods are order-independent, read-only
    assertions over the same populated table.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def vector_store(embedding_model, sample_chunks):
        """Create and populate a uniquely named table; drop it and close on teardown."""
        store = PGVectorStore(
            embedding_model=embedding_model, config=PGVectorConfig.from_env(), collection_name="ai4rag_integration_test"
        )
        store.add_documents(sample_chunks)
        try:
            yield store
        finally:
            store.clean_collection()
            store.close()

    def test_table_is_created(self, vector_store):
        """The store's backing table exists in the database."""
        assert vector_store.collection_name.startswith("ai4rag")
        assert _table_exists(vector_store)

    def test_all_documents_are_added(self, vector_store, sample_chunks):
        """Every added chunk is persisted and independently retrievable."""
        for chunk in sample_chunks:
            results = vector_store.search(chunk.text, k=len(sample_chunks))
            assert chunk.text in {result.text for result in results}

    def test_search_returns_relevant_chunk(self, vector_store, sample_chunks):
        """A query for a chunk's exact text ranks that chunk first, metadata intact."""
        target = sample_chunks[0]
        results = vector_store.search(target.text, k=len(sample_chunks))

        assert results, "search returned no results"
        assert results[0].text == target.text
        assert results[0].metadata["document_id"] == target.metadata["document_id"]

    def test_search_respects_k(self, vector_store, sample_chunks):
        """``k`` bounds the number of returned chunks."""
        results = vector_store.search(sample_chunks[0].text, k=2)
        assert len(results) == 2

    def test_search_with_scores_is_ranked(self, vector_store, sample_chunks):
        """``include_scores`` returns (chunk, score) tuples ordered best-first."""
        target = sample_chunks[0]
        results = vector_store.search(target.text, k=len(sample_chunks), include_scores=True)

        assert all(isinstance(result, tuple) for result in results)
        top_chunk, _ = results[0]
        assert top_chunk.text == target.text
        # Exact match is the most similar, so scores are non-increasing.
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_clean_collection_removes_it(self, embedding_model):
        """Dropping a collection removes its backing table from the database."""
        store = PGVectorStore(embedding_model=embedding_model, config=PGVectorConfig.from_env())
        try:
            store.add_documents([AI4RAGChunk(text="ephemeral", metadata={"document_id": "tmp", "sequence_number": 0})])
            assert _table_exists(store)

            store.clean_collection()

            assert not _table_exists(store)
        finally:
            store.close()


def _index_names(store: PGVectorStore) -> set[str]:
    """Return the names of every index currently defined on the store's table."""

    async def _fetch() -> list:
        async with store._db.pool.acquire() as conn:
            return await conn.fetch("SELECT indexname FROM pg_indexes WHERE tablename = $1", store.collection_name)

    rows = store._run(_fetch())
    return {row["indexname"] for row in rows}


@pytest.mark.pgvector
class TestPGVectorHighDimensionIntegration:
    """Verifies the >2000-dim fallback against a real pgvector extension.

    pgvector caps HNSW/IVFFlat indexes at 2000 dimensions
    (see :attr:`PGVectorStore._MAX_INDEXABLE_DIMENSION`); above that, the store must
    still store, index (full-text only), and search correctly — just without an ANN
    index on the embedding column. This class exercises that against a live server,
    since ``TestPGVectorIntegration`` above only covers the ≤2000-dim path and the
    unit suite only mocks the DDL rather than running it against a real pgvector
    extension.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def vector_store(sample_chunks):
        store = PGVectorStore(
            embedding_model=DeterministicEmbeddingModel(dimension=2100),
            config=PGVectorConfig.from_env(),
            collection_name="ai4rag_integration_test_highdim",
        )
        store.add_documents(sample_chunks)
        try:
            yield store
        finally:
            store.clean_collection()
            store.close()

    def test_hnsw_index_is_not_created_but_gin_is(self, vector_store):
        """Triggering index creation (first search) skips HNSW but still builds GIN."""
        vector_store.search("anything", k=1)  # triggers _ensure_indexes

        names = _index_names(vector_store)
        assert not any("hnsw" in name for name in names)
        assert any("gin" in name for name in names)

    def test_search_returns_exact_relevant_chunk(self, vector_store, sample_chunks):
        """Without an ANN index, search still returns the exact nearest match."""
        target = sample_chunks[0]
        results = vector_store.search(target.text, k=len(sample_chunks))

        assert results, "search returned no results"
        assert results[0].text == target.text
        assert results[0].metadata["document_id"] == target.metadata["document_id"]

    def test_hybrid_search_still_works(self, vector_store, sample_chunks):
        """Hybrid (dense + keyword) fusion is unaffected by the missing ANN index."""
        target = sample_chunks[0]
        results = vector_store.search(target.text, k=len(sample_chunks), search_mode="hybrid", ranker_strategy="rrf")

        assert results
        assert target.text in {result.text for result in results}
