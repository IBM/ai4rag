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

pytestmark = pytest.mark.skipif(
    os.environ.get("PGVECTOR_HOST") is None,
    reason="PGVECTOR_HOST is not set; skipping live pgvector integration tests.",
)


def _table_exists(store: PGVectorStore) -> bool:
    """Return whether the store's backing table currently exists in the database."""
    # ``to_regclass`` resolves a relation name to its OID, or NULL if it does not
    # exist — a non-throwing existence check. The collection name IS the table
    # name; ``_quoted_table`` supplies the safely-quoted identifier.
    row = store._conn.execute("SELECT to_regclass(%s)", (store._quoted_table(),)).fetchone()
    return row is not None and row[0] is not None


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
