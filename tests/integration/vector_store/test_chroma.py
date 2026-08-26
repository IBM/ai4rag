# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Integration tests for :class:`ChromaVectorStore` against a live Chroma server.

These tests exercise the full collection lifecycle — create, add embeddings,
search, drop — over the network against a running Chroma server, and are skipped
unless ``CHROMA_HOST`` is set (the in-memory and persistent modes are covered by
the unit suite). Connection settings are read via :meth:`ChromaConfig.from_env`
(``CHROMA_HOST``, ``CHROMA_PORT``, ``CHROMA_PERSIST_DIR``); see
``tests/integration/conftest.py`` for how ``.env.local`` is loaded.
If no env variables are set, in-memory chroma with no persistence is used.
"""

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.chroma import ChromaVectorStore
from ai4rag.rag.vector_store.config import ChromaConfig


def _collection_exists(store: ChromaVectorStore, name: str) -> bool:
    """Return whether a collection named *name* exists on the connected server."""
    return name in {collection.name for collection in store._client.list_collections()}


@pytest.mark.chroma
class TestChromaIntegration:
    """Full create → add → search → drop lifecycle against a live Chroma server.

    A single class-scoped ``vector_store`` fixture owns the lifecycle: it creates
    and populates the collection on setup and drops it on teardown, so the
    remaining test methods are order-independent, read-only assertions over the
    same populated collection.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def vector_store(embedding_model, sample_chunks):
        """Create and populate a uniquely named collection; drop it on teardown."""
        store = ChromaVectorStore(embedding_model=embedding_model, config=ChromaConfig.from_env())
        store.add_documents(sample_chunks)
        try:
            yield store
        finally:
            store.clean_collection()

    def test_collection_is_created(self, vector_store):
        """The store's collection exists on the connected server."""
        assert vector_store.collection_name.startswith("ai4rag")
        assert _collection_exists(vector_store, vector_store.collection_name)

    def test_all_documents_are_added(self, vector_store, sample_chunks):
        """Every added chunk is persisted and independently retrievable."""
        assert vector_store.count() == len(sample_chunks)
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
        top_chunk, top_score = results[0]
        assert top_chunk.text == target.text
        # Exact match is the most similar, so scores are non-increasing.
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)
        assert top_score == pytest.approx(1.0, abs=1e-4)

    def test_clean_collection_removes_it(self, embedding_model):
        """Dropping a collection removes it from the server."""
        store = ChromaVectorStore(embedding_model=embedding_model, config=ChromaConfig.from_env())
        store.add_documents([AI4RAGChunk(text="ephemeral", metadata={"document_id": "tmp", "sequence_number": 0})])
        name = store.collection_name
        assert _collection_exists(store, name)

        store.clean_collection()

        assert not _collection_exists(store, name)
