# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Integration tests for :class:`MilvusVectorStore` against a live Milvus instance.

These tests exercise the full collection lifecycle — create, add embeddings,
search, drop — against a real Milvus server, and are skipped unless
``MILVUS_URI`` is set. Connection settings are read via
:meth:`MilvusConfig.from_env` (``MILVUS_URI``, ``MILVUS_TOKEN``,
``MILVUS_SERVER_CERT``); see ``tests/integration/conftest.py`` for how
``.env.local`` is loaded.

:class:`MilvusVectorStore` requests ``consistency_level="Strong"`` on every
search (see ``_search_vector``/``_search_hybrid``), so a freshly upserted row is
guaranteed visible on the very next search — Milvus's Bounded-staleness default
would not make that guarantee, which is why the store overrides it. Read
assertions still go through the shared ``retry`` helper as a defensive net
against transient server-side hiccups, not to paper over staleness.
"""

import os

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import MilvusConfig
from ai4rag.rag.vector_store.milvus import MilvusVectorStore

pytestmark = pytest.mark.skipif(
    os.environ.get("MILVUS_URI") is None,
    reason="MILVUS_URI is not set; skipping live Milvus integration tests.",
)


@pytest.mark.milvus
class TestMilvusIntegration:
    """Full create → add → search → drop lifecycle against a live Milvus server.

    A single class-scoped ``vector_store`` fixture owns the lifecycle: it creates
    and populates the collection on setup and drops it on teardown, so the
    remaining test methods are order-independent, read-only assertions over the
    same populated collection.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def vector_store(embedding_model, sample_chunks):
        """Create and populate a uniquely named collection; drop it on teardown."""
        store = MilvusVectorStore(embedding_model=embedding_model, config=MilvusConfig.from_env())
        store.add_documents(sample_chunks)
        try:
            yield store
        finally:
            store.clean_collection()

    def test_collection_is_created(self, vector_store):
        """The store's collection exists on the connected server."""
        assert vector_store.collection_name.startswith("ai4rag")
        assert vector_store._client.has_collection(vector_store.collection_name)

    def test_all_documents_are_added(self, vector_store, sample_chunks, retry):
        """Every added chunk is persisted and independently retrievable."""
        for chunk in sample_chunks:
            results = retry(lambda c=chunk: vector_store.search(c.text, k=len(sample_chunks)))
            assert chunk.text in {result.text for result in results}

    def test_search_returns_relevant_chunk(self, vector_store, sample_chunks, retry):
        """A query for a chunk's exact text ranks that chunk first, metadata intact."""
        target = sample_chunks[0]
        results = retry(lambda: vector_store.search(target.text, k=len(sample_chunks)))

        assert results, "search returned no results"
        assert results[0].text == target.text
        assert results[0].metadata["document_id"] == target.metadata["document_id"]

    def test_search_respects_k(self, vector_store, sample_chunks, retry):
        """``k`` bounds the number of returned chunks."""
        results = retry(lambda: vector_store.search(sample_chunks[0].text, k=2))
        assert len(results) == 2

    def test_search_with_scores_is_ranked(self, vector_store, sample_chunks, retry):
        """``include_scores`` returns (chunk, score) tuples ordered best-first."""
        target = sample_chunks[0]
        results = retry(lambda: vector_store.search(target.text, k=len(sample_chunks), include_scores=True))

        assert results, "search returned no results"
        assert all(isinstance(result, tuple) for result in results)
        top_chunk, _ = results[0]
        assert top_chunk.text == target.text
        # Exact match is the most similar, so fused scores are non-increasing.
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_clean_collection_removes_it(self, embedding_model):
        """Dropping a collection removes it from the server."""
        store = MilvusVectorStore(embedding_model=embedding_model, config=MilvusConfig.from_env())
        store.add_documents([AI4RAGChunk(text="ephemeral", metadata={"document_id": "tmp", "sequence_number": 0})])
        name = store.collection_name
        assert store._client.has_collection(name)

        store.clean_collection()

        assert not store._client.has_collection(name)
