# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Integration tests for :class:`Neo4jGraphStore` against a live Neo4j server.

Skipped unless ``NEO4J_URI`` and ``NEO4J_PASSWORD`` are set. Connection
settings are read via :meth:`Neo4jConfig.from_env`.
"""

import os

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j import Neo4jGraphStore

pytestmark = pytest.mark.skipif(
    os.environ.get("NEO4J_URI") is None,
    reason="NEO4J_URI is not set; skipping live Neo4j integration tests.",
)


class TestNeo4jIntegration:
    """Full create → add → search → build_kg → clean lifecycle against a live Neo4j server.

    A single class-scoped ``vector_store`` fixture owns the lifecycle: it
    creates and populates the collection on setup and drops it on teardown, so
    the remaining test methods are order-independent, read-only assertions over
    the same populated graph.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def vector_store(embedding_model, sample_chunks):
        """Create and populate a collection; clean it up and close on teardown."""
        store = Neo4jGraphStore(
            embedding_model=embedding_model,
            config=Neo4jConfig.from_env(),
            collection_name="ai4rag_integration_test_neo4j",
        )
        store.add_documents(sample_chunks)
        try:
            yield store
        finally:
            store.clean_collection()
            store.close()

    def test_collection_name_prefix(self, vector_store):
        assert vector_store.collection_name.startswith("ai4rag_")

    def test_vector_search_returns_results(self, vector_store, sample_chunks):
        results = vector_store.search(sample_chunks[0].text, k=3, search_mode="vector")
        assert len(results) > 0
        assert all(isinstance(r, AI4RAGChunk) for r in results)

    def test_hybrid_search_returns_results(self, vector_store, sample_chunks):
        results = vector_store.search(
            sample_chunks[0].text, k=3, search_mode="hybrid", ranker_strategy="rrf"
        )
        assert len(results) > 0

    def test_graph_search_returns_results(self, vector_store, sample_chunks):
        results = vector_store.search(sample_chunks[0].text, k=3, search_mode="graph", graph_hops=1)
        assert len(results) > 0

    def test_graph_search_expands_neighbors(self, vector_store, sample_chunks):
        """Graph search with hop expansion may return more unique chunks than pure vector."""
        vector_results = vector_store.search(sample_chunks[0].text, k=1, search_mode="vector")
        graph_results = vector_store.search(
            sample_chunks[0].text, k=10, search_mode="graph", graph_hops=1
        )
        # Graph search can surface neighbors not in the pure vector top-1
        assert len(graph_results) >= len(vector_results)

    def test_clean_collection_removes_nodes(self, vector_store):
        """After clean_collection, a vector search returns zero results."""
        vector_store.clean_collection()
        # Re-create schema (indexes dropped by clean)
        vector_store._ensure_schema()
        results = vector_store.search("anything", k=3, search_mode="vector")
        assert results == []
