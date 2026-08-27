# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for :class:`Neo4jGraphStore`.

Exercises the full stack — embedding model + vector store — from the public
API surface only (no mocking). Skipped unless ``NEO4J_URI`` and
``NEO4J_PASSWORD`` are set.
"""

import os

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j import Neo4jGraphStore

pytestmark = pytest.mark.skipif(
    os.environ.get("NEO4J_URI") is None,
    reason="NEO4J_URI is not set; skipping live Neo4j functional tests.",
)


@pytest.fixture(scope="module")
def neo4j_store(embedding_model, sample_chunks):
    """Create and populate a uniquely named collection; drop it on teardown."""
    store = Neo4jGraphStore(
        embedding_model=embedding_model,
        config=Neo4jConfig.from_env(),
        collection_name="ai4rag_functional_test_neo4j",
    )
    store.add_documents(sample_chunks)
    try:
        yield store
    finally:
        store.clean_collection()
        store.close()


def test_vector_search_round_trip(neo4j_store, sample_chunks):
    """add_documents then search(mode='vector') returns the expected chunk."""
    query = sample_chunks[0].text
    results = neo4j_store.search(query, k=3)
    texts = [r.text for r in results]
    assert query in texts


def test_hybrid_search_returns_results(neo4j_store, sample_chunks):
    """Hybrid search returns results, potentially ranked differently from vector."""
    results = neo4j_store.search(
        sample_chunks[0].text, k=3, search_mode="hybrid", ranker_strategy="rrf"
    )
    assert len(results) > 0
    assert all(isinstance(r, AI4RAGChunk) for r in results)


def test_graph_search_expands_sequential_neighbors(neo4j_store, sample_chunks):
    """Graph search with graph_hops=1 can return sequential neighbor chunks."""
    results = neo4j_store.search(
        sample_chunks[0].text,
        k=10,
        search_mode="graph",
        graph_hops=1,
        include_entity_neighbors=False,
    )
    assert len(results) > 0
    assert all(isinstance(r, AI4RAGChunk) for r in results)


def test_collection_name_prefix_guard():
    """A collection name without the ai4rag_ prefix must raise ValueError."""
    with pytest.raises(ValueError, match="ai4rag"):
        Neo4jGraphStore(
            embedding_model=None,
            config=Neo4jConfig.from_env(),
            collection_name="no_prefix_here",
        )


def test_context_manager_does_not_raise(embedding_model, sample_chunks):
    """Using Neo4jGraphStore as a context manager must not raise."""
    with Neo4jGraphStore(
        embedding_model=embedding_model,
        config=Neo4jConfig.from_env(),
        collection_name="ai4rag_ctx_test_neo4j",
    ) as store:
        store.add_documents(sample_chunks[:1])
        results = store.search(sample_chunks[0].text, k=1)
        assert len(results) > 0
    # close() was called — no exception expected
