# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Integration test for concurrent querying of an embedded Milvus Lite store.

The experiment fans retrieval out across a thread pool: ``query_rag`` runs
``ThreadPoolExecutor(max_workers=...).map(...)`` so every question's
``retriever.search()`` executes concurrently against a *single shared* vector
store and client (see :func:`ai4rag.core.experiment.utils.query_rag`). Milvus
Lite is one embedded engine backed by a local ``.db`` file, so this test
verifies the storage-layer contract the experiment relies on: many concurrent
reads on one shared client return correct, uncorrupted results and never raise.

Milvus Lite needs no server (a local file ``uri`` selects it), so — unlike the
remote-Milvus module in this package — this test is *not* gated on ``MILVUS_URI``
and runs anywhere. Embeddings are the local ``DeterministicEmbeddingModel`` from
``conftest`` (no MaaS), so search ordering is reproducible: a query for a chunk's
exact text is guaranteed to rank that chunk first.
"""

import concurrent.futures as cf

import pytest

from ai4rag.rag.vector_store.config import MilvusLiteConfig
from ai4rag.rag.vector_store.milvus import MilvusVectorStore

#: Concurrent workers, matching ``query_rag``'s default ``max_threads``.
_MAX_WORKERS = 10
#: Total concurrent queries — many more than there are chunks, so workers contend
#: on the shared client and every chunk is queried repeatedly under load.
_QUERY_COUNT = 100


@pytest.fixture
def vector_store(embedding_model, sample_chunks, tmp_path):
    """A populated, embedded Milvus Lite store on a per-test temporary database."""
    store = MilvusVectorStore(
        embedding_model=embedding_model, config=MilvusLiteConfig(db_path=str(tmp_path / "concurrent.db"))
    )
    store.add_documents(sample_chunks)
    try:
        yield store
    finally:
        store.clean_collection()
        store.close()


@pytest.mark.milvus
def test_concurrent_search_is_correct_and_error_free(vector_store, sample_chunks):
    """Concurrent dense and hybrid reads on one shared store stay correct and error-free.

    Reproduces the experiment's retrieval fan-out and alternates the two search
    modes an experiment may run (dense ``"vector"`` and dense + BM25 ``"hybrid"``)
    so both server-side paths are exercised under contention.
    """
    queries = [sample_chunks[i % len(sample_chunks)] for i in range(_QUERY_COUNT)]

    def _search(chunk):
        # Alternate modes by the chunk's sequence number so the concurrent load
        # mixes pure-vector and hybrid (fused) searches, as an experiment might.
        hybrid = chunk.metadata["sequence_number"] % 2 == 1
        results = vector_store.search(
            chunk.text,
            k=len(sample_chunks),
            search_mode="hybrid" if hybrid else "vector",
            ranker_strategy="rrf" if hybrid else None,
        )
        return chunk.text, hybrid, results

    with cf.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        # ``map`` re-raises the first worker exception when its result is consumed,
        # so any concurrency failure (e.g. an embedded-engine race) fails the test.
        outcomes = list(executor.map(_search, queries))

    assert len(outcomes) == _QUERY_COUNT
    for query_text, hybrid, results in outcomes:
        assert results, "a concurrent search returned no results"
        if hybrid:
            # Fusion may reorder ties, but the exact-text match must still surface.
            assert query_text in {result.text for result in results}
        else:
            # A pure-vector query for a chunk's own text ranks that chunk first;
            # a wrong top hit would betray results cross-contaminated across threads.
            assert results[0].text == query_text
