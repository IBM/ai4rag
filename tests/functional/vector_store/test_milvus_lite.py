# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional test for concurrent querying of Milvus Lite with real MaaS embeddings.

Reproduces the experiment's concurrent retrieval fan-out — ``query_rag`` runs
every question's retrieval on a thread pool against one shared store (see
:func:`ai4rag.core.experiment.utils.query_rag`) — using a *real* MaaS embedding
model and a local, embedded Milvus Lite store. It verifies that the concurrent
embed-then-search path returns the correct passage for every question with no
errors: the semantic guarantee an experiment depends on when it queries the
store from many threads at once.

Needs MaaS credentials (real embeddings); Milvus Lite itself needs no server, so
the store runs from a local ``.db`` file.
"""

import concurrent.futures as cf

import pytest

from ai4rag.rag.vector_store.config import MilvusLiteConfig
from ai4rag.rag.vector_store.milvus import MilvusVectorStore
from tests.functional.vector_store.conftest import STORY_QUESTIONS


@pytest.fixture
def vector_store(embedding_model, story_chunks, tmp_path):
    """A populated, embedded Milvus Lite store on a per-test temporary database."""
    store = MilvusVectorStore(
        embedding_model=embedding_model, config=MilvusLiteConfig(db_path=str(tmp_path / "concurrent.db"))
    )
    store.add_documents(story_chunks)
    try:
        yield store
    finally:
        store.clean_collection()
        store.close()


def test_concurrent_retrieval_returns_expected_passages(vector_store, check_retrieval):
    """Every story question, retrieved concurrently, still returns its answer passage."""
    questions = [question for question, _ in STORY_QUESTIONS]

    with cf.ThreadPoolExecutor(max_workers=len(questions)) as executor:
        # Fire every question's retrieval concurrently against the one shared store,
        # mirroring query_rag(max_threads=...). ``map`` re-raises the first worker
        # exception on iteration, so a concurrency failure fails the test here.
        results_by_question = dict(
            zip(questions, executor.map(lambda question: vector_store.search(question, k=1), questions))
        )

    # Reuse the shared correctness assertion, now served from the concurrently
    # computed results: each question's top hit must be the passage that answers it.
    check_retrieval(lambda question: results_by_question[question])
