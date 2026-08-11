# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional semantic-retrieval test for :class:`MilvusVectorStore`.

Runs against a live Milvus server (skipped unless ``MILVUS_URI`` is set) using a
real MaaS embedding model. Milvus serves searches under bounded-staleness
consistency, so freshly upserted rows may not be immediately visible; reads are
therefore wrapped in the shared ``retry`` helper.

This module is where Milvus-specific retrieval tests (e.g. future lexical /
hybrid search) will live; the dense semantic check below is shared with the
other backends via ``check_retrieval``.
"""

import os

import pytest

from ai4rag.rag.vector_store.config import MilvusConfig
from ai4rag.rag.vector_store.milvus import MilvusVectorStore

pytestmark = pytest.mark.skipif(
    os.environ.get("MILVUS_URI") is None,
    reason="MILVUS_URI is not set; skipping live Milvus functional tests.",
)


@pytest.fixture(scope="module")
def vector_store(embedding_model, story_chunks):
    """Build a live Milvus store populated with the story; drop it on teardown."""
    store = MilvusVectorStore(embedding_model=embedding_model, config=MilvusConfig.from_env())
    store.add_documents(story_chunks)
    try:
        yield store
    finally:
        store.clean_collection()


def test_question_retrieves_expected_passage(vector_store, check_retrieval, retry):
    """Each story question retrieves the single passage that answers it."""
    check_retrieval(lambda question: retry(lambda: vector_store.search(question, k=1)))
