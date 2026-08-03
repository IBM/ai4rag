# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional semantic-retrieval test for :class:`PGVectorStore`.

Runs against a live PostgreSQL + pgvector server (skipped unless ``PGVECTOR_HOST``
is set) using a real OGX embedding model. pgvector is strongly consistent, so
reads need no retry; the store holds an open connection, which teardown closes.

This module is where pgvector-specific retrieval tests (e.g. future lexical /
hybrid search over ``tsvector``) will live; the dense semantic check below is
shared with the other backends via ``check_retrieval``.
"""

import os

import pytest

from ai4rag.rag.vector_store.config import PGVectorConfig
from ai4rag.rag.vector_store.pgvector import PGVectorStore

pytestmark = pytest.mark.skipif(
    os.environ.get("PGVECTOR_HOST") is None,
    reason="PGVECTOR_HOST is not set; skipping live pgvector functional tests.",
)


@pytest.fixture(scope="module")
def vector_store(embedding_model, story_chunks):
    """Build a live pgvector store populated with the story; drop it and close on teardown."""
    store = PGVectorStore(embedding_model=embedding_model, config=PGVectorConfig.from_env())
    store.add_documents(story_chunks)
    try:
        yield store
    finally:
        store.clean_collection()
        store.close()


def test_question_retrieves_expected_passage(vector_store, check_retrieval):
    """Each story question retrieves the single passage that answers it."""
    check_retrieval(lambda question: vector_store.search(question, k=1))
