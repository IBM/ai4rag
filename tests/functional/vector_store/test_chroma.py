# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional semantic-retrieval test for :class:`ChromaVectorStore`.

Chroma runs fully in-memory here (an ephemeral :class:`ChromaConfig`), so this
test needs no Chroma server — only MaaS credentials for the real embedding model,
which the shared ``embedding_model`` fixture enforces by skipping when they are
absent. Chroma has no lexical search, so — unlike the Milvus and pgvector
modules — this one covers dense semantic retrieval only.
"""

import pytest

from ai4rag.rag.vector_store.chroma import ChromaVectorStore
from ai4rag.rag.vector_store.config import ChromaConfig


@pytest.fixture(scope="module")
def vector_store(embedding_model, story_chunks):
    """Build an in-memory Chroma store populated with the story; drop it on teardown."""
    store = ChromaVectorStore(embedding_model=embedding_model, config=ChromaConfig())
    store.add_documents(story_chunks)
    try:
        yield store
    finally:
        store.clean_collection()


def test_question_retrieves_expected_passage(vector_store, check_retrieval):
    """Each story question retrieves the single passage that answers it."""
    check_retrieval(lambda question: vector_store.search(question, k=1))
