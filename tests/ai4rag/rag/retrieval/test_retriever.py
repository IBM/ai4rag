# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from ai4rag.rag.retrieval.retriever import Retriever


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    vs = MagicMock()
    vs.search.return_value = [
        Document(page_content="doc1", metadata={"document_id": "d1"}),
        Document(page_content="doc2", metadata={"document_id": "d2"}),
    ]
    return vs


class TestRetrieverInit:
    """Test Retriever initialization with hybrid search parameters."""

    def test_default_search_mode(self, mock_vector_store):
        retriever = Retriever(vector_store=mock_vector_store, number_of_chunks=5)

        assert retriever.search_mode == "vector"
        assert retriever.ranker_strategy is None
        assert retriever.ranker_k is None
        assert retriever.ranker_alpha is None

    def test_hybrid_search_mode(self, mock_vector_store):
        retriever = Retriever(
            vector_store=mock_vector_store,
            number_of_chunks=5,
            search_mode="hybrid",
            ranker_strategy="rrf",
            ranker_k=60,
        )

        assert retriever.search_mode == "hybrid"
        assert retriever.ranker_strategy == "rrf"
        assert retriever.ranker_k == 60
        assert retriever.ranker_alpha is None


class TestRetrieverRetrieve:
    """Test Retriever.retrieve passes hybrid params to vector store."""

    def test_retrieve_passes_default_vector_mode(self, mock_vector_store):
        retriever = Retriever(vector_store=mock_vector_store, number_of_chunks=5)

        retriever.retrieve("test query")

        mock_vector_store.search.assert_called_once_with(
            "test query",
            k=5,
            search_mode="vector",
            ranker_strategy=None,
            ranker_k=None,
            ranker_alpha=None,
        )

    def test_retrieve_passes_hybrid_params(self, mock_vector_store):
        retriever = Retriever(
            vector_store=mock_vector_store,
            number_of_chunks=3,
            search_mode="hybrid",
            ranker_strategy="rrf",
            ranker_k=60,
        )

        retriever.retrieve("test query")

        mock_vector_store.search.assert_called_once_with(
            "test query",
            k=3,
            search_mode="hybrid",
            ranker_strategy="rrf",
            ranker_k=60,
            ranker_alpha=None,
        )

    def test_retrieve_passes_weighted_ranker(self, mock_vector_store):
        retriever = Retriever(
            vector_store=mock_vector_store,
            number_of_chunks=5,
            search_mode="hybrid",
            ranker_strategy="weighted",
            ranker_k=60,
            ranker_alpha=0.7,
        )

        retriever.retrieve("test query")

        mock_vector_store.search.assert_called_once_with(
            "test query",
            k=5,
            search_mode="hybrid",
            ranker_strategy="weighted",
            ranker_k=60,
            ranker_alpha=0.7,
        )

    def test_retrieve_respects_number_of_chunks_override(self, mock_vector_store):
        retriever = Retriever(
            vector_store=mock_vector_store,
            number_of_chunks=5,
            search_mode="hybrid",
            ranker_strategy="rrf",
            ranker_k=60,
        )

        retriever.retrieve("test query", number_of_chunks=10)

        mock_vector_store.search.assert_called_once_with(
            "test query",
            k=10,
            search_mode="hybrid",
            ranker_strategy="rrf",
            ranker_k=60,
            ranker_alpha=None,
        )
