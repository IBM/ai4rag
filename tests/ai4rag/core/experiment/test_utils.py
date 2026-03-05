# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pandas as pd
import pytest
from langchain_core.documents import Document

from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.exception_handler import GenerationError
from ai4rag.core.experiment.utils import (
    RAGExperimentError,
    build_evaluation_data,
    get_chunking_params,
    get_retrieval_params,
    query_rag,
)
from ai4rag.evaluator.base_evaluator import EvaluationData


class TestQueryRag:
    """Test suite for query_rag function."""

    @pytest.fixture
    def mock_rag(self, mocker):
        """Create a mock RAG template."""
        mock = mocker.MagicMock()
        mock.foundation_model.model_id = "test-model-id"
        mock.generate.side_effect = lambda q: {
            "question": q,
            "answer": f"Answer to {q}",
            "reference_documents": [
                Document(page_content="Doc content", metadata={"document_id": "doc1"}),
            ],
        }
        return mock

    def test_query_rag_single_question(self, mock_rag, mocker):
        """Test query_rag with a single question."""
        mock_logger = mocker.patch("ai4rag.core.experiment.utils.logger")

        questions = ["What is AI?"]
        responses = query_rag(mock_rag, questions)

        assert len(responses) == 1
        assert responses[0]["question"] == "What is AI?"
        assert responses[0]["answer"] == "Answer to What is AI?"
        mock_logger.debug.assert_called()

    def test_query_rag_multiple_questions(self, mock_rag):
        """Test query_rag with multiple questions."""
        questions = ["What is AI?", "What is ML?", "What is DL?"]
        responses = query_rag(mock_rag, questions)

        assert len(responses) == 3
        assert all("question" in r for r in responses)
        assert all("answer" in r for r in responses)

    def test_query_rag_with_custom_max_threads(self, mock_rag):
        """Test query_rag with custom max_threads parameter."""
        questions = ["Q1", "Q2"]
        responses = query_rag(mock_rag, questions, max_threads=2)

        assert len(responses) == 2

    def test_query_rag_logs_debug_messages(self, mock_rag, mocker):
        """Test that query_rag logs debug messages."""
        mock_logger = mocker.patch("ai4rag.core.experiment.utils.logger")

        questions = ["Test question"]
        query_rag(mock_rag, questions)

        assert mock_logger.debug.call_count == 2  # Start and finish messages
        log_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Starting concurrent RAG execution" in msg for msg in log_calls)
        assert any("Finished concurrent RAG execution" in msg for msg in log_calls)

    def test_query_rag_raises_generation_error_on_exception(self, mocker):
        """Test that query_rag raises GenerationError when RAG fails."""
        mock_rag = mocker.MagicMock()
        mock_rag.foundation_model.model_id = "test-model"
        mock_rag.generate.side_effect = Exception("Test error")

        questions = ["Test question"]

        with pytest.raises(GenerationError) as exc_info:
            query_rag(mock_rag, questions)

        assert "test-model" in str(exc_info.value)

    def test_query_rag_empty_questions_list(self, mock_rag):
        """Test query_rag with empty questions list."""
        questions = []
        responses = query_rag(mock_rag, questions)

        assert responses == []


class TestBuildEvaluationData:
    """Test suite for build_evaluation_data function."""

    @pytest.fixture
    def benchmark_data(self):
        """Create sample benchmark data."""
        df = pd.DataFrame(
            {
                "question": ["What is AI?", "What is ML?"],
                "correct_answers": [["AI is artificial intelligence"], ["ML is machine learning"]],
                "correct_answer_document_ids": [["doc1"], ["doc2"]],
            }
        )
        return BenchmarkData(df)

    @pytest.fixture
    def inference_response(self):
        """Create sample inference response."""
        return [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence.",
                "reference_documents": [
                    Document(page_content="AI context", metadata={"document_id": "doc1"}),
                    Document(page_content="More AI context", metadata={"document_id": "doc2"}),
                ],
            },
            {
                "question": "What is ML?",
                "answer": "ML is machine learning.",
                "reference_documents": [
                    Document(page_content="ML context", metadata={"document_id": "doc3"}),
                ],
            },
        ]

    def test_build_evaluation_data_structure(self, benchmark_data, inference_response):
        """Test that build_evaluation_data returns correct structure."""
        result = build_evaluation_data(benchmark_data, inference_response)

        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, EvaluationData) for item in result)

    def test_build_evaluation_data_first_entry(self, benchmark_data, inference_response):
        """Test first entry in evaluation data."""
        result = build_evaluation_data(benchmark_data, inference_response)

        first_entry = result[0]
        assert first_entry.question == "What is AI?"
        assert first_entry.answer == "AI is artificial intelligence."
        assert first_entry.question_id == "q0"
        assert first_entry.ground_truths == ["AI is artificial intelligence"]

    def test_build_evaluation_data_contexts(self, benchmark_data, inference_response):
        """Test that contexts are correctly extracted."""
        result = build_evaluation_data(benchmark_data, inference_response)

        first_entry = result[0]
        assert len(first_entry.contexts) == 2
        assert first_entry.contexts[0] == "AI context"
        assert first_entry.contexts[1] == "More AI context"

    def test_build_evaluation_data_context_ids(self, benchmark_data, inference_response):
        """Test that context_ids are correctly extracted."""
        result = build_evaluation_data(benchmark_data, inference_response)

        first_entry = result[0]
        assert len(first_entry.context_ids) == 2
        assert first_entry.context_ids[0] == "doc1"
        assert first_entry.context_ids[1] == "doc2"

    def test_build_evaluation_data_ground_truths_context_ids(self, benchmark_data, inference_response):
        """Test that ground_truths_context_ids are correctly set."""
        result = build_evaluation_data(benchmark_data, inference_response)

        first_entry = result[0]
        assert first_entry.ground_truths_context_ids == ["doc1"]

    def test_build_evaluation_data_with_empty_document_ids(self, inference_response):
        """Test build_evaluation_data when benchmark_data has empty document_ids."""
        df = pd.DataFrame(
            {
                "question": ["What is AI?", "What is ML?"],
                "correct_answers": [["AI is artificial intelligence"], ["ML is machine learning"]],
                "correct_answer_document_ids": [[], []],
            }
        )
        benchmark_data = BenchmarkData(df)

        result = build_evaluation_data(benchmark_data, inference_response)

        # Should still work with empty document IDs
        assert result[0].ground_truths_context_ids == []
        assert result[1].ground_truths_context_ids == []

    def test_build_evaluation_data_with_missing_page_content(self, benchmark_data):
        """Test handling of documents without page_content attribute."""
        inference_response = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence.",
                "reference_documents": [
                    type("Doc", (), {"metadata": {"document_id": "doc1"}})(),
                ],
            },
            {
                "question": "What is ML?",
                "answer": "ML is machine learning.",
                "reference_documents": [],
            },
        ]

        result = build_evaluation_data(benchmark_data, inference_response)

        assert result[0].contexts[0] is None
        assert result[0].context_ids[0] == "doc1"

    def test_build_evaluation_data_with_missing_metadata(self, benchmark_data):
        """Test handling of documents without metadata."""
        inference_response = [
            {
                "question": "What is AI?",
                "answer": "AI is artificial intelligence.",
                "reference_documents": [
                    type("Doc", (), {"page_content": "Content"})(),
                ],
            },
            {
                "question": "What is ML?",
                "answer": "ML is machine learning.",
                "reference_documents": [],
            },
        ]

        result = build_evaluation_data(benchmark_data, inference_response)

        assert result[0].contexts[0] == "Content"
        assert result[0].context_ids[0] is None


class TestGetChunkingParams:
    """Test suite for get_chunking_params function."""

    def test_get_chunking_params_with_int_overlap(self):
        """Test get_chunking_params with integer chunk_overlap."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 128,
        }

        result = get_chunking_params(rag_params)

        assert result["chunking_method"] == "recursive"
        assert result["chunk_size"] == 512
        assert result["chunk_overlap"] == 128

    def test_get_chunking_params_with_float_overlap(self):
        """Test get_chunking_params with float chunk_overlap (percentage)."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 0.25,
        }

        result = get_chunking_params(rag_params)

        assert result["chunking_method"] == "recursive"
        assert result["chunk_size"] == 512
        assert result["chunk_overlap"] == 128  # 25% of 512

    def test_get_chunking_params_with_zero_float_overlap(self):
        """Test get_chunking_params with zero float chunk_overlap."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 0.0,
        }

        result = get_chunking_params(rag_params)

        assert result["chunk_overlap"] == 0

    def test_get_chunking_params_raises_on_missing_method(self):
        """Test that get_chunking_params raises error when chunking_method is missing."""
        rag_params = {
            "chunk_size": 512,
            "chunk_overlap": 128,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_chunking_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)

    def test_get_chunking_params_raises_on_missing_chunk_size(self):
        """Test that get_chunking_params raises error when chunk_size is missing."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_overlap": 128,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_chunking_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)

    def test_get_chunking_params_raises_on_missing_chunk_overlap(self):
        """Test that get_chunking_params raises error when chunk_overlap is missing."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_chunking_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)

    def test_get_chunking_params_raises_on_invalid_float_overlap_negative(self):
        """Test that invalid negative float chunk_overlap raises error."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": -0.1,
        }

        with pytest.raises(ValueError) as exc_info:
            get_chunking_params(rag_params)

        assert "between 0 and 1" in str(exc_info.value)

    def test_get_chunking_params_raises_on_invalid_float_overlap_greater_than_one(self):
        """Test that float chunk_overlap > 1 raises error."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 1.5,
        }

        with pytest.raises(ValueError) as exc_info:
            get_chunking_params(rag_params)

        assert "between 0 and 1" in str(exc_info.value)

    def test_get_chunking_params_with_max_float_overlap(self):
        """Test get_chunking_params with maximum float overlap (1.0)."""
        rag_params = {
            "chunking_method": "recursive",
            "chunk_size": 512,
            "chunk_overlap": 1.0,
        }

        result = get_chunking_params(rag_params)

        assert result["chunk_overlap"] == 512


class TestGetRetrievalParams:
    """Test suite for get_retrieval_params function."""

    def test_get_retrieval_params_valid(self):
        """Test get_retrieval_params with valid parameters."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 3,
            "number_of_chunks": 5,
        }

        result = get_retrieval_params(rag_params)

        assert result["retrieval_method"] == "simple"
        assert result["window_size"] == 3
        assert result["number_of_chunks"] == 5

    def test_get_retrieval_params_window_method(self):
        """Test get_retrieval_params with window retrieval method."""
        rag_params = {
            "retrieval_method": "window",
            "window_size": 5,
            "number_of_chunks": 10,
        }

        result = get_retrieval_params(rag_params)

        assert result["retrieval_method"] == "window"
        assert result["window_size"] == 5
        assert result["number_of_chunks"] == 10

    def test_get_retrieval_params_zero_window_size(self):
        """Test get_retrieval_params with zero window_size (valid for simple method)."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 0,
            "number_of_chunks": 5,
        }

        result = get_retrieval_params(rag_params)

        assert result["window_size"] == 0

    def test_get_retrieval_params_raises_on_missing_method(self):
        """Test that get_retrieval_params raises error when retrieval_method is missing."""
        rag_params = {
            "window_size": 3,
            "number_of_chunks": 5,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_retrieval_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)

    def test_get_retrieval_params_raises_on_missing_window_size(self):
        """Test that get_retrieval_params raises error when window_size is missing."""
        rag_params = {
            "retrieval_method": "simple",
            "number_of_chunks": 5,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_retrieval_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)

    def test_get_retrieval_params_raises_on_missing_number_of_chunks(self):
        """Test that get_retrieval_params raises error when number_of_chunks is missing."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 3,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_retrieval_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)

    def test_get_retrieval_params_raises_on_none_method(self):
        """Test that get_retrieval_params raises error when method is None."""
        rag_params = {
            "retrieval_method": None,
            "window_size": 3,
            "number_of_chunks": 5,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_retrieval_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)


class TestGetRetrievalParamsHybridSearch:
    """Test suite for get_retrieval_params with hybrid search parameters."""

    def test_get_retrieval_params_with_hybrid_search(self):
        """Test get_retrieval_params with all hybrid search parameters."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 0,
            "number_of_chunks": 5,
            "search_mode": "hybrid",
            "ranker_strategy": "rrf",
            "ranker_k": 60,
            "ranker_alpha": 0,
        }

        result = get_retrieval_params(rag_params)

        assert result["search_mode"] == "hybrid"
        assert result["ranker_strategy"] == "rrf"
        assert result["ranker_k"] == 60
        assert result["ranker_alpha"] == 0

    def test_get_retrieval_params_defaults_to_none_when_not_present(self):
        """Test get_retrieval_params defaults hybrid params to None when not present."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 0,
            "number_of_chunks": 5,
        }

        result = get_retrieval_params(rag_params)

        assert result["search_mode"] is None
        assert result["ranker_strategy"] is None
        assert result["ranker_k"] is None
        assert result["ranker_alpha"] is None

    def test_get_retrieval_params_with_keyword_mode(self):
        """Test get_retrieval_params with keyword search mode."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 0,
            "number_of_chunks": 5,
            "search_mode": "keyword",
        }

        result = get_retrieval_params(rag_params)

        assert result["search_mode"] == "keyword"
        assert result["ranker_strategy"] is None

    def test_get_retrieval_params_with_weighted_ranker(self):
        """Test get_retrieval_params with weighted ranker and alpha."""
        rag_params = {
            "retrieval_method": "simple",
            "window_size": 0,
            "number_of_chunks": 5,
            "search_mode": "hybrid",
            "ranker_strategy": "weighted",
            "ranker_k": 60,
            "ranker_alpha": 0.7,
        }

        result = get_retrieval_params(rag_params)

        assert result["ranker_strategy"] == "weighted"
        assert result["ranker_alpha"] == 0.7

    def test_get_retrieval_params_still_validates_required_fields(self):
        """Test that missing required fields still raise error even with hybrid params."""
        rag_params = {
            "search_mode": "hybrid",
            "ranker_strategy": "rrf",
            "ranker_k": 60,
        }

        with pytest.raises(RAGExperimentError) as exc_info:
            get_retrieval_params(rag_params)

        assert "Missing or invalid values" in str(exc_info.value)


class TestRAGExperimentError:
    """Test suite for RAGExperimentError exception."""

    def test_rag_experiment_error_creation(self):
        """Test creating RAGExperimentError."""
        error = RAGExperimentError("Test error message")
        assert str(error) == "Test error message"

    def test_rag_experiment_error_is_exception(self):
        """Test that RAGExperimentError is an Exception."""
        assert issubclass(RAGExperimentError, Exception)

    def test_rag_experiment_error_can_be_raised(self):
        """Test that RAGExperimentError can be raised and caught."""
        with pytest.raises(RAGExperimentError) as exc_info:
            raise RAGExperimentError("Custom error")

        assert "Custom error" in str(exc_info.value)
