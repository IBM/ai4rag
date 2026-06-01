# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for AI4RAGExperiment._stream_finished_pattern payload construction."""

import pandas as pd
import pytest
from langchain_core.documents import Document

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.experiment.results import EvaluationResult, ExperimentResults
from ai4rag.core.hpo.random_opt import RandomOptSettings
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.mocks import MockedEmbeddingModel, MockedFoundationModel, MockedOGXClient

_EMBEDDING_DIMENSION = 64


def _make_search_space(fm, em):
    return AI4RAGSearchSpace(
        vector_store_type="chroma",
        params=[
            Parameter(name="foundation_model", param_type="C", values=[fm]),
            Parameter(name="embedding_model", param_type="C", values=[em]),
        ],
    )


def _make_evaluation_result(
    vector_io_provider_type="chroma::local",
    search_mode="vector",
    window_size=None,
    ranker_strategy=None,
    ranker_k=None,
    ranker_alpha=None,
):
    return EvaluationResult(
        pattern_name="Pattern1",
        collection="test-collection-abc",
        indexing_params={
            "chunking": {
                AI4RAGParamNames.CHUNKING_METHOD: "recursive",
                AI4RAGParamNames.CHUNK_SIZE: 512,
                AI4RAGParamNames.CHUNK_OVERLAP: 64,
            },
            "embedding": {
                "model_id": "mock-em-0",
                "distance_metric": "cosine",
            },
        },
        rag_params={
            "retrieval": {
                AI4RAGParamNames.RETRIEVAL_METHOD: "simple",
                AI4RAGParamNames.NUMBER_OF_CHUNKS: 3,
                AI4RAGParamNames.SEARCH_MODE: search_mode,
                AI4RAGParamNames.WINDOW_SIZE: window_size,
                AI4RAGParamNames.RANKER_STRATEGY: ranker_strategy,
                AI4RAGParamNames.RANKER_K: ranker_k,
                AI4RAGParamNames.RANKER_ALPHA: ranker_alpha,
            },
            "generation": {
                "model_id": "mock-fm-0",
                "context_template_text": "Context: {context}",
                "user_message_text": "Answer: {question}",
                "system_message_text": "You are a helpful assistant.",
            },
            "vector_io_provider_type": vector_io_provider_type,
        },
        scores={
            "scores": {"answer_correctness": {"mean": 0.5}},
            "question_scores": {"answer_correctness": {"q0": 0.5}},
        },
        execution_time=10.0,
        final_score=0.5,
    )


@pytest.fixture
def foundation_model():
    return MockedFoundationModel(model_id="mock-fm-0", params=None)


@pytest.fixture
def embedding_model():
    return MockedEmbeddingModel(
        model_id="mock-em-0",
        params={"embedding_dimension": _EMBEDDING_DIMENSION},
    )


@pytest.fixture
def minimal_documents():
    return [Document(page_content="Test content.", metadata={"document_id": "doc_0"})]


@pytest.fixture
def minimal_benchmark():
    return pd.DataFrame(
        {
            "question": ["What is test?"],
            "correct_answers": [["Test content."]],
            "correct_answer_document_ids": [["doc_0"]],
        }
    )


def _make_chroma_experiment(foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker):
    event_handler = mocker.MagicMock(spec=LocalEventHandler)
    experiment = AI4RAGExperiment(
        documents=minimal_documents,
        benchmark_data=minimal_benchmark,
        search_space=_make_search_space(foundation_model, embedding_model),
        vector_store_type="chroma",
        optimizer_settings=RandomOptSettings(max_evals=1),
        event_handler=event_handler,
    )
    experiment.results = ExperimentResults()
    return experiment


def _make_ogx_experiment(foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker):
    event_handler = mocker.MagicMock(spec=LocalEventHandler)
    experiment = AI4RAGExperiment(
        documents=minimal_documents,
        benchmark_data=minimal_benchmark,
        search_space=_make_search_space(foundation_model, embedding_model),
        vector_store_type="ogx",
        optimizer_settings=RandomOptSettings(max_evals=1),
        event_handler=event_handler,
        client=MockedOGXClient(),
        ogx_vector_io_provider_id="test-provider",
    )
    experiment.results = ExperimentResults()
    return experiment


class TestStreamFinishedPatternChroma:

    def test_chroma_payload_excludes_responses_template(
        self, foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
    ):
        experiment = _make_chroma_experiment(
            foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
        )
        eval_result = _make_evaluation_result(vector_io_provider_type="chroma::local")

        experiment._stream_finished_pattern(eval_result, [])

        experiment.event_handler.on_pattern_creation.assert_called_once()
        payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]

        assert "responses_template" not in payload
        binding = payload["settings"]["vector_store_binding"]
        assert "provider_id" in binding
        assert "provider_type" in binding
        assert "vector_store_id" in binding
        assert "vector_store_name" in binding
        assert binding["provider_type"] == "chroma::local"

    def test_chroma_payload_has_required_top_level_keys(
        self, foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
    ):
        experiment = _make_chroma_experiment(
            foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
        )
        eval_result = _make_evaluation_result()

        experiment._stream_finished_pattern(eval_result, [])

        payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]
        expected_keys = {"pattern_name", "scores", "execution_time", "final_score", "schema_version", "producer", "settings", "iteration"}
        assert expected_keys.issubset(payload.keys())


class TestStreamFinishedPatternOGX:

    def test_ogx_payload_includes_responses_template(
        self, foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
    ):
        experiment = _make_ogx_experiment(
            foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
        )
        eval_result = _make_evaluation_result(vector_io_provider_type="mock_provider")

        experiment._stream_finished_pattern(eval_result, [])

        payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]

        assert "responses_template" in payload
        responses = payload["responses_template"]
        assert responses["model"] == "mock-fm-0"
        assert responses["tools"][0]["type"] == "file_search"
        assert responses["tools"][0]["vector_store_ids"] == ["test-collection-abc"]
        assert responses["include"] == ["file_search_call.results"]
        assert responses["stream"] is False
        assert responses["store"] is True


class TestStreamFinishedPatternRetrieval:

    def test_hybrid_retrieval_includes_ranker_fields(
        self, foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
    ):
        experiment = _make_chroma_experiment(
            foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
        )
        eval_result = _make_evaluation_result(
            search_mode="hybrid",
            ranker_strategy="rrf",
            ranker_k=60,
            ranker_alpha=0.5,
        )

        experiment._stream_finished_pattern(eval_result, [])

        payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]
        retrieval = payload["settings"]["retrieval"]
        assert retrieval["search_mode"] == "hybrid"
        assert retrieval["ranker_strategy"] == "rrf"
        assert retrieval["ranker_k"] == 60
        assert retrieval["ranker_alpha"] == 0.5

    def test_window_size_included_when_set(
        self, foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
    ):
        experiment = _make_chroma_experiment(
            foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
        )
        eval_result = _make_evaluation_result(window_size=2)

        experiment._stream_finished_pattern(eval_result, [])

        payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]
        assert payload["settings"]["retrieval"]["window_size"] == 2

    def test_window_size_excluded_when_none(
        self, foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
    ):
        experiment = _make_chroma_experiment(
            foundation_model, embedding_model, minimal_documents, minimal_benchmark, mocker
        )
        eval_result = _make_evaluation_result(window_size=None)

        experiment._stream_finished_pattern(eval_result, [])

        payload = experiment.event_handler.on_pattern_creation.call_args.kwargs["payload"]
        assert "window_size" not in payload["settings"]["retrieval"]
