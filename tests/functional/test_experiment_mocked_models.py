# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Functional tests for AI4RAGExperiment with mocked models.

These tests run a full experiment end-to-end using MockedFoundationModel and
MockedEmbeddingModel to avoid real LLM/embedding API calls, while still
exercising the real evaluation pipeline (UnitxtEvaluator) and verifying that
ModelsPreSelector (MPS) correctly reduces the model pool before HPO.

With 4 foundation models and 3 embedding models:
- MPS thresholds: DEFAULT_N_FOUNDATION_MODELS=3, DEFAULT_N_EMBEDDING_MODELS=2
- Both thresholds are exceeded, so MPS is triggered automatically in every test
  that does not explicitly pass skip_mps=True.
"""

import pandas as pd
import pytest
from langchain_core.documents import Document

from ai4rag.core.experiment.experiment import AI4RAGExperiment
from ai4rag.core.experiment.mps import ModelsPreSelector
from ai4rag.core.hpo.random_opt import RandomOptimizer, RandomOptSettings
from ai4rag.evaluator.base_evaluator import MetricType
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames
from ai4rag.utils.event_handler import LocalEventHandler
from dev_utils.mocks import MockedEmbeddingModel, MockedFoundationModel

_N_FOUNDATION_MODELS = 4
_N_EMBEDDING_MODELS = 3
_EMBEDDING_DIMENSION = 64


@pytest.fixture(scope="module")
def foundation_models():
    """4 mocked foundation models — exceeds DEFAULT_N_FOUNDATION_MODELS=3 to trigger MPS."""
    return [MockedFoundationModel(model_id=f"mock-fm-{i}", params=None) for i in range(_N_FOUNDATION_MODELS)]


@pytest.fixture(scope="module")
def embedding_models():
    """3 mocked embedding models — exceeds DEFAULT_N_EMBEDDING_MODELS=2 to trigger MPS."""
    return [
        MockedEmbeddingModel(
            model_id=f"mock-em-{i}",
            params={"embedding_dimension": _EMBEDDING_DIMENSION},
        )
        for i in range(_N_EMBEDDING_MODELS)
    ]


@pytest.fixture(scope="module")
def mock_client():
    """Mock OGX client for experiments."""
    from dev_utils.mocks import MockedOGXClient
    return MockedOGXClient()


@pytest.fixture(scope="module")
def documents():
    """5 documents with enough content to be split into multiple chunks at chunk_size=512."""
    paragraph = (
        "This document covers topic {topic}. "
        "The concept of {topic} is central to understanding the broader subject. "
        "Researchers have studied {topic} extensively over many decades. "
        "Practical applications of {topic} span engineering, science, and industry. "
        "The historical development of {topic} offers insight into current practices. "
        "Future directions for {topic} research include novel methodologies and tools. "
    )
    long_content = (paragraph * 8).strip()

    return [
        Document(
            page_content=long_content.format(topic=f"topic_{i}"),
            metadata={"document_id": f"doc_{i}"},
        )
        for i in range(5)
    ]


@pytest.fixture(scope="module")
def benchmark_data():
    """3-record benchmark dataset referencing documents doc_0 through doc_2."""
    return pd.DataFrame(
        {
            "question": [
                "What is topic_0 about?",
                "Describe the applications of topic_1.",
                "What have researchers discovered about topic_2?",
            ],
            "correct_answers": [
                ["topic_0 is central to understanding the broader subject."],
                ["Applications of topic_1 span engineering, science, and industry."],
                ["Researchers have studied topic_2 extensively over many decades."],
            ],
            "correct_answer_document_ids": [
                ["doc_0"],
                ["doc_1"],
                ["doc_2"],
            ],
        }
    )


def _build_search_space(foundation_models, embedding_models):
    return AI4RAGSearchSpace(
        vector_store_type="chroma",
        params=[
            Parameter(name="foundation_model", param_type="C", values=foundation_models),
            Parameter(name="embedding_model", param_type="C", values=embedding_models),
        ],
    )


def _make_experiment(documents, benchmark_data, foundation_models, embedding_models, mock_client, **kwargs):
    return AI4RAGExperiment(
        documents=documents,
        benchmark_data=benchmark_data,
        search_space=_build_search_space(foundation_models, embedding_models),
        vector_store_type="chroma",
        client=mock_client,
        optimizer_settings=RandomOptSettings(max_evals=3),
        event_handler=LocalEventHandler(),
        **kwargs,
    )


class TestExperimentChromaWithMockedModels:
    """Full experiment runs with mocked models, real Chroma, and real UnitxtEvaluator."""

    def test_mps_is_triggered_and_reduces_model_pool(
        self, documents, benchmark_data, foundation_models, embedding_models, mock_client
    ):
        """
        With 4 FMs (> DEFAULT_N_FOUNDATION_MODELS=3) and 3 EMs (> DEFAULT_N_EMBEDDING_MODELS=2),
        MPS must be triggered automatically. After search(), the search space must contain
        at most DEFAULT_N_FOUNDATION_MODELS FMs and DEFAULT_N_EMBEDDING_MODELS EMs, and
        every selected model must belong to the original input pool.
        """
        experiment = _make_experiment(documents, benchmark_data, foundation_models, embedding_models, mock_client)

        assert len(experiment.search_space[AI4RAGParamNames.FOUNDATION_MODEL].values) == _N_FOUNDATION_MODELS
        assert len(experiment.search_space[AI4RAGParamNames.EMBEDDING_MODEL].values) == _N_EMBEDDING_MODELS

        experiment.search(optimizer=RandomOptimizer)

        fm_selected = list(experiment.search_space[AI4RAGParamNames.FOUNDATION_MODEL].values)
        em_selected = list(experiment.search_space[AI4RAGParamNames.EMBEDDING_MODEL].values)

        assert len(fm_selected) <= ModelsPreSelector.DEFAULT_N_FOUNDATION_MODELS, (
            f"MPS should reduce foundation models to ≤{ModelsPreSelector.DEFAULT_N_FOUNDATION_MODELS}, "
            f"got {len(fm_selected)}"
        )
        assert len(em_selected) <= ModelsPreSelector.DEFAULT_N_EMBEDDING_MODELS, (
            f"MPS should reduce embedding models to ≤{ModelsPreSelector.DEFAULT_N_EMBEDDING_MODELS}, "
            f"got {len(em_selected)}"
        )
        assert all(
            fm in foundation_models for fm in fm_selected
        ), "MPS selected a foundation model that was not in the original pool"
        assert all(
            em in embedding_models for em in em_selected
        ), "MPS selected an embedding model that was not in the original pool"

    def test_skip_mps_preserves_full_model_pool(self, documents, benchmark_data, foundation_models, embedding_models, mock_client):
        """
        When skip_mps=True, MPS is bypassed entirely. The search space must retain all
        originally provided models after search() completes.
        """
        experiment = _make_experiment(documents, benchmark_data, foundation_models, embedding_models, mock_client)

        experiment.search(optimizer=RandomOptimizer, skip_mps=True)

        fm_after = list(experiment.search_space[AI4RAGParamNames.FOUNDATION_MODEL].values)
        em_after = list(experiment.search_space[AI4RAGParamNames.EMBEDDING_MODEL].values)

        assert len(fm_after) == _N_FOUNDATION_MODELS, (
            f"With skip_mps=True, all {_N_FOUNDATION_MODELS} foundation models should remain, " f"got {len(fm_after)}"
        )
        assert len(em_after) == _N_EMBEDDING_MODELS, (
            f"With skip_mps=True, all {_N_EMBEDDING_MODELS} embedding models should remain, " f"got {len(em_after)}"
        )

    def test_evaluation_scores_are_in_valid_range(self, documents, benchmark_data, foundation_models, embedding_models, mock_client):
        """
        Every EvaluationResult produced by the experiment must have a final_score in [0, 1]
        and per-metric mean scores that are either None or in [0, 1].
        MockedFoundationModel returns a fixed answer, so scores will be low but valid.
        """
        experiment = _make_experiment(
            documents,
            benchmark_data,
            foundation_models,
            embedding_models,
            mock_client,
            optimization_metric=MetricType.FAITHFULNESS,
            metrics=(MetricType.FAITHFULNESS, MetricType.ANSWER_CORRECTNESS, MetricType.CONTEXT_CORRECTNESS),
        )

        experiment.search(optimizer=RandomOptimizer)

        assert len(experiment.results) > 0, "Experiment produced no evaluation results"

        for evaluation in experiment.results:
            assert evaluation.final_score is not None
            assert 0.0 <= evaluation.final_score <= 1.0, (
                f"final_score {evaluation.final_score!r} is outside [0, 1] " f"for {evaluation.pattern_name}"
            )

            for metric_name, metric_data in evaluation.scores["scores"].items():
                mean = metric_data.get("mean")
                assert mean is None or 0.0 <= mean <= 1.0, (
                    f"Mean score {mean!r} is outside [0, 1] for metric '{metric_name}' " f"in {evaluation.pattern_name}"
                )

    def test_best_pattern_can_generate_answer(self, documents, benchmark_data, foundation_models, embedding_models, mock_client):
        """
        The best RAG pattern returned by the experiment must produce a non-empty answer,
        confirming the full inference pipeline (retrieval + generation) is intact.
        """
        experiment = _make_experiment(documents, benchmark_data, foundation_models, embedding_models, mock_client)

        experiment.search(optimizer=RandomOptimizer)

        best_evals = experiment.results.get_best_evaluations(k=1)
        assert len(best_evals) == 1

        result = best_evals[0].rag_pattern.generate("What is topic_0 about?")
        assert isinstance(result, dict), f"Expected dict from generate(), got {type(result)}"
        answer = result.get("answer")
        assert isinstance(answer, str) and len(answer) > 0, f"Expected a non-empty answer string, got {answer!r}"

    def test_pattern_params_include_generation_config(self, documents, benchmark_data, foundation_models, embedding_models, mock_client):
        """
        Verify that RAG patterns include complete generation configuration.
        This ensures the API migration properly captures model parameters.
        """
        experiment = _make_experiment(documents, benchmark_data, foundation_models, embedding_models, mock_client)

        experiment.search(
            optimizer=RandomOptimizer,
            skip_mps=True,  # Skip MPS to speed up test
        )

        # Get best pattern
        best_evaluations = experiment.results.get_best_evaluations(k=1)
        assert len(best_evaluations) > 0, f"No evaluations generated. Total results: {len(experiment.results)}"

        # Verify rag_params contains both generation and retrieval configuration
        rag_params = best_evaluations[0].rag_params
        assert "generation" in rag_params, "RAG params should include generation config"
        assert "retrieval" in rag_params, "RAG params should include retrieval config"

        # Verify generation configuration fields
        generation = rag_params["generation"]
        assert "model_id" in generation, "Generation config should include model_id"
        assert "context_template_text" in generation, "Generation config should include context_template_text"
        assert "user_message_text" in generation, "Generation config should include user_message_text"
        assert "system_message_text" in generation, "Generation config should include system_message_text"

        # Verify field values are valid (not just present)
        assert isinstance(generation["model_id"], str) and generation["model_id"], (
            "model_id should be a non-empty string"
        )
        assert isinstance(generation["context_template_text"], str), (
            "context_template_text should be a string"
        )
        assert isinstance(generation["user_message_text"], str), (
            "user_message_text should be a string"
        )
        assert isinstance(generation["system_message_text"], str), (
            "system_message_text should be a string"
        )

        # Verify the model_id matches one of our mocked models
        assert generation["model_id"] in [fm.model_id for fm in foundation_models], (
            f"Model ID {generation['model_id']} not in foundation models list"
        )
