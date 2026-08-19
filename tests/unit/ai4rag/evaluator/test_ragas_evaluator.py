# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import importlib.util
from unittest.mock import MagicMock, PropertyMock

import pandas as pd
import pytest

from ai4rag.core.experiment.exception_handler import EvaluationError
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData
from ai4rag.evaluator.metric import Metrics
from ai4rag.evaluator.ragas_evaluator import RagasEvaluator

ragas_installed = importlib.util.find_spec("ragas") is not None
requires_ragas = pytest.mark.skipif(not ragas_installed, reason="ragas not installed")

RAGAS_METRICS = [
    Metrics.RAGAS_FAITHFULNESS,
    Metrics.RAGAS_ANSWER_RELEVANCY,
    Metrics.RAGAS_CONTEXT_PRECISION,
    Metrics.RAGAS_CONTEXT_RECALL,
]


@pytest.fixture
def sample_evaluation_data() -> list[EvaluationData]:
    return [
        EvaluationData(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
            context_ids=["doc1"],
            ground_truths=["Python is a programming language."],
            ground_truths_context_ids=["doc1"],
            question_id="q1",
        ),
        EvaluationData(
            question="What is AI?",
            answer="AI is Artificial Intelligence.",
            contexts=["AI stands for Artificial Intelligence."],
            context_ids=["doc2"],
            ground_truths=["AI is Artificial Intelligence."],
            ground_truths_context_ids=["doc2"],
            question_id="q2",
        ),
    ]


def _make_model(model_id: str = "ragas-model") -> MagicMock:
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value=model_id)
    return model


def _make_evaluator() -> RagasEvaluator:
    return RagasEvaluator(model=_make_model(), embedding_model=MagicMock())


class TestRagasEvaluatorContract:
    def test_is_base_evaluator_subclass(self):
        assert issubclass(RagasEvaluator, BaseEvaluator)

    def test_evaluator_type(self):
        assert RagasEvaluator.EVALUATOR_TYPE == "ragas"

    def test_supported_metrics(self):
        supported = _make_evaluator().get_supported_metrics()
        assert supported == [m.name for m in RAGAS_METRICS]

    def test_empty_data_returns_empty(self):
        result = _make_evaluator().evaluate_metrics([], RAGAS_METRICS)
        assert result == {"metrics": [], "question_scores": []}

    def test_no_supported_metrics_returns_empty(self, sample_evaluation_data):
        result = _make_evaluator().evaluate_metrics(sample_evaluation_data, [Metrics.ANSWER_CORRECTNESS])
        assert result == {"metrics": [], "question_scores": []}


class TestExtractScores:
    def test_maps_columns_and_cleans_nan(self):
        df = pd.DataFrame(
            {
                "faithfulness": [1.0, float("nan")],
                "answer_relevancy": [0.5, 0.75],
            }
        )
        columns = {"faithfulness": "faithfulness", "answer_relevancy": "answer_relevancy"}
        scores = RagasEvaluator._extract_scores(result_df=df, columns=columns, question_ids=["q1", "q2"])

        assert scores["faithfulness"] == {"q1": 1.0, "q2": None}
        assert scores["answer_relevancy"] == {"q1": 0.5, "q2": 0.75}

    def test_missing_column_defaults_to_none(self):
        df = pd.DataFrame({"faithfulness": [1.0]})
        columns = {"context_recall": "context_recall"}
        scores = RagasEvaluator._extract_scores(result_df=df, columns=columns, question_ids=["q1"])
        assert scores["context_recall"] == {"q1": None}


@requires_ragas
class TestRagasEvaluatorEndToEnd:
    def test_evaluate_metrics(self, sample_evaluation_data, monkeypatch):
        evaluator = _make_evaluator()

        fake_scores = pd.DataFrame(
            {
                "faithfulness": [1.0, 0.5],
                "answer_relevancy": [0.8, 0.6],
            }
        )
        run_mock = MagicMock(return_value=fake_scores)
        monkeypatch.setattr(evaluator, "_run_ragas", run_mock)

        metrics = [Metrics.RAGAS_FAITHFULNESS, Metrics.RAGAS_ANSWER_RELEVANCY]
        result = evaluator.evaluate_metrics(sample_evaluation_data, metrics)

        # _run_ragas received a real ragas dataset built from the eval data.
        assert run_mock.call_count == 1
        dataset = run_mock.call_args.kwargs["dataset"]
        assert len(dataset.samples) == 2

        names = {m["name"]: m for m in result["metrics"]}
        assert set(names) == {"faithfulness", "answer_relevancy"}
        assert names["faithfulness"]["evaluator"] == "ragas"
        assert names["faithfulness"]["scores"]["mean"] == 0.75
        assert names["answer_relevancy"]["scores"]["mean"] == 0.7

        q_scores = {qs["question_id"]: qs for qs in result["question_scores"]}
        q1 = {m["name"]: m["value"] for m in q_scores["q1"]["metrics"]}
        assert q1 == {"faithfulness": 1.0, "answer_relevancy": 0.8}

    def test_ragas_failure_wrapped(self, sample_evaluation_data, monkeypatch):
        evaluator = _make_evaluator()
        monkeypatch.setattr(evaluator, "_run_ragas", MagicMock(side_effect=RuntimeError("boom")))

        with pytest.raises(EvaluationError):
            evaluator.evaluate_metrics(sample_evaluation_data, [Metrics.RAGAS_FAITHFULNESS])
