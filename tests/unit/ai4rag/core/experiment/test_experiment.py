# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ai4rag.core.experiment.utils import merge_evaluation_results
from ai4rag.evaluator.base_evaluator import (
    AggregateMetric,
    BaseEvaluator,
    ConfidenceInterval,
    EvaluationMetricsResult,
    QuestionMetric,
    QuestionScore,
)
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.evaluator.metric import Metrics
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BENCHMARK_DF = pd.DataFrame(
    {
        "question": ["What is Python?"],
        "correct_answers": [["A programming language."]],
        "correct_answer_document_ids": [["doc1"]],
    }
)


def _make_result(
    metric_name: str,
    evaluator: str,
    mean: float,
    question_scores: dict[str, float],
) -> EvaluationMetricsResult:
    """Build a minimal EvaluationMetricsResult for testing."""
    return EvaluationMetricsResult(
        metrics=[
            AggregateMetric(
                name=metric_name,
                evaluator=evaluator,
                description="",
                scores=ConfidenceInterval(mean=mean, ci_low=None, ci_high=None),
            )
        ],
        question_scores=[
            QuestionScore(
                question_id=qid,
                metrics=[QuestionMetric(name=metric_name, evaluator=evaluator, value=val)],
            )
            for qid, val in question_scores.items()
        ],
    )


def _build_experiment(evaluators=None, optimization_metric=Metrics.FAITHFULNESS, metrics=None):
    """Construct an AI4RAGExperiment with all heavy deps mocked out."""
    from ai4rag.core.experiment.experiment import AI4RAGExperiment

    kwargs = {}
    if evaluators is not None:
        kwargs["evaluators"] = evaluators
    if metrics is not None:
        kwargs["metrics"] = metrics

    return AI4RAGExperiment(
        documents=[],
        benchmark_data=_BENCHMARK_DF,
        search_space=MagicMock(),
        vector_store_type="chroma",
        optimizer_settings=MagicMock(),
        event_handler=MagicMock(),
        client=MagicMock(),
        optimization_metric=optimization_metric,
        **kwargs,
    )


def _make_llmaj_evaluator():
    model = MagicMock()
    model.model_id = "judge-model"
    return LLMaJEvaluator(model=model)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEvaluatorType:
    def test_base_evaluator_has_empty_default(self):
        assert BaseEvaluator.EVALUATOR_TYPE == ""

    def test_unitxt_evaluator_type(self):
        assert UnitxtEvaluator.EVALUATOR_TYPE == "unitxt"

    def test_llmaj_evaluator_type(self):
        assert LLMaJEvaluator.EVALUATOR_TYPE == "judge"


class TestEvaluatorsSetter:
    def test_default_evaluators_is_unitxt_only(self):
        exp = _build_experiment()
        assert len(exp.evaluators) == 1
        assert isinstance(exp.evaluators[0], UnitxtEvaluator)

    def test_explicit_evaluators_are_stored(self):
        evals = [UnitxtEvaluator(), _make_llmaj_evaluator()]
        exp = _build_experiment(evaluators=evals)
        assert len(exp.evaluators) == 2
        assert isinstance(exp.evaluators[0], UnitxtEvaluator)
        assert isinstance(exp.evaluators[1], LLMaJEvaluator)

    def test_setter_rejects_non_evaluator_instances(self):
        with pytest.raises(ValueError, match="BaseEvaluator"):
            _build_experiment(evaluators=[UnitxtEvaluator(), "not_an_evaluator"])


class TestDefaultMetrics:
    def test_unitxt_only_defaults(self):
        exp = _build_experiment()
        names = [m.name for m in exp.metrics]
        assert "answer_correctness" in names
        assert "faithfulness" in names
        assert "context_correctness" in names
        assert "overall_score" in names
        assert "answer_relevance" not in names

    def test_with_judge_evaluator_includes_answer_relevance(self):
        evals = [UnitxtEvaluator(), _make_llmaj_evaluator()]
        exp = _build_experiment(evaluators=evals)
        names = [m.name for m in exp.metrics]
        assert "answer_relevance" in names
        assert "overall_score" in names

    def test_explicit_metrics_override_defaults(self):
        exp = _build_experiment(metrics=(Metrics.FAITHFULNESS,))
        assert len(exp.metrics) == 1
        assert exp.metrics[0].name == "faithfulness"


class TestMetricEvaluatorValidation:
    def test_unitxt_metric_with_unitxt_evaluator_passes(self):
        _build_experiment(optimization_metric=Metrics.FAITHFULNESS)

    def test_custom_metric_always_passes(self):
        _build_experiment(optimization_metric=Metrics.OVERALL_SCORE)

    def test_judge_metric_without_judge_evaluator_raises(self):
        with pytest.raises(ValueError, match="requires a 'judge' evaluator"):
            _build_experiment(optimization_metric=Metrics.JUDGE_ANSWER_RELEVANCE)

    def test_judge_metric_with_judge_evaluator_passes(self):
        evals = [UnitxtEvaluator(), _make_llmaj_evaluator()]
        _build_experiment(evaluators=evals, optimization_metric=Metrics.JUDGE_ANSWER_RELEVANCE)


class TestMergeEvaluationResults:
    def test_empty_list_returns_empty_result(self):
        merged = merge_evaluation_results([])
        assert merged["metrics"] == []
        assert merged["question_scores"] == []

    def test_single_result_returned_as_is(self):
        r = _make_result("faithfulness", "unitxt", 0.8, {"q1": 0.9, "q2": 0.7})
        merged = merge_evaluation_results([r])
        assert merged is r

    def test_two_results_merged(self):
        r1 = _make_result("faithfulness", "unitxt", 0.8, {"q1": 0.9, "q2": 0.7})
        r2 = _make_result("answer_relevance", "judge", 0.6, {"q1": 0.5, "q2": 0.7})
        merged = merge_evaluation_results([r1, r2])

        assert len(merged["metrics"]) == 2
        metric_names = {m["name"] for m in merged["metrics"]}
        assert metric_names == {"faithfulness", "answer_relevance"}

        q_scores = {qs["question_id"]: qs for qs in merged["question_scores"]}
        assert len(q_scores["q1"]["metrics"]) == 2
        assert len(q_scores["q2"]["metrics"]) == 2

    def test_question_ids_preserved(self):
        r1 = _make_result("f", "unitxt", 0.5, {"q1": 0.5, "q2": 0.5, "q3": 0.5})
        r2 = _make_result("a", "judge", 0.5, {"q1": 0.5, "q2": 0.5, "q3": 0.5})
        merged = merge_evaluation_results([r1, r2])
        qids = [qs["question_id"] for qs in merged["question_scores"]]
        assert qids == ["q1", "q2", "q3"]
