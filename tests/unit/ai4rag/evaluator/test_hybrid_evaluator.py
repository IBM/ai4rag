# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

from ai4rag.evaluator.base_evaluator import MetricType
from ai4rag.evaluator.hybrid_evaluator import ALL_PATTERN_METRICS, HybridEvaluator


def test_hybrid_evaluator_merges_backends_and_derives_overall_score():
    unitxt = MagicMock()
    llmaj = MagicMock()
    unitxt.evaluate_metrics_raw.return_value = {
        "scores": {
            MetricType.FAITHFULNESS: {"mean": 0.8, "ci_low": None, "ci_high": None},
            MetricType.ANSWER_CORRECTNESS: {"mean": 0.6, "ci_low": None, "ci_high": None},
            MetricType.CONTEXT_CORRECTNESS: {"mean": 0.4, "ci_low": None, "ci_high": None},
        },
        "question_scores": {
            MetricType.FAITHFULNESS: {"q1": 0.8},
            MetricType.ANSWER_CORRECTNESS: {"q1": 0.6},
            MetricType.CONTEXT_CORRECTNESS: {"q1": 0.4},
        },
    }
    llmaj.evaluate_metrics.return_value = {
        "scores": {MetricType.ANSWER_RELEVANCE: {"mean": 1.0, "ci_low": None, "ci_high": None}},
        "question_scores": {MetricType.ANSWER_RELEVANCE: {"q1": 1.0}},
    }

    evaluator = HybridEvaluator(unitxt, llmaj)
    result = evaluator.evaluate_metrics([], list(ALL_PATTERN_METRICS))

    assert result["scores"][MetricType.OVERALL_SCORE]["mean"] == 0.7
    assert result["question_scores"][MetricType.OVERALL_SCORE]["q1"] == 0.7
    unitxt.evaluate_metrics_raw.assert_called_once()
    llmaj.evaluate_metrics.assert_called_once()


def test_hybrid_evaluator_supported_metrics():
    evaluator = HybridEvaluator(MagicMock(), MagicMock())
    assert MetricType.OVERALL_SCORE in evaluator.get_supported_metrics()
