# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import math

import pytest

from ai4rag.evaluator.custom_metrics import calculate_overall_score
from ai4rag.evaluator.metric import Metrics


@pytest.fixture
def two_metric_result():
    return {
        "metrics": [
            {
                "name": "answer_correctness",
                "evaluator": "unitxt",
                "description": "",
                "scores": {"mean": 0.8, "ci_low": 0.7, "ci_high": 0.9},
            },
            {
                "name": "faithfulness",
                "evaluator": "unitxt",
                "description": "",
                "scores": {"mean": 0.6, "ci_low": 0.5, "ci_high": 0.7},
            },
        ],
        "question_scores": [
            {
                "question_id": "q1",
                "metrics": [
                    {"name": "answer_correctness", "evaluator": "unitxt", "value": 0.9},
                    {"name": "faithfulness", "evaluator": "unitxt", "value": 0.7},
                ],
            },
            {
                "question_id": "q2",
                "metrics": [
                    {"name": "answer_correctness", "evaluator": "unitxt", "value": 0.7},
                    {"name": "faithfulness", "evaluator": "unitxt", "value": 0.5},
                ],
            },
        ],
    }


class TestCalculateOverallScore:

    def test_aggregate_mean(self, two_metric_result):
        calculate_overall_score(two_metric_result)

        overall = next(m for m in two_metric_result["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert overall["scores"]["mean"] == 0.7

    def test_aggregate_ci(self, two_metric_result):
        calculate_overall_score(two_metric_result)

        overall = next(m for m in two_metric_result["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert overall["scores"]["ci_low"] == 0.6
        assert overall["scores"]["ci_high"] == 0.8

    def test_aggregate_evaluator_and_description(self, two_metric_result):
        calculate_overall_score(two_metric_result)

        overall = next(m for m in two_metric_result["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert overall["evaluator"] == "custom"

    def test_per_question_values(self, two_metric_result):
        calculate_overall_score(two_metric_result)

        q1 = next(q for q in two_metric_result["question_scores"] if q["question_id"] == "q1")
        q2 = next(q for q in two_metric_result["question_scores"] if q["question_id"] == "q2")

        q1_overall = next(m for m in q1["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        q2_overall = next(m for m in q2["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)

        assert q1_overall["value"] == 0.8
        assert q2_overall["value"] == 0.6

    def test_per_question_evaluator(self, two_metric_result):
        calculate_overall_score(two_metric_result)

        q1 = two_metric_result["question_scores"][0]
        q1_overall = next(m for m in q1["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert q1_overall["evaluator"] == "custom"

    def test_mutates_in_place(self, two_metric_result):
        original_metrics = two_metric_result["metrics"]
        calculate_overall_score(two_metric_result)
        assert two_metric_result["metrics"] is original_metrics
        assert any(m["name"] == Metrics.OVERALL_SCORE.name for m in original_metrics)

    def test_with_none_ci(self):
        result = {
            "metrics": [
                {
                    "name": "answer_correctness",
                    "evaluator": "unitxt",
                    "description": "",
                    "scores": {"mean": 0.9, "ci_low": None, "ci_high": None},
                },
            ],
            "question_scores": [],
        }
        calculate_overall_score(result)

        overall = next(m for m in result["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert overall["scores"]["mean"] == 0.9
        assert overall["scores"]["ci_low"] is None
        assert overall["scores"]["ci_high"] is None

    def test_with_nan_question_value(self):
        result = {
            "metrics": [
                {
                    "name": "answer_correctness",
                    "evaluator": "unitxt",
                    "description": "",
                    "scores": {"mean": 0.8, "ci_low": 0.7, "ci_high": 0.9},
                },
            ],
            "question_scores": [
                {
                    "question_id": "q1",
                    "metrics": [
                        {"name": "answer_correctness", "evaluator": "unitxt", "value": float("nan")},
                        {"name": "faithfulness", "evaluator": "unitxt", "value": 0.6},
                    ],
                },
            ],
        }
        calculate_overall_score(result)

        q1_overall = next(m for m in result["question_scores"][0]["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert q1_overall["value"] == 0.6

    def test_all_nan_question_values(self):
        result = {
            "metrics": [],
            "question_scores": [
                {
                    "question_id": "q1",
                    "metrics": [
                        {"name": "answer_correctness", "evaluator": "unitxt", "value": float("nan")},
                    ],
                },
            ],
        }
        calculate_overall_score(result)

        q1_overall = next(m for m in result["question_scores"][0]["metrics"] if m["name"] == Metrics.OVERALL_SCORE.name)
        assert math.isnan(q1_overall["value"])

    def test_empty_metrics(self):
        result = {"metrics": [], "question_scores": []}
        calculate_overall_score(result)

        assert len(result["metrics"]) == 0
        assert len(result["question_scores"]) == 0
