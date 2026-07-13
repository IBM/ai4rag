# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import math
from collections.abc import Sequence
from statistics import fmean

from ai4rag.evaluator.base_evaluator import EvaluationMetricsResult
from ai4rag.evaluator.metric import Metrics, RAGMetric

def calculate_overall_score(scores: EvaluationMetricsResult) -> None:
    """Append an overall-score metric (mean of all other metrics) to *scores* in-place.

    Aggregate scores average mean/ci_low/ci_high independently across existing
    metrics.  Per-question scores average that question's metric values.
    """
    overall = Metrics.OVERALL_SCORE

    def avg(values: list[float | None]) -> float | None:
        valid = [v for v in values if v is not None and not math.isnan(v)]
        return round(fmean(valid), 4) if valid else None

    if scores["metrics"]:
        scores["metrics"].append(
            {
                "name": overall.name,
                "evaluator": overall.evaluator,
                "description": overall.description,
                "scores": {
                    "mean": avg([m["scores"]["mean"] for m in scores["metrics"]]),
                    "ci_low": avg([m["scores"]["ci_low"] for m in scores["metrics"]]),
                    "ci_high": avg([m["scores"]["ci_high"] for m in scores["metrics"]]),
                },
            }
        )

    for question in scores["question_scores"]:
        score = avg([m["value"] for m in question["metrics"]])
        question["metrics"].append(
            {
                "name": overall.name,
                "evaluator": overall.evaluator,
                "value": score if score is not None else float("nan"),
            }
        )


def apply_custom_metrics(scores: EvaluationMetricsResult, metrics: Sequence[RAGMetric]) -> None:
    """Evaluate and append all requested custom metrics to *scores* in-place."""
    for metric in metrics:
        if metric.evaluator != "custom":
            continue
        if metric.name == "overall_score":
            calculate_overall_score(scores)
