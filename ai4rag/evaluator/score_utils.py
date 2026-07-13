# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

import numpy as np

from ai4rag.evaluator.base_evaluator import MetricType

STANDARD_METRICS = (
    MetricType.FAITHFULNESS,
    MetricType.ANSWER_CORRECTNESS,
    MetricType.CONTEXT_CORRECTNESS,
    MetricType.ANSWER_RELEVANCE,
)


def compute_confidence_interval(
    scores: list[float], confidence: float = 0.95, n_bootstrap: int = 1000
) -> tuple[float | None, float | None]:
    """Compute bootstrap confidence interval for the mean score."""
    if len(scores) < 2:
        return None, None

    rng = np.random.default_rng(seed=42)
    bootstrap_means = [float(np.mean(rng.choice(scores, size=len(scores), replace=True))) for _ in range(n_bootstrap)]

    alpha = (1 - confidence) / 2
    return (
        round(float(np.percentile(bootstrap_means, alpha * 100)), 4),
        round(float(np.percentile(bootstrap_means, (1 - alpha) * 100)), 4),
    )


def enrich_with_overall_score(result: dict[str, Any]) -> dict[str, Any]:
    """Add derived ``overall_score`` to pattern-level and per-question scores."""
    scores = result.get("scores") or {}
    question_scores = result.get("question_scores") or {}

    question_ids: set[str] = set()
    for metric in STANDARD_METRICS:
        question_ids.update((question_scores.get(metric) or {}).keys())

    per_question_overall: list[float] = []
    overall_by_question: dict[str, float | None] = {}
    for qid in question_ids:
        values = [
            question_scores[metric][qid]
            for metric in STANDARD_METRICS
            if metric in question_scores and question_scores[metric].get(qid) is not None
        ]
        if values:
            mean_val = round(float(np.mean(values)), 4)
            overall_by_question[qid] = mean_val
            per_question_overall.append(mean_val)
        else:
            overall_by_question[qid] = None

    question_scores["overall_score"] = overall_by_question

    ci_low, ci_high = compute_confidence_interval(per_question_overall)
    scores["overall_score"] = {
        "mean": round(float(np.mean(per_question_overall)), 4) if per_question_overall else None,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

    return {"scores": scores, "question_scores": question_scores}
