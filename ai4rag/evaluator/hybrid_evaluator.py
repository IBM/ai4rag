# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Sequence

from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData, MetricType
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.evaluator.score_utils import enrich_with_overall_score
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator

UNITXT_METRICS = (
    MetricType.FAITHFULNESS,
    MetricType.ANSWER_CORRECTNESS,
    MetricType.CONTEXT_CORRECTNESS,
)
LLMAJ_METRICS = (MetricType.ANSWER_RELEVANCE,)
ALL_PATTERN_METRICS = UNITXT_METRICS + LLMAJ_METRICS


class HybridEvaluator(BaseEvaluator):
    """Routes standard RAG metrics to Unitxt and ``answer_relevance`` to LLM-as-a-Judge."""

    def __init__(self, unitxt_evaluator: UnitxtEvaluator, llmaj_evaluator: LLMaJEvaluator):
        self._unitxt = unitxt_evaluator
        self._llmaj = llmaj_evaluator

    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[str],
    ) -> dict:
        """Evaluate with Unitxt and LLMaJ backends, then derive ``overall_score``."""
        unitxt_result = self._unitxt.evaluate_metrics_raw(evaluation_data, UNITXT_METRICS)
        llmaj_result = self._llmaj.evaluate_metrics(evaluation_data, LLMAJ_METRICS)

        scores = {**unitxt_result["scores"], **llmaj_result["scores"]}
        question_scores = {**unitxt_result["question_scores"], **llmaj_result["question_scores"]}
        merged = enrich_with_overall_score({"scores": scores, "question_scores": question_scores})

        if set(metrics) == set(ALL_PATTERN_METRICS) or MetricType.OVERALL_SCORE in metrics:
            return merged

        filtered_scores = {name: merged["scores"][name] for name in metrics if name in merged["scores"]}
        filtered_question_scores = {
            name: merged["question_scores"][name] for name in metrics if name in merged["question_scores"]
        }
        return {"scores": filtered_scores, "question_scores": filtered_question_scores}

    def get_supported_metrics(self) -> list[str]:
        """Return all metrics produced by the hybrid evaluator."""
        return list(ALL_PATTERN_METRICS) + [MetricType.OVERALL_SCORE]
