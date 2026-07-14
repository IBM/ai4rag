# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from typing import Sequence

import numpy as np

from ai4rag import logger
from ai4rag.evaluator.base_evaluator import (
    AggregateMetric,
    BaseEvaluator,
    ConfidenceInterval,
    EvaluationData,
    EvaluationMetricsResult,
    QuestionMetric,
    QuestionScore,
)
from ai4rag.evaluator.metric import Metrics, RAGMetric
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel


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


JUDGE_PROMPT_TEMPLATE = """\
You are an impartial judge evaluating the quality of an AI assistant's response.

## Task
{guidelines}

## Context
Question: {question}

## Response to evaluate
{answer}

## Scoring rubric
- 1 = completely fails the criterion
- 2 = mostly fails with some relevant elements
- 3 = partially meets the criterion
- 4 = mostly meets with minor gaps
- 5 = fully meets the criterion
"""

JUDGE_RESPONSE_FORMAT: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_response",
        "schema": {
            "type": "object",
            "properties": {
                "score": {
                    "type": "integer",
                    "description": "Score from 1 (completely fails) to 5 (fully meets the criterion)",
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation for the assigned score",
                },
            },
            "required": ["score", "rationale"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


class LLMaJEvaluator(BaseEvaluator):
    """Evaluator that scores ``answer_relevance`` with an LLM judge.

    Uses structured output (``response_format``) via the model's
    ``chat()`` method to guarantee well-formed JSON responses.
    All scores are normalized from the judge scale (1-5) to [0.0, 1.0].

    Parameters
    ----------
    model : BaseFoundationModel
        Foundation model instance used as the judge.
    """

    EVALUATOR_TYPE = "judge"

    METRIC_GUIDELINES = {
        Metrics.JUDGE_ANSWER_RELEVANCE.name: (
            "Evaluate whether the response directly and helpfully addresses the user's question. "
            "Consider relevance, helpfulness, accuracy, and whether the answer stays on topic."
        ),
    }

    def __init__(self, model: BaseFoundationModel) -> None:
        self.model = model

    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[RAGMetric],
    ) -> EvaluationMetricsResult:
        """Evaluate responses with the configured judge model."""
        aggregate_metrics: list[AggregateMetric] = []
        per_metric_scores: dict[str, dict[str, float | None]] = {}
        question_ids = [ed.question_id or str(i) for i, ed in enumerate(evaluation_data)]

        for metric in metrics:
            guidelines = self.METRIC_GUIDELINES.get(metric.name)
            if guidelines is None:
                continue

            row_scores: list[float] = []
            per_question: dict[str, float | None] = {}
            for qid, ed in zip(question_ids, evaluation_data):
                normalized = self._judge_row(ed, guidelines)
                per_question[qid] = round(normalized, 4) if normalized is not None else None
                if normalized is not None:
                    row_scores.append(normalized)

            per_metric_scores[metric.name] = per_question

            ci = compute_confidence_interval(row_scores)
            aggregate_metrics.append(
                AggregateMetric(
                    name=metric.name,
                    evaluator=metric.evaluator,
                    description=metric.description,
                    scores=ConfidenceInterval(
                        mean=round(float(np.mean(row_scores)), 4) if row_scores else None,
                        ci_low=ci[0],
                        ci_high=ci[1],
                    ),
                    model_id=self.model.model_id,
                )
            )

        question_scores: list[QuestionScore] = [
            QuestionScore(
                question_id=qid,
                metrics=[
                    QuestionMetric(
                        name=metric.name, evaluator=metric.evaluator, value=per_metric_scores[metric.name][qid]
                    )
                    for metric in metrics
                    if metric.name in per_metric_scores and per_metric_scores[metric.name].get(qid) is not None
                ],
            )
            for qid in question_ids
        ]

        return EvaluationMetricsResult(metrics=aggregate_metrics, question_scores=question_scores)

    def _judge_row(self, evaluation_data: EvaluationData, guidelines: str) -> float | None:
        """Score a single row with the judge model using structured output."""
        question_id = evaluation_data.question_id or "unknown"
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            guidelines=guidelines,
            question=evaluation_data.question or "",
            answer=evaluation_data.answer or "",
        )
        try:
            choices = self.model.chat(
                [{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=256,
                response_format=JUDGE_RESPONSE_FORMAT,
            )
            content = choices[0].message.content.strip()
            data = json.loads(content)
            raw_score = int(data["score"])
            normalized = _normalize_score(raw_score) if 1 <= raw_score <= 5 else None

            return normalized

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning(
                "LLM judge call failed [model=%s question_id=%s]: %s",
                self.model.model_id,
                question_id,
                exc,
            )
            return None

    def get_supported_metrics(self) -> list[str]:
        """Return metric names supported by this evaluator."""
        return list(self.METRIC_GUIDELINES.keys())


def _normalize_score(score: int) -> float:
    """Normalize a score from 1-5 to [0.0, 1.0]."""
    return (score - 1) / 4
