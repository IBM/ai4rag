# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Sequence, TypedDict

from ai4rag.evaluator.metric import RAGMetric


# pylint: disable=too-many-instance-attributes
@dataclass
class EvaluationData:
    """
    Representation of data sent for evaluation.

    Parameters
    ----------
    question : str | None, default=None
        Original question from the benchmark dataset

    answer : str | None, default=None
        Answer returned by the model.

    contexts : list[str] | None = None
        Contexts used by the model to generate response.

    context_ids: list[str] | None, default=None
        IDs of contexts used by the model to generate response.

    ground_truths : list[str] | None = None
        Correct answers from the benchmark dataset.

    ground_truths_context_ids : list[str] | None = None
        IDs of the correct documents used for answers in the benchmark dataset.

    question_id : str | None = None
        ID of the question.

    additional_data: dict[str, Any] | None = None
        Any additional data associated with the evaluation results.

    Methods
    -------
    to_dict() -> dict[str, Any]
        Used for casting instance to the dictionary
    """

    question: str | None = None
    answer: str | None = None
    contexts: list[str] | None = None
    context_ids: list[str] | None = None
    ground_truths: list[str] | None = None
    ground_truths_context_ids: list[str] | None = None
    question_id: str | None = None
    additional_data: list[Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Cast given instance of dataclass to the dict."""
        return asdict(self)


class ConfidenceInterval(TypedDict):
    """Aggregate scores with confidence interval bounds."""

    mean: float | None
    ci_low: float | None
    ci_high: float | None


class AggregateMetric(TypedDict):
    """Single metric with its aggregate scores."""

    name: str
    evaluator: str
    description: str
    scores: ConfidenceInterval


class QuestionMetric(TypedDict):
    """Single metric value for one question."""

    name: str
    evaluator: str
    value: float


class QuestionScore(TypedDict):
    """Per-question breakdown of all evaluated metrics."""

    question_id: str
    metrics: list[QuestionMetric]


class EvaluationMetricsResult(TypedDict):
    """Top-level return type of ``evaluate_metrics``."""

    metrics: list[AggregateMetric]
    question_scores: list[QuestionScore]


class BaseEvaluator(ABC):
    """
    This class defines the functionality to evaluate a RAG application
    and compare different RAG applications.
    """

    @abstractmethod
    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[RAGMetric],
    ) -> EvaluationMetricsResult:
        """
        Evaluate the model's responses against list of different metrics.

        Parameters
        ----------
        evaluation_data : list[EvaluationData]
            List of EvaluationData instances containing all the data needed
            to perform evaluation.

        metrics : Sequence[RAGMetric]
            Metric definitions to evaluate.

        Returns
        -------
        EvaluationMetricsResult
            Aggregate metrics with confidence intervals and per-question scores.
        """
