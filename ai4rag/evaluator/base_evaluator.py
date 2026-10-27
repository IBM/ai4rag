# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, NotRequired, Sequence, TypedDict

import numpy as np

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
    model_id: NotRequired[str]


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


def build_aggregate_metric(
    metric: RAGMetric,
    confidence_interval: tuple[float | None, float | None],
    values: list[float] | None = None,
    model_id: str | None = None,
    mean: float | None = None,
) -> AggregateMetric:
    """Assemble an :class:`AggregateMetric` from per-question scores.

    Shared by the concrete evaluators so the mean / confidence-interval
    envelope is built identically everywhere.

    Parameters
    ----------
    metric : RAGMetric
        The metric definition (name, evaluator, description).
    confidence_interval : tuple[float | None, float | None]
        Pre-computed ``(ci_low, ci_high)`` bounds.
    values : list[float] | None, default=None
        Non-null per-question scores. Used to compute the mean unless a
        pre-computed ``mean`` is supplied. Evaluators that already hold an
        aggregated score (e.g. unitxt) pass ``mean`` directly and omit this.
    model_id : str | None, default=None
        Optional evaluating-model identifier; added only when provided.
    mean : float | None, default=None
        Pre-computed mean. When ``None`` the mean is derived from ``values``
        (evaluators such as unitxt already receive an aggregated score and
        pass it here directly to avoid recomputation).

    Returns
    -------
    AggregateMetric
        The metric with its mean and confidence-interval bounds populated.

    Raises
    ------
    ValueError
        If both ``values`` and ``mean`` are ``None`` (nothing to aggregate
        and no pre-computed mean to fall back on).
    """
    if values is None and mean is None:
        raise ValueError("build_aggregate_metric requires either 'values' or a pre-computed 'mean'.")
    if mean is None:
        mean = round(float(np.mean(values)), 4) if values else None
    aggregate = AggregateMetric(
        name=metric.name,
        evaluator=metric.evaluator,
        description=metric.description,
        scores=ConfidenceInterval(
            mean=mean,
            ci_low=confidence_interval[0],
            ci_high=confidence_interval[1],
        ),
    )
    if model_id is not None:
        aggregate["model_id"] = model_id
    return aggregate


class BaseEvaluator(ABC):
    """
    This class defines the functionality to evaluate a RAG application
    and compare different RAG applications.
    """

    EVALUATOR_TYPE: str = ""

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
