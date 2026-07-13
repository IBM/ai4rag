# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Sequence

from ai4rag.utils.constants import ConstantMeta


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


class MetricType(metaclass=ConstantMeta):
    """
    Holder for metric names used in evaluation.

    Uses :class:`~ai4rag.utils.constants.ConstantMeta` so values can be
    iterated and membership-tested without defining a standard ``Enum``.
    """

    ANSWER_CORRECTNESS = "answer_correctness"
    FAITHFULNESS = "faithfulness"
    CONTEXT_CORRECTNESS = "context_correctness"
    ANSWER_RELEVANCE = "answer_relevance"
    OVERALL_SCORE = "overall_score"


SUPPORTED_OPTIMIZATION_METRICS = frozenset(
    {
        MetricType.FAITHFULNESS,
        MetricType.ANSWER_CORRECTNESS,
        MetricType.CONTEXT_CORRECTNESS,
        MetricType.ANSWER_RELEVANCE,
        MetricType.OVERALL_SCORE,
    }
)


class BaseEvaluator(ABC):
    """
    This class defines the functionality to evaluate a RAG application
    and compare different RAG applications.
    """

    @abstractmethod
    def evaluate_metrics(self, evaluation_data: list[EvaluationData], metrics: Sequence[str]) -> dict:
        """
        Evaluate the model's responses against list of different metrics.

        Parameters
        ----------
        evaluation_data : list[EvaluationData]
            List of EvaluationData instances containing all the data needed
            to perform evaluation.

        metrics : Sequence[str]
            Metrics given as strings. They should be referred to MetricType.

        Returns
        -------
        dict
            Evaluation result data.
        """
