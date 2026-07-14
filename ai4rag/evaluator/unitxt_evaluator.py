# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Sequence

import pandas as pd
from unitxt.eval_utils import evaluate

from ai4rag.core.experiment.exception_handler import EvaluationError
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


class UnitxtEvaluator(BaseEvaluator):
    """Unitxt wrapper making evaluation of the RAG's usage."""

    EVALUATOR_TYPE = "unitxt"

    METRIC_TYPE_MAP: dict[str, str] = {
        Metrics.ANSWER_CORRECTNESS.name: "metrics.rag.external_rag.answer_correctness",
        Metrics.FAITHFULNESS.name: "metrics.rag.external_rag.faithfulness",
        Metrics.CONTEXT_CORRECTNESS.name: "metrics.rag.external_rag.context_correctness",
    }

    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[RAGMetric],
    ) -> EvaluationMetricsResult:
        """
        Perform evaluation on the given instances with chosen metric types.

        Parameters
        ----------
        evaluation_data : list[EvaluationData]
            Instances that hold data needed for the unitxt algorithms to perform evaluation.

        metrics : Sequence[RAGMetric]
            Metric definitions to evaluate.

        Returns
        -------
        EvaluationMetricsResult
            Aggregate metrics with confidence intervals and per-question scores.
        """
        evaluation_primitives = [prim.to_dict() for prim in evaluation_data]
        df = pd.DataFrame(evaluation_primitives)

        metric_lookup = {self.METRIC_TYPE_MAP[m.name]: m for m in metrics if m.name in self.METRIC_TYPE_MAP}
        unitxt_metrics = list(metric_lookup)

        try:
            scores_df, ci_table = evaluate(df, metric_names=unitxt_metrics, compute_conf_intervals=True)

            aggregate_metrics = self._build_aggregate_metrics(ci_table=ci_table, metric_lookup=metric_lookup)
            question_scores = self._build_question_scores(scores_df=scores_df, metric_lookup=metric_lookup)

            return {"metrics": aggregate_metrics, "question_scores": question_scores}

        except Exception as exc:
            raise EvaluationError(exc) from exc

    @staticmethod
    def _build_question_scores(scores_df: pd.DataFrame, metric_lookup: dict[str, RAGMetric]) -> list[QuestionScore]:
        """
        Pivot per-metric columns into a question-centric list.

        Parameters
        ----------
        scores_df : pd.DataFrame
            Data returned by the unitxt evaluate function.

        metric_lookup : dict[str, RAGMetric]
            Mapping from unitxt metric name to the originating ``RAGMetric``.

        Returns
        -------
        list[QuestionScore]
            One entry per question with ``question_id`` and ``metrics``.
        """
        scores_df = scores_df.mask(scores_df == "")
        metric_columns = [col for col in scores_df.columns if col in metric_lookup]
        records = scores_df.round(4).to_dict(orient="records")

        return [
            QuestionScore(
                question_id=record["question_id"],
                metrics=[
                    QuestionMetric(
                        name=metric_lookup[col].name,
                        evaluator=metric_lookup[col].evaluator,
                        value=record[col],
                    )
                    for col in metric_columns
                ],
            )
            for record in records
        ]

    @staticmethod
    def _build_aggregate_metrics(ci_table: pd.DataFrame, metric_lookup: dict[str, RAGMetric]) -> list[AggregateMetric]:
        """
        Transform the confidence-interval table into a list of metric summaries.

        Parameters
        ----------
        ci_table : pd.DataFrame
            Data with calculated confidence intervals via unitxt evaluate.

        metric_lookup : dict[str, RAGMetric]
            Mapping from unitxt metric name to the originating ``RAGMetric``.

        Returns
        -------
        list[AggregateMetric]
            One entry per metric with ``name``, ``evaluator``, ``description``, and ``scores``.
        """
        ci_dict = ci_table.to_dict()

        def round_or_none(x: float | None) -> float | None:
            return None if x is None or pd.isna(x) else round(x, 4)

        return [
            AggregateMetric(
                name=metric_lookup[key].name,
                evaluator=metric_lookup[key].evaluator,
                description=metric_lookup[key].description,
                scores=ConfidenceInterval(
                    mean=round_or_none(val["score"]),
                    ci_low=round_or_none(val.get("score_ci_low")),
                    ci_high=round_or_none(val.get("score_ci_high")),
                ),
            )
            for key, val in ci_dict.items()
            if key in metric_lookup
        ]

    @classmethod
    def get_metric_types(cls, metrics: Sequence[RAGMetric]) -> list[str]:
        """
        Map ``RAGMetric`` instances to unitxt-specific metric strings.

        Parameters
        ----------
        metrics : Sequence[RAGMetric]
            Metric definitions.

        Returns
        -------
        list[str]
            Unitxt metric identifiers for the given metrics.
        """
        return [cls.METRIC_TYPE_MAP[m.name] for m in metrics if m.name in cls.METRIC_TYPE_MAP]

    @classmethod
    def decode_unitxt_metric(cls, unitxt_metrics: list[str]) -> list[str]:
        """
        Decode metrics from the unitxt names to general names.

        Parameters
        ----------
        unitxt_metrics : list[str]
            Encoded unitxt metrics.

        Returns
        -------
        list[str]
            Corresponding decoded names.
        """
        reversed_mapping = {v: k for k, v in cls.METRIC_TYPE_MAP.items()}
        return [reversed_mapping[metric] for metric in unitxt_metrics]
