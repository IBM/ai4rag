# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from unitxt.eval_utils import evaluate

from ai4rag.core.experiment.exception_handler import EvaluationError
from ai4rag.evaluator.base_evaluator import (
    AggregateMetric,
    BaseEvaluator,
    EvaluationData,
    EvaluationMetricsResult,
    QuestionMetric,
    QuestionScore,
    build_aggregate_metric,
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

    # Reference-based unitxt metrics compute a token overlap against a list of
    # references; unitxt crashes (``max()`` on an empty sequence) when that list
    # is empty for a record. We therefore evaluate each such metric only over the
    # records that actually carry its references — the field named below.
    _REFERENCE_FIELD: dict[str, str] = {
        "metrics.rag.external_rag.faithfulness": "contexts",
        "metrics.rag.external_rag.answer_correctness": "ground_truths",
    }

    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[RAGMetric],
    ) -> EvaluationMetricsResult:
        """
        Perform evaluation on the given instances with chosen metric types.

        Records that lack the references a metric needs are excluded from that
        metric: they contribute no per-question score and are left out of the
        mean/confidence interval, rather than crashing the whole evaluation.

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
        df = pd.DataFrame([prim.to_dict() for prim in evaluation_data])

        metric_lookup = {self.METRIC_TYPE_MAP[m.name]: m for m in metrics if m.name in self.METRIC_TYPE_MAP}
        if not metric_lookup or df.empty:
            return {"metrics": [], "question_scores": []}

        try:
            scores_df, ci_table = self._evaluate_by_reference_group(df, metric_lookup)

            aggregate_metrics = self._build_aggregate_metrics(ci_table=ci_table, metric_lookup=metric_lookup)
            question_scores = self._build_question_scores(scores_df=scores_df, metric_lookup=metric_lookup)

            return {"metrics": aggregate_metrics, "question_scores": question_scores}

        except Exception as exc:
            raise EvaluationError(exc) from exc

    def _evaluate_by_reference_group(
        self, df: pd.DataFrame, metric_lookup: dict[str, RAGMetric]
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Run unitxt per group of metrics that share the same evaluable rows.

        Metrics sharing a mask (the common case: every record is evaluable) run in
        a single ``evaluate`` call, preserving the original behaviour and cost.
        Returns a combined per-question ``scores_df`` (``NaN`` where a record was
        excluded) and a combined ``ci_table``, both ordered as ``metric_lookup``.
        """
        combined_scores = pd.DataFrame({"question_id": df["question_id"].tolist()})
        combined_ci: dict[str, dict] = {}

        masks = {name: self._evaluable_mask(df, name) for name in metric_lookup}
        groups: dict[tuple, list[str]] = defaultdict(list)
        for name, mask in masks.items():
            groups[tuple(mask.tolist())].append(name)

        for names in groups.values():
            self._evaluate_group(df, masks[names[0]], names, combined_scores, combined_ci)

        ordered_ci = pd.DataFrame({name: combined_ci[name] for name in metric_lookup})
        ordered_scores = combined_scores[["question_id", *metric_lookup]]
        return ordered_scores, ordered_ci

    @classmethod
    def _evaluable_mask(cls, df: pd.DataFrame, unitxt_name: str) -> pd.Series:
        """Return a boolean mask of rows that carry the references ``unitxt_name`` needs.

        Metrics without a reference requirement (or whose reference column is
        absent) evaluate over every row.
        """
        field = cls._REFERENCE_FIELD.get(unitxt_name)
        if field is None or field not in df.columns:
            return pd.Series(True, index=df.index)
        return df[field].apply(lambda v: isinstance(v, (list, tuple)) and len(v) > 0)

    @staticmethod
    def _empty_ci() -> dict[str, float]:
        """Confidence-interval entry for a metric with no evaluable records."""
        return {"score": np.nan, "score_ci_low": np.nan, "score_ci_high": np.nan}

    def _evaluate_group(
        self,
        df: pd.DataFrame,
        mask: pd.Series,
        names: list[str],
        combined_scores: pd.DataFrame,
        combined_ci: dict[str, dict],
    ) -> None:
        """Evaluate one group of metrics over ``mask``-selected rows, in place.

        Populates ``combined_scores`` (a per-question column per metric, ``NaN``
        for excluded rows) and ``combined_ci`` (the metric's CI entry).
        """
        subset = df[mask]
        if subset.empty:
            for name in names:
                combined_scores[name] = np.nan
                combined_ci[name] = self._empty_ci()
            return

        scores_df, ci_table = evaluate(subset, metric_names=list(names), compute_conf_intervals=True)
        ci_dict = ci_table.to_dict()
        for name in names:
            if name in scores_df.columns:
                per_question = scores_df.set_index("question_id")[name]
                combined_scores[name] = combined_scores["question_id"].map(per_question)
            else:
                combined_scores[name] = np.nan
            combined_ci[name] = ci_dict.get(name, self._empty_ci())

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
                    # Excluded/unscored records carry NaN for the metric; omit them
                    # so an unevaluable record contributes no per-question score.
                    if not pd.isna(record[col])
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
            build_aggregate_metric(
                metric=metric_lookup[key],
                values=[],
                confidence_interval=(round_or_none(val.get("score_ci_low")), round_or_none(val.get("score_ci_high"))),
                mean=round_or_none(val["score"]),
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
