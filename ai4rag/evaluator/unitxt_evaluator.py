# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from unitxt.eval_utils import evaluate

from ai4rag.evaluator.base_evaluator import (
    BaseEvaluator,
    MetricType,
    EvaluationData,
)
from ai4rag.core.experiment.exception_handler import EvaluationError


class UnitxtEvaluator(BaseEvaluator):
    """Unitxt wrapper making evaluation of the RAG's usage."""

    METRIC_TYPE_MAP = {
        MetricType.ANSWER_CORRECTNESS: "metrics.rag.external_rag.answer_correctness",
        MetricType.FAITHFULNESS: "metrics.rag.external_rag.faithfulness",
        MetricType.CONTEXT_CORRECTNESS: "metrics.rag.external_rag.context_correctness",
    }

    def evaluate_metrics(
        self,
        evaluation_data: Iterable[EvaluationData],
        metrics: Sequence[str],
    ) -> dict:
        """
        Perform evaluation on the given instances with chosen metric types.

        Parameters
        ----------
        evaluation_data : Iterable[EvaluationData]
            Iterable of instances that hold data needed for the unitxt
            algorithms to perform evaluation.

        metrics : Sequence[str]
            Values describing which specific evaluation metrics should be used
            withing evaluation process.

        Returns
        -------
        dict
            Dictionary of scores given for each EvaluationData.
        """
        evaluation_primitives = [prim.to_dict() for prim in evaluation_data]
        df = pd.DataFrame(evaluation_primitives)
        unitxt_metrics = self.get_metric_types(metric_types=metrics)
        try:
            scores_df, ci_table = evaluate(df, metric_names=unitxt_metrics, compute_conf_intervals=True)

            returned_ci = self._handle_ci_calculations(ci_table=ci_table)
            question_scores = self._handle_questions_scores(scores_df=scores_df)

            return {"scores": returned_ci, "question_scores": question_scores}

        except Exception as exc:
            raise EvaluationError(exc) from exc

    def _handle_questions_scores(self, scores_df: pd.DataFrame) -> dict:
        """
        Handle transformations of questions scores data frame.

        Parameters
        ----------
        scores_df : pd.DataFrame
            Data returned by the unitxt evaluate function containing scores without ids.

        Returns
        -------
        dict
            Scores calculated for each question in the evaluation data.
        """
        reversed_metrics_mapping = {v: k for k, v in self.METRIC_TYPE_MAP.items()}
        scores_df.replace("", np.nan, inplace=True)
        raw_ret_dict = scores_df.round(4).to_dict()
        without_id = {
            reversed_metrics_mapping[key]: val for key, val in raw_ret_dict.items() if key in reversed_metrics_mapping
        }
        question_scores = {}
        for key, val in without_id.items():
            question_scores[key] = {raw_ret_dict["question_id"][k]: v for k, v in val.items()}

        return question_scores

    def _handle_ci_calculations(self, ci_table: pd.DataFrame) -> dict:
        """
        Handle transformations of confidence interval scores data frame.

        Parameters
        ----------
        ci_table : pd.DataFrame
            Data with calculated confidence intervals via unitxt evaluate.

        Returns
        -------
        dict
            Transformed confidence interval data that will be further processed.
        """
        reversed_metrics_mapping = {v: k for k, v in self.METRIC_TYPE_MAP.items()}
        ci_table.replace(np.nan, None, inplace=True)
        ci_dict = ci_table.to_dict()

        def round_or_none(x: float | None) -> float | None:
            return None if x is None else round(x, 4)

        returned_ci = {}
        for key, val in ci_dict.items():
            returned_ci[reversed_metrics_mapping[key]] = {
                "mean": round_or_none(val["score"]),
                "ci_low": round_or_none(val.get("score_ci_low")),
                "ci_high": round_or_none(val.get("score_ci_high")),
            }

        return returned_ci

    @classmethod
    def get_metric_types(cls, metric_types: Sequence[str]) -> list[str]:
        """
        Perform mapping of general metric names to the specific metric names
        in the unitxt library.

        Parameters
        ----------
        metric_types : Sequence[str]
            Metrics defined in the MetricType class.

        Returns
        -------
        list
            Specific versions of the metrics that can be used within
            unitxt evaluation process.
        """
        mapping = [cls.METRIC_TYPE_MAP.get(metric, None) for metric in metric_types]
        return [metric for metric in mapping if metric is not None]

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
            Corresponding decoded messages
        """

        reversed_mapping = {v: k for k, v in cls.METRIC_TYPE_MAP.items()}
        decoded = [reversed_mapping[metric] for metric in unitxt_metrics]

        return decoded
