# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""RAGAS-backed evaluator.

Provides LLM-based RAG metrics (faithfulness, answer relevancy, context
precision/recall) as an independent alternative to the in-house LLM judge.
"""

import math
from typing import Any, Sequence

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
from ai4rag.evaluator.llmaj_evaluator import compute_confidence_interval
from ai4rag.evaluator.metric import Metrics, RAGMetric
from ai4rag.evaluator.ragas_adapters import AI4RAGRagasEmbeddings, AI4RAGRagasLLM
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel


class RagasEvaluator(BaseEvaluator):
    """Evaluator that scores RAG metrics using the RAGAS library.

    RAGAS runs its metrics through the provided foundation and embedding
    models (wrapped via :mod:`ai4rag.evaluator.ragas_adapters`).  Scores are
    already in ``[0.0, 1.0]`` and are returned in the shared
    :class:`EvaluationMetricsResult` shape, with bootstrap confidence
    intervals computed the same way as the other evaluators.

    Parameters
    ----------
    model : BaseFoundationModel
        Foundation model used by RAGAS as the evaluating LLM.
    embedding_model : BaseEmbeddingModel
        Embedding model used by RAGAS metrics that require embeddings
        (e.g. ``answer_relevancy``).
    timeout : int, default=1200
        Per-sample RAGAS timeout in seconds.  Kept high because the underlying
        model may retry slowly (e.g. CPU-deployed endpoints).
    max_workers : int, default=4
        Maximum concurrent RAGAS workers.  Kept low so a small evaluating model
        is not overwhelmed and the async adapter's thread pool is not starved.
    max_completion_tokens : int, default=1024
        Upper bound on tokens per RAGAS evaluation call.  Kept generous so the
        structured JSON that RAGAS metrics emit is not truncated (which would
        otherwise surface as unparseable responses / ``NaN`` scores).
    """

    EVALUATOR_TYPE = "ragas"

    # ai4rag metric name -> (ragas metric factory, ragas result column name)
    _METRIC_SPECS: dict[str, tuple[str, str]] = {
        Metrics.RAGAS_FAITHFULNESS.name: ("faithfulness", "faithfulness"),
        Metrics.RAGAS_ANSWER_RELEVANCY.name: ("answer_relevancy", "answer_relevancy"),
        Metrics.RAGAS_CONTEXT_PRECISION.name: ("context_precision", "context_precision"),
        Metrics.RAGAS_CONTEXT_RECALL.name: ("context_recall", "context_recall"),
    }

    def __init__(
        self,
        model: BaseFoundationModel,
        embedding_model: BaseEmbeddingModel,
        *,
        timeout: int = 1200,
        max_workers: int = 4,
        max_completion_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.embedding_model = embedding_model
        self.timeout = timeout
        self.max_workers = max_workers
        self.max_completion_tokens = max_completion_tokens

    def evaluate_metrics(
        self,
        evaluation_data: list[EvaluationData],
        metrics: Sequence[RAGMetric],
    ) -> EvaluationMetricsResult:
        """Evaluate responses with the configured RAGAS metrics."""
        supported = [m for m in metrics if m.name in self._METRIC_SPECS]
        if not supported or not evaluation_data:
            return EvaluationMetricsResult(metrics=[], question_scores=[])

        question_ids = [ed.question_id or str(i) for i, ed in enumerate(evaluation_data)]

        try:
            ragas_metrics, columns = self._build_ragas_metrics(supported)
            dataset = self._build_dataset(evaluation_data)
            result_df = self._run_ragas(dataset=dataset, ragas_metrics=ragas_metrics)
            per_metric_scores = self._extract_scores(result_df=result_df, columns=columns, question_ids=question_ids)
        except Exception as exc:
            raise EvaluationError(exc) from exc

        aggregate_metrics = self._build_aggregate_metrics(supported=supported, per_metric_scores=per_metric_scores)
        question_scores = self._build_question_scores(
            supported=supported, per_metric_scores=per_metric_scores, question_ids=question_ids
        )
        return EvaluationMetricsResult(metrics=aggregate_metrics, question_scores=question_scores)

    def _build_ragas_metrics(self, supported: list[RAGMetric]) -> tuple[list[Any], dict[str, str]]:
        """Resolve ai4rag metrics to RAGAS metric objects and result columns.

        Parameters
        ----------
        supported : list[RAGMetric]
            The subset of requested metrics this evaluator can score.

        Returns
        -------
        tuple[list[Any], dict[str, str]]
            The RAGAS metric singletons to evaluate and a mapping from ai4rag
            metric name to the RAGAS result-DataFrame column.
        """
        # ragas re-exports these singletons dynamically (deprecation shim), so pylint
        # cannot resolve them statically.
        from ragas.metrics import (  # pylint: disable=no-name-in-module
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        registry = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
        }
        ragas_metrics: list[Any] = []
        columns: dict[str, str] = {}
        for metric in supported:
            ragas_key, column = self._METRIC_SPECS[metric.name]
            ragas_metrics.append(registry[ragas_key])
            columns[metric.name] = column
        return ragas_metrics, columns

    @staticmethod
    def _build_dataset(evaluation_data: list[EvaluationData]) -> Any:
        """Convert ai4rag evaluation data into a RAGAS ``EvaluationDataset``.

        Parameters
        ----------
        evaluation_data : list[EvaluationData]
            The per-question data (question, answer, contexts, ground truth).

        Returns
        -------
        Any
            A ``ragas.dataset_schema.EvaluationDataset`` of single-turn samples.
        """
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample

        samples = [
            SingleTurnSample(
                user_input=ed.question or "",
                response=ed.answer or "",
                retrieved_contexts=list(ed.contexts) if ed.contexts else [],
                reference=(ed.ground_truths[0] if ed.ground_truths else None),
            )
            for ed in evaluation_data
        ]
        return EvaluationDataset(samples=samples)

    def _run_ragas(self, dataset: Any, ragas_metrics: list[Any]) -> Any:
        """Run ``ragas.evaluate`` and return the per-sample scores DataFrame.

        A :class:`~ragas.run_config.RunConfig` bounds per-sample latency and
        concurrency.  RAGAS's defaults (16 concurrent workers, 180s timeout)
        overwhelm a small evaluating model (e.g. an 8B served by vLLM) and, via
        the thread-pool used by the async adapter, leave coroutines waiting for a
        thread while their timeout ticks — producing spurious ``TimeoutError``s.
        The timeout is also raised above the ai4rag model's own slow-model retry
        window.  ``raise_exceptions`` is disabled so a single slow/failed sample
        yields ``NaN`` (mapped to ``None`` downstream) instead of aborting the
        whole pattern evaluation.

        Parameters
        ----------
        dataset : Any
            The RAGAS ``EvaluationDataset`` built by :meth:`_build_dataset`.
        ragas_metrics : list[Any]
            The RAGAS metric singletons to evaluate.

        Returns
        -------
        Any
            A pandas DataFrame with one row per sample and one column per metric.
        """
        from ragas import evaluate
        from ragas.run_config import RunConfig

        run_config = RunConfig(timeout=self.timeout, max_workers=self.max_workers)
        result = evaluate(
            dataset=dataset,
            metrics=ragas_metrics,
            llm=AI4RAGRagasLLM(self.model, max_completion_tokens=self.max_completion_tokens),
            embeddings=AI4RAGRagasEmbeddings(self.embedding_model),
            run_config=run_config,
            raise_exceptions=False,
            show_progress=False,
        )
        return result.to_pandas()

    @staticmethod
    def _extract_scores(
        result_df: Any, columns: dict[str, str], question_ids: list[str]
    ) -> dict[str, dict[str, float | None]]:
        """Map RAGAS per-sample scores to ``{metric_name: {question_id: value}}``.

        Parameters
        ----------
        result_df : Any
            The per-sample scores DataFrame returned by :meth:`_run_ragas`.
        columns : dict[str, str]
            Mapping from ai4rag metric name to RAGAS result column.
        question_ids : list[str]
            Question identifiers aligned with the DataFrame rows.

        Returns
        -------
        dict[str, dict[str, float | None]]
            Cleaned scores keyed by metric name then question id; unparseable or
            ``NaN`` values become ``None``.
        """

        def clean(value: Any) -> float | None:
            if value is None:
                return None
            try:
                num = float(value)
            except (TypeError, ValueError):
                return None
            return None if math.isnan(num) else round(num, 4)

        per_metric_scores: dict[str, dict[str, float | None]] = {}
        for metric_name, column in columns.items():
            column_values = list(result_df[column]) if column in result_df else [None] * len(question_ids)
            per_metric_scores[metric_name] = {qid: clean(value) for qid, value in zip(question_ids, column_values)}
        return per_metric_scores

    @staticmethod
    def _build_aggregate_metrics(
        supported: list[RAGMetric], per_metric_scores: dict[str, dict[str, float | None]]
    ) -> list[AggregateMetric]:
        """Build aggregate metrics with bootstrap confidence intervals.

        Parameters
        ----------
        supported : list[RAGMetric]
            The metrics that were evaluated.
        per_metric_scores : dict[str, dict[str, float | None]]
            Per-question scores keyed by metric name then question id.

        Returns
        -------
        list[AggregateMetric]
            One aggregate per metric, with mean and confidence-interval bounds.
        """
        aggregate_metrics: list[AggregateMetric] = []
        for metric in supported:
            values = [v for v in per_metric_scores[metric.name].values() if v is not None]
            ci = compute_confidence_interval(values)
            aggregate_metrics.append(build_aggregate_metric(metric, ci, values))
        return aggregate_metrics

    @staticmethod
    def _build_question_scores(
        supported: list[RAGMetric],
        per_metric_scores: dict[str, dict[str, float | None]],
        question_ids: list[str],
    ) -> list[QuestionScore]:
        """Pivot per-metric scores into a question-centric list.

        Parameters
        ----------
        supported : list[RAGMetric]
            The metrics that were evaluated.
        per_metric_scores : dict[str, dict[str, float | None]]
            Per-question scores keyed by metric name then question id.
        question_ids : list[str]
            Question identifiers, one entry per output row.

        Returns
        -------
        list[QuestionScore]
            One entry per question with its non-null metric values.
        """
        return [
            QuestionScore(
                question_id=qid,
                metrics=[
                    QuestionMetric(
                        name=metric.name,
                        evaluator=metric.evaluator,
                        value=per_metric_scores[metric.name][qid],
                    )
                    for metric in supported
                    if per_metric_scores[metric.name].get(qid) is not None
                ],
            )
            for qid in question_ids
        ]

    def get_supported_metrics(self) -> list[str]:
        """Return metric names supported by this evaluator.

        Returns
        -------
        list[str]
            The ai4rag metric names this evaluator can score.
        """
        return list(self._METRIC_SPECS.keys())
