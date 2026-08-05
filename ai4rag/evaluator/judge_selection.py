# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

import numpy as np
from docling_core.types.doc import DoclingDocument

from ai4rag import logger
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.utils import build_evaluation_data, query_rag
from ai4rag.evaluator.base_evaluator import EvaluationMetricsResult
from ai4rag.evaluator.llmaj_evaluator import LLMaJEvaluator
from ai4rag.evaluator.metric import Metrics
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.template.simple_rag_template import SimpleRAG
from ai4rag.rag.vector_store.chroma import ChromaVectorStore


def calibration_subset_size(total_rows: int) -> int:
    """Return calibration row count: min(20, 10 % of benchmark rows).

    Parameters
    ----------
    total_rows
        Total number of rows in the benchmark dataset.

    Returns
    -------
    int
        Number of rows to sample for calibration. Returns ``0`` when
        *total_rows* is non-positive.
    """
    if total_rows <= 0:
        return 0
    return max(1, min(20, int(total_rows * 0.1)))


def select_judge_model(
    generation_models: list[BaseFoundationModel],
    embedding_models: list[BaseEmbeddingModel],
    benchmark_data: BenchmarkData,
    documents: list[DoclingDocument],
    max_threads: int = 10,
) -> BaseFoundationModel:
    """Select the best judge model for ``answer_relevance`` evaluation.

    When only one generation model is available it is returned directly.
    Otherwise a calibration round scores each candidate on a small
    benchmark subset and picks the one with the highest spread-and-stability
    score.

    Parameters
    ----------
    generation_models
        Candidate generation models; each is evaluated as a potential judge.
    embedding_models
        Embedding models available for building a calibration RAG pipeline.
    benchmark_data
        Benchmark questions and expected answers used for calibration.
    documents
        Parsed documents used as context in the calibration RAG pipeline.
    max_threads
        Maximum concurrent threads during calibration inference.

    Returns
    -------
    BaseFoundationModel
        The selected judge model instance.

    Raises
    ------
    ValueError
        If *generation_models* is empty.
    """
    if not generation_models:
        raise ValueError("At least one generation model is required to select a judge model.")
    if len(generation_models) == 1:
        return generation_models[0]

    subset_size = calibration_subset_size(len(benchmark_data.questions))
    calibration_benchmark = benchmark_data.get_random_sample(n_records=subset_size, random_seed=17)
    eval_data = _run_reference_rag(
        foundation_model=generation_models[0],
        embedding_model=embedding_models[0],
        documents=documents,
        benchmark_data=calibration_benchmark,
        max_threads=max_threads,
    )

    reference_model_id = generation_models[0].model_id
    rankings = _score_judge_candidates(
        candidates=generation_models,
        eval_data=eval_data,
        reference_model_id=reference_model_id,
    )
    selected = rankings[0]["model"]
    logger.info("Selected judge model: %s (calibration score=%s)", selected.model_id, rankings[0]["score"])
    return selected


def _score_judge_candidates(
    candidates: list[BaseFoundationModel],
    eval_data: list,
    reference_model_id: str,
) -> list[dict[str, Any]]:
    """Score each candidate as a judge and return rankings (best first).

    Parameters
    ----------
    candidates
        Foundation model instances to evaluate as judges.
    eval_data
        Evaluation data produced by the reference RAG pipeline.
    reference_model_id
        Model used to generate the reference RAG responses; preferred
        as a tiebreaker when calibration scores are equal.

    Returns
    -------
    list[dict[str, Any]]
        Ranked list of ``{"model": BaseFoundationModel, "score": float}``
        dicts, sorted by calibration score descending.
    """
    rankings: list[dict[str, Any]] = []
    for model in candidates:
        judge = LLMaJEvaluator(model=model)
        result = judge.evaluate_metrics(eval_data, [Metrics.JUDGE_ANSWER_RELEVANCE])
        scores = _ordered_question_scores(result, Metrics.JUDGE_ANSWER_RELEVANCE.name)
        calibration_score = _spread_and_stability_score(scores)
        rankings.append({"model": model, "score": calibration_score})
        logger.info("Judge calibration for %s: score=%.4f", model.model_id, calibration_score)

    rankings.sort(
        key=lambda item: (
            item["score"],
            item["model"].model_id == reference_model_id,
            item["model"].model_id,
        ),
        reverse=True,
    )
    return rankings


def _spread_and_stability_score(scores: list[float | None]) -> float:
    """Combine score spread and successful-call ratio for judge calibration.

    Parameters
    ----------
    scores
        Per-question scores returned by a judge candidate.  ``None``
        entries indicate failed judge calls.

    Returns
    -------
    float
        ``std(valid_scores) * (n_valid / n_total)``, or ``-1.0`` when
        fewer than two valid scores are available.
    """
    valid = [s for s in scores if s is not None]
    if len(valid) < 2:
        return -1.0
    stability = len(valid) / len(scores)
    return float(np.std(valid)) * stability


def _run_reference_rag(
    foundation_model: BaseFoundationModel,
    embedding_model: BaseEmbeddingModel,
    documents: list[DoclingDocument],
    benchmark_data: BenchmarkData,
    max_threads: int,
) -> list:
    """Build a lightweight RAG pipeline and generate calibration responses.

    Parameters
    ----------
    foundation_model
        Generation model used to produce answers.
    embedding_model
        Embedding model used to index document chunks.
    documents
        Parsed documents to chunk and embed.
    benchmark_data
        Calibration subset of benchmark questions and expected answers.
    max_threads
        Maximum concurrent threads for RAG inference.

    Returns
    -------
    list
        List of :class:`EvaluationData` instances ready for judge scoring.
    """
    chunker = LangChainChunker(chunk_size=512, method="recursive", chunk_overlap=128)
    chunks = chunker.split_documents(documents)
    vector_store = ChromaVectorStore(embedding_model=embedding_model, collection_name="ai4rag_judge_calibration")
    vector_store.add_documents(chunks)
    retriever = Retriever(vector_store=vector_store, number_of_chunks=3, method="simple", search_mode="vector")
    rag = SimpleRAG(foundation_model=foundation_model, retriever=retriever)
    inference_response = query_rag(rag=rag, questions=list(benchmark_data.questions), max_threads=max_threads)
    return build_evaluation_data(benchmark_data=benchmark_data, inference_response=inference_response)


def _ordered_question_scores(evaluation_result: EvaluationMetricsResult, metric_name: str) -> list[float | None]:
    """Extract per-question scores for a metric, sorted by question id.

    Parameters
    ----------
    evaluation_result
        Structured result returned by :meth:`LLMaJEvaluator.evaluate_metrics`.
    metric_name
        Name of the metric to extract (e.g. ``"answer_relevance"``).

    Returns
    -------
    list[float | None]
        Scores ordered by question id.  Missing entries become ``None``.
    """
    scored: dict[str, float | None] = {}
    for qs in evaluation_result.get("question_scores", []):
        value: float | None = None
        for m in qs["metrics"]:
            if m["name"] == metric_name:
                value = m["value"]
                break
        scored[qs["question_id"]] = value
    return [scored.get(qid) for qid in sorted(scored.keys())]
