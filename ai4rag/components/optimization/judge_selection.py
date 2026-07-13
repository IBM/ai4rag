# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from docling_core.types.doc import DoclingDocument

from ai4rag import handler
from ai4rag.components.utils.ogx_client import openai_compatible_base_url
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.utils import build_evaluation_data, query_rag
from ai4rag.evaluator.base_evaluator import MetricType
from ai4rag.evaluator.llmaj_evaluator import LLMaJConfig, LLMaJEvaluator
from ai4rag.rag.chunking.langchain_chunker import LangChainChunker
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.retrieval.retriever import Retriever
from ai4rag.rag.template.simple_rag_template import SimpleRAG
from ai4rag.rag.vector_store.chroma import ChromaVectorStore

_logger = logging.getLogger("judge-selection")
_logger.addHandler(handler)

UNITXT_EVALUATION_METRICS = [
    MetricType.FAITHFULNESS,
    MetricType.ANSWER_CORRECTNESS,
    MetricType.CONTEXT_CORRECTNESS,
]
JUDGE_EVALUATION_METRICS = [MetricType.ANSWER_RELEVANCE]


@dataclass
class JudgeSelectionContext:
    """Inputs required to resolve or calibrate a judge model."""

    generation_models: list[BaseFoundationModel]
    embedding_models: list[BaseEmbeddingModel]
    benchmark_data: BenchmarkData
    documents: list[DoclingDocument]
    ogx_base_url: str
    ogx_api_key: str
    max_threads: int = 10


def calibration_subset_size(total_rows: int) -> int:
    """Return calibration row count: min(20, 10% of benchmark rows)."""
    if total_rows <= 0:
        return 0
    return max(1, min(20, int(total_rows * 0.1)))


def build_evaluation_block(judge_model_id: str) -> list[dict[str, Any]]:
    """Build the ADR ``evaluation`` artifact describing metric backends."""
    return [
        {
            "evaluator": "judge",
            "model_id": judge_model_id,
            "metrics": list(JUDGE_EVALUATION_METRICS),
        },
        {
            "evaluator": "unitxt",
            "metrics": list(UNITXT_EVALUATION_METRICS),
        },
    ]


def select_judge_model(context: JudgeSelectionContext) -> str:
    """
    Resolve the judge model used for ``answer_relevance``.

    When only one generation model is available it is used as the judge.
    Otherwise auto-selects from the generation pool using answer-relevance calibration.
    """
    candidates = [model.model_id for model in context.generation_models]
    if not candidates:
        raise ValueError("At least one generation model is required to select a judge model.")
    if len(candidates) == 1:
        return candidates[0]

    return _calibrate_judge_model(context, candidates)


def _calibrate_judge_model(context: JudgeSelectionContext, candidates: list[str]) -> str:
    subset_size = calibration_subset_size(len(context.benchmark_data.questions))
    calibration_benchmark = context.benchmark_data.get_random_sample(n_records=subset_size, random_seed=17)
    eval_data = _run_reference_rag(
        foundation_model=context.generation_models[0],
        embedding_model=context.embedding_models[0],
        documents=context.documents,
        benchmark_data=calibration_benchmark,
        max_threads=context.max_threads,
    )

    reference_generation_model_id = context.generation_models[0].model_id
    rankings = _score_judge_candidates(
        candidates=candidates,
        eval_data=eval_data,
        ogx_base_url=context.ogx_base_url,
        ogx_api_key=context.ogx_api_key,
        reference_generation_model_id=reference_generation_model_id,
    )
    selected = rankings[0]["model_id"]
    _logger.info("Selected judge model: %s (calibration score=%s)", selected, rankings[0]["judge_calibration_score"])
    return selected


def _score_judge_candidates(
    *,
    candidates: list[str],
    eval_data: list,
    ogx_base_url: str,
    ogx_api_key: str,
    reference_generation_model_id: str,
) -> list[dict[str, Any]]:
    rankings: list[dict[str, Any]] = []
    for model_id in candidates:
        judge = LLMaJEvaluator(
            LLMaJConfig(
                base_url=openai_compatible_base_url(ogx_base_url),
                api_key=ogx_api_key,
                model=model_id,
            )
        )
        judge_scores = judge.evaluate_metrics(eval_data, [MetricType.ANSWER_RELEVANCE])
        candidate_scores = _ordered_question_scores(judge_scores, MetricType.ANSWER_RELEVANCE)
        calibration_score = _spread_and_stability_score(candidate_scores)
        rankings.append({"model_id": model_id, "judge_calibration_score": calibration_score})
        _logger.info("Judge calibration for %s: score=%.4f", model_id, calibration_score)

    rankings.sort(
        key=lambda item: (
            item["judge_calibration_score"] if item["judge_calibration_score"] is not None else -2.0,
            item["model_id"] == reference_generation_model_id,
            item["model_id"],
        ),
        reverse=True,
    )
    return rankings


def _spread_and_stability_score(scores: list[float | None]) -> float:
    """Combine score spread and successful-call ratio for judge calibration."""
    valid = [score for score in scores if score is not None]
    if len(valid) < 2:
        return -1.0
    stability = len(valid) / len(scores)
    return float(np.std(valid)) * stability


def _run_reference_rag(
    *,
    foundation_model: BaseFoundationModel,
    embedding_model: BaseEmbeddingModel,
    documents: list[DoclingDocument],
    benchmark_data: BenchmarkData,
    max_threads: int,
) -> list:
    chunker = LangChainChunker(chunk_size=512, method="recursive", chunk_overlap=128)
    chunks = chunker.split_documents(documents)
    vector_store = ChromaVectorStore(embedding_model=embedding_model, collection_name="judge_calibration")
    vector_store.add_documents(chunks)
    retriever = Retriever(vector_store=vector_store, number_of_chunks=3, method="simple", search_mode="vector")
    rag = SimpleRAG(foundation_model=foundation_model, retriever=retriever)
    inference_response = query_rag(
        rag=rag,
        questions=list(benchmark_data.questions),
        max_threads=max_threads,
    )
    return build_evaluation_data(benchmark_data=benchmark_data, inference_response=inference_response)


def _ordered_question_scores(evaluation_result: dict, metric: str) -> list[float | None]:
    question_scores = (evaluation_result.get("question_scores") or {}).get(metric) or {}
    return [question_scores.get(qid) for qid in sorted(question_scores.keys())]


def resolve_judge_model_id(evaluation: Any) -> str:
    """Read the selected judge ``model_id`` from a search-space or pattern evaluation block."""
    if isinstance(evaluation, list):
        for block in evaluation:
            if block.get("evaluator") == "judge":
                model_id = block.get("model_id")
                if model_id:
                    return model_id
    if isinstance(evaluation, dict):
        if evaluation.get("evaluator") == "judge":
            return evaluation.get("model_id") or evaluation.get("judge_model_id") or ""
    raise ValueError("Judge model_id is missing from the search-space evaluation block.")
