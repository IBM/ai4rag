# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypeAlias, TypedDict

from ai4rag import logger
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.exception_handler import GenerationError
from ai4rag.evaluator.base_evaluator import EvaluationData, EvaluationMetricsResult, QuestionMetric
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.template.base_template import BaseRAGTemplate
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = [
    "VectorStoreType",
    "RAGExperimentError",
    "RAGParamsType",
    "query_rag",
    "build_evaluation_data",
    "merge_evaluation_results",
    "get_retrieval_params",
    "get_chunking_params",
    "RAGRetrievalParamsType",
]

VectorStoreType: TypeAlias = Literal["milvus", "milvus_lite", "pgvector"]


class RAGExperimentError(Exception):
    """Exception representing error in the experiment."""


class RAGParamsType(TypedDict):
    """Parameters required for single AutoRAG Pattern evaluation."""

    template_id: str
    embedding_model: BaseEmbeddingModel
    foundation_model: BaseFoundationModel
    chunk_size: int
    chunk_overlap: int | float
    chunking_method: Literal["recursive", "hybrid"]
    window_size: int
    number_of_chunks: int
    retrieval_method: Literal["simple", "window"]
    search_mode: Literal["vector", "hybrid", "graph"]
    ranker_strategy: str
    ranker_k: int
    ranker_alpha: int | float


class RAGChunkingParamsType(TypedDict):
    """Required chunking parameters."""

    chunk_size: int
    chunk_overlap: int | float
    chunking_method: Literal["recursive", "hybrid"]


class RAGRetrievalParamsType(TypedDict):
    """Resolved retrieval parameters for one candidate."""

    window_size: int
    number_of_chunks: int
    retrieval_method: Literal["simple", "window"]
    search_mode: Literal["vector", "hybrid", "graph"]
    ranker_strategy: str | None
    ranker_k: int | None
    ranker_alpha: int | float | None


def query_rag(rag: BaseRAGTemplate, questions: list[str], max_threads: int = 10) -> list[dict[str, Any]]:
    """
    Generate responses to benchmark questions concurrently.

    Parameters
    ----------
    rag : BaseRAGTemplate
        Instance of the BaseRAGTemplate to be used for retrieval-augmented generation.

    questions : list[str]
        Questions used for AI Service (RAG).

    max_threads : int
        Limit of the concurrent workers querying the AI service.

    Returns
    -------
    list[dict[str, Any]]
        Template responses in question order. Built-in templates return the
        question, answer, and retrieved ``AI4RAGChunk`` objects under
        ``reference_documents``. Agentic templates also return a conversation.
    """
    logger.debug(
        "Starting concurrent RAG execution. Limit of concurrent executions: %s for %s calls. Model: %s",
        max_threads,
        len(questions),
        rag.foundation_model.model_id,
    )

    try:
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            responses = list(executor.map(rag.generate, questions))

    except Exception as exc:
        raise GenerationError(exc, model_id=rag.foundation_model.model_id) from exc

    logger.debug("Finished concurrent RAG execution!")

    return responses


def build_evaluation_data(
    benchmark_data: BenchmarkData, inference_response: list[dict[str, Any]]
) -> list[EvaluationData]:
    """
    Helper function responsible for building payload for response evaluation.

    Parameters
    ----------
    benchmark_data : BenchmarkData
        Instance holding information about questions, answers and ids.

    inference_response : list[dict[str, Any]]
        List of model's responses containing question, answer and used
        reference documents for each record.

    Returns
    -------
    list[EvaluationData]
        Sequence containing data that will be used for evaluation.
    """
    evaluation_data = []

    for idx in range(len(benchmark_data)):
        contexts = []
        context_ids = []
        for el in inference_response[idx]["reference_documents"]:
            contexts.append(el.text)
            context_ids.append(el.metadata.get("document_id"))

        evaluation_data.append(
            EvaluationData(
                question=benchmark_data.questions[idx],
                answer=inference_response[idx]["answer"],
                contexts=contexts,
                context_ids=context_ids,
                ground_truths=benchmark_data.correct_answers[idx],
                question_id=benchmark_data.questions_ids[idx],
                ground_truths_context_ids=benchmark_data.document_keys[idx],
            )
        )

    return evaluation_data


def _get_chunk_overlap(chunk_size: int, chunk_overlap: int | float) -> int:
    """
    Get chunking overlap as number of tokens/characters used as cross-chunk overlap

    Parameters
    ----------
    chunk_size : int
        Size of the created chunks.

    chunk_overlap : int | float
        If "int", the chunk_overlap is considered as number of token/characters.
        If "float", it's expected to be in the range [0, 1] and it'll be treated as a
        percentage of the chunk_size.

    Returns
    -------
    int
        number of characters/tokens used as overlap between chunks
    """
    if isinstance(chunk_overlap, float):
        if chunk_overlap < 0 or chunk_overlap > 1:
            raise ValueError(
                "chunk_overlap is expected to be an integer >= 0 or a floating-point number between 0 and 1."
            )
        chunk_overlap = int(chunk_size * chunk_overlap)
    return chunk_overlap


def get_chunking_params(rag_params: RAGParamsType) -> RAGChunkingParamsType:
    """
    Extract chunking settings and normalize fractional overlap to an integer.

    Parameters
    ----------
    rag_params : RAGParamsType
        Candidate settings containing ``chunking_method``, ``chunk_size``, and
        ``chunk_overlap``.

    Returns
    -------
    RAGChunkingParamsType
        The three chunking settings with normalized overlap.

    Raises
    ------
    RAGExperimentError
        Raised when chunking parameters are missing.
    """
    chunking_params = {
        k: rag_params.get(k)
        for k in [AI4RAGParamNames.CHUNKING_METHOD, AI4RAGParamNames.CHUNK_SIZE, AI4RAGParamNames.CHUNK_OVERLAP]
    }
    chunking_params[AI4RAGParamNames.CHUNK_OVERLAP] = _get_chunk_overlap(
        chunking_params[AI4RAGParamNames.CHUNK_SIZE], chunking_params[AI4RAGParamNames.CHUNK_OVERLAP]
    )
    if any(v is None for v in chunking_params.values()):
        raise RAGExperimentError(f"Missing or invalid values in chunking configuration: {chunking_params}.")

    return chunking_params


def get_retrieval_params(rag_params: RAGParamsType) -> RAGRetrievalParamsType:
    """
    Extract retrieval settings from a candidate.

    Parameters
    ----------
    rag_params : RAGParamsType
        Candidate settings containing ``retrieval_method``, ``window_size``,
        ``number_of_chunks``, and ``search_mode``.

    Returns
    -------
    RAGRetrievalParamsType
        The four required retrieval settings and nullable ranker settings.

    Raises
    ------
    RAGExperimentError
        Raised when retrieval parameters are missing.
    """
    retrieval_params = {
        AI4RAGParamNames.WINDOW_SIZE: rag_params.get(AI4RAGParamNames.WINDOW_SIZE),
        AI4RAGParamNames.NUMBER_OF_CHUNKS: rag_params.get(AI4RAGParamNames.NUMBER_OF_CHUNKS),
        AI4RAGParamNames.SEARCH_MODE: rag_params.get(AI4RAGParamNames.SEARCH_MODE),
        AI4RAGParamNames.RETRIEVAL_METHOD: rag_params.get(AI4RAGParamNames.RETRIEVAL_METHOD),
        AI4RAGParamNames.RANKER_STRATEGY: rag_params.get(AI4RAGParamNames.RANKER_STRATEGY),
        AI4RAGParamNames.RANKER_K: rag_params.get(AI4RAGParamNames.RANKER_K),
        AI4RAGParamNames.RANKER_ALPHA: rag_params.get(AI4RAGParamNames.RANKER_ALPHA),
    }
    required_keys = (
        AI4RAGParamNames.WINDOW_SIZE,
        AI4RAGParamNames.NUMBER_OF_CHUNKS,
        AI4RAGParamNames.SEARCH_MODE,
        AI4RAGParamNames.RETRIEVAL_METHOD,
    )
    if any(retrieval_params.get(k) is None for k in required_keys):
        raise RAGExperimentError(f"Missing or invalid values in retrieval configuration: {retrieval_params}.")

    return retrieval_params


def merge_evaluation_results(results: list[EvaluationMetricsResult]) -> EvaluationMetricsResult:
    """Merge partial ``EvaluationMetricsResult`` dicts from multiple evaluators.

    Aggregate metrics are concatenated. Per-question scores are joined by
    ``question_id`` so each question carries metrics from all evaluators.

    Parameters
    ----------
    results
        Partial results, one per evaluator.

    Returns
    -------
    EvaluationMetricsResult
        Single merged result.
    """
    if not results:
        return EvaluationMetricsResult(metrics=[], question_scores=[])
    if len(results) == 1:
        return results[0]

    all_metrics = [m for r in results for m in r["metrics"]]

    scores_by_qid: dict[str, list[QuestionMetric]] = {}
    for result in results:
        for qs in result["question_scores"]:
            scores_by_qid.setdefault(qs["question_id"], []).extend(qs["metrics"])

    return EvaluationMetricsResult(
        metrics=all_metrics,
        question_scores=[{"question_id": qid, "metrics": qmetrics} for qid, qmetrics in scores_by_qid.items()],
    )
