# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Literal, TypeAlias, TypedDict, TypeVar

from ai4rag import logger
from ai4rag.core.experiment.benchmark_data import BenchmarkData
from ai4rag.core.experiment.exception_handler import GenerationError
from ai4rag.evaluator.base_evaluator import EvaluationData
from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.foundation_models.base_model import FoundationModel
from ai4rag.rag.template.base_template import BaseRAGTemplate
from ai4rag.utils.constants import AI4RAGParamNames

T = TypeVar("T")

__all__ = [
    "VectorStoreType",
    "RAGExperimentError",
    "RAGParamsType",
    "query_rag",
    "build_evaluation_data",
    "get_retrieval_params",
    "get_chunking_params",
    "RAGRetrievalParamsType",
]

_semantic_chunker_cache = {}

VectorStoreType: TypeAlias = Literal["milvus", "chroma", "local_milvus", "elasticsearch", "local_elasticsearch"]


class RAGExperimentError(Exception):
    """Exception representing error in the experiment."""


class RAGParamsType(TypedDict):
    """Parameters required for single AutoRAG Pattern evaluation."""

    embedding_model: EmbeddingModel
    foundation_model: FoundationModel
    chunk_size: int
    chunk_overlap: int | float
    chunking_method: Literal["recursive"]
    window_size: int
    number_of_chunks: int
    retrieval_method: Literal["simple", "window"]


class RAGChunkingParamsType(TypedDict):
    """Required chunking parameters."""

    chunk_size: int
    chunk_overlap: int | float
    chunking_method: Literal["recursive"]


class RAGRetrievalParamsType(TypedDict):
    """Required retrieval parameters."""

    retrieval_window_size: int
    number_of_retrieved_chunks: int
    retrieval_method: Literal["simple", "window"]


def query_rag(rag: BaseRAGTemplate, questions: list[str], max_threads: int = 10) -> list[dict[str, Any]]:
    """
    Function to perform parallel queries on RAG inference service.

    Parameters
    ----------
    rag : BaseRAGTemplate
        Instance of the BaseRAGTemplate to be used for retrieval-augmented generation.

    questions : list[str]
        Questions used for AI Service (RAG).

    max_threads : int, default=10
        Limit of the concurrent workers querying the AI service. Value of 10
        is chosen based on the client's inference endpoint settings.

    Returns
    -------
    list[dict[str, Any]]
        List of dicts as in the _generate_response.
    """
    logger.debug(
        "Starting concurrent RAG execution. Limit of concurrent executions: %s for %s calls. Model: %s",
        max_threads,
        len(questions),
        rag.foundation_model.model_id,
    )

    try:
        _generate_function = partial(_generate_response, rag=rag)

        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            responses = list(executor.map(_generate_function, questions))

    except Exception as exc:
        raise GenerationError(exc, model_id=rag.foundation_model.model_id) from exc

    logger.debug("Finished concurrent RAG execution!")

    return responses


def _generate_response(question: str, rag: BaseRAGTemplate) -> dict[str, Any]:
    """
    Make a single call to the RAG instance.
    Notice that question parameter should remain first to be easily
    utilised by concurrent executor.

    Parameters
    ----------
    question : str
        Question for the RAG.

    rag : BaseRAGTemplate
        Instance capable of performing Retrieval-Augmented Generation.

    Returns
    -------
    dict[str, Any]
        Example result:
        {
            "question": "What is the meaning of life?",
            "answer": "Being good to other people."
            "reference_documents": [
                {"page_content": "Document content 1", "metadata": {"document_id": "doc_id_1", ...}},
                {"page_content": "Document content 2", "metadata": {"document_id": "doc_id_2", ...}},
                ...,
            ]
        }
    """
    return rag.generate(question)


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
            contexts.append(getattr(el, "page_content", None))
            context_ids.append(getattr(el, "metadata", {}).get("document_id"))

        evaluation_data.append(
            EvaluationData(
                question=benchmark_data.questions[idx],
                answer=inference_response[idx]["answer"],
                contexts=contexts,
                context_ids=context_ids,
                ground_truths=benchmark_data.answers[idx],
                question_id=benchmark_data.questions_ids[idx],
                ground_truths_context_ids=benchmark_data.document_ids[idx] if benchmark_data.document_ids else None,
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
        if chunk_overlap is None or (chunk_overlap < 0 or chunk_overlap > 1):
            raise ValueError(
                "chunk_overlap is expected to be an integer >= 0 or a floating-point number between 0 and 1."
            )
        chunk_overlap = int(chunk_size * chunk_overlap)
    return chunk_overlap


def get_chunking_params(rag_params: RAGChunkingParamsType) -> dict:
    """
    Extracts chunking parameters from the provided rag parameters.
    All three configurations are mandatory as part of single `chunking` setting:
        `method`, `chunk_size`, `chunk_overlap`

    Parameters
    ----------
    rag_params : RAGParamsType
        Dictionary with chunking setting for single evaluation run.

    Returns
    -------
    dict
        Dictionary with chunking parameters: chunking_method, chunk_size, chunk_overlap.

    Raises
    ------
    RAGExperimentError
        Raised when chunking parameters are missing.
    """
    chunking_params = {
        k: rag_params.get(k)
        for k in [AI4RAGParamNames.CHUNKING_METHOD, AI4RAGParamNames.CHUNK_SIZE, AI4RAGParamNames.CHUNK_OVERLAP]
    }

    if chunking_params[AI4RAGParamNames.CHUNKING_METHOD] == "semantic":
        chunking_params[AI4RAGParamNames.CHUNK_OVERLAP] = 0
    else:
        chunking_params[AI4RAGParamNames.CHUNK_OVERLAP] = _get_chunk_overlap(
            chunking_params[AI4RAGParamNames.CHUNK_SIZE], chunking_params[AI4RAGParamNames.CHUNK_OVERLAP]
        )
    if any(v is None for v in chunking_params.values()):
        raise RAGExperimentError(f"Missing or invalid values in chunking configuration: {chunking_params}.")

    return chunking_params


def get_retrieval_params(rag_params: RAGParamsType) -> RAGRetrievalParamsType:
    """
    Extracts retrieval parameters from the provided rag parameters.
    All three setting's configurations are mandatory under `retrieval` key:
        `method`, `window_size`, `number_of_chunks`

    Parameters
    ----------
    rag_params : RAGParamsType
        Dictionary with retrieval setting for single evaluation run.

    Returns
    -------
    RAGRetrievalParamsType
        retrieval_method, retrieval_window_size, number_of_retrieved_chunks.

    Raises
    ------
    RAGExperimentError
        Raised when retrieval parameters are missing.
    """
    retrieval_method = rag_params.get("retrieval_method")
    retrieval_window_size = rag_params.get("window_size")
    number_of_retrieved_chunks = rag_params.get("number_of_chunks")

    retrieval_params = {
        AI4RAGParamNames.WINDOW_SIZE: retrieval_window_size,
        AI4RAGParamNames.NUMBER_OF_CHUNKS: number_of_retrieved_chunks,
        AI4RAGParamNames.RETRIEVAL_METHOD: retrieval_method,
    }
    if retrieval_window_size is None or not all((retrieval_method, number_of_retrieved_chunks)):
        p = {
            "window_size": retrieval_window_size,
            "number_of_chunks": number_of_retrieved_chunks,
            "method": retrieval_method,
        }
        raise RAGExperimentError(f"Missing or invalid values in retrieval configuration: {p}.")

    return retrieval_params
