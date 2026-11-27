# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from ai4rag.core.experiment.results import EvaluationResult
from ai4rag.evaluator.metric import RAGMetric
from ai4rag.rag.vector_store.neo4j import Neo4jGraphRetrievalConfig
from ai4rag.utils.constants import AI4RAGParamNames


def build_pattern_payload(
    evaluation_result: EvaluationResult,
    *,
    vector_store_provider: str,
    optimization_metric: RAGMetric,
    max_combinations: int,
    pattern_name: str,
    iteration: int,
) -> dict[str, Any]:
    """Build the pattern payload for a completed evaluation.

    Parameters
    ----------
    evaluation_result : EvaluationResult
        Evaluated candidate and its aggregate scores.
    vector_store_provider : str
        Backend that owns the candidate's collection.
    optimization_metric : RAGMetric
        Metric to mark as the optimization objective.
    max_combinations : int
        Number of combinations in the experiment's search space.
    pattern_name : str
        Name under which to publish the candidate.
    iteration : int
        Iteration under which to publish the candidate.

    Returns
    -------
    dict[str, Any]
        Pattern metadata, settings, and aggregate metrics for the event handler.
    """
    retrieval_payload = _build_retrieval_payload(evaluation_result.rag_params["retrieval"])
    indexing_params = evaluation_result.indexing_params
    chunking_params = indexing_params["chunking"]
    indexing_payload = {
        "chunking": {
            "method": chunking_params[AI4RAGParamNames.CHUNKING_METHOD],
            "chunk_size": chunking_params[AI4RAGParamNames.CHUNK_SIZE],
            "chunk_overlap": chunking_params[AI4RAGParamNames.CHUNK_OVERLAP],
            "include_metadata": chunking_params["include_metadata"],
        },
        "embedding": indexing_params.get("embedding"),
    }
    if vector_store_provider == "neo4j" and (knowledge_graph := indexing_params.get("knowledge_graph")) is not None:
        indexing_payload["graph_extraction"] = knowledge_graph["extraction_config"]

    # Different evaluators can report a metric with the same name.
    objective = (optimization_metric.name, optimization_metric.evaluator)
    metrics_payload = [
        {**metric, "optimization_metric": True} if (metric["name"], metric["evaluator"]) == objective else metric
        for metric in evaluation_result.scores["metrics"]
    ]
    return {
        "name": pattern_name,
        "template_id": evaluation_result.rag_params["template_id"],
        "max_combinations": max_combinations,
        "evaluation": {"metrics": metrics_payload},
        "duration_seconds": int(evaluation_result.execution_time),
        "settings": {
            "store_binding": {
                "provider_type": vector_store_provider,
                "collection_name": evaluation_result.collection,
            },
            **indexing_payload,
            "retrieval": retrieval_payload,
            "generation": evaluation_result.rag_params.get("generation"),
        },
        "iteration": iteration,
    }


def _build_retrieval_payload(retrieval_params: dict[str, Any]) -> dict[str, Any]:
    """Include retrieval options applicable to the candidate's search mode."""
    payload = {
        "method": retrieval_params[AI4RAGParamNames.RETRIEVAL_METHOD],
        "number_of_chunks": retrieval_params[AI4RAGParamNames.NUMBER_OF_CHUNKS],
        "search_mode": retrieval_params.get(AI4RAGParamNames.SEARCH_MODE, "vector"),
    }
    if window_size := retrieval_params[AI4RAGParamNames.WINDOW_SIZE]:
        payload["window_size"] = window_size

    if payload["search_mode"] == "hybrid":
        ranker_strategy = retrieval_params.get(AI4RAGParamNames.RANKER_STRATEGY)
        payload["ranker_strategy"] = ranker_strategy
        if ranker_strategy == "rrf":
            payload["ranker_k"] = retrieval_params.get(AI4RAGParamNames.RANKER_K)
        elif ranker_strategy == "weighted":
            payload["ranker_alpha"] = retrieval_params.get(AI4RAGParamNames.RANKER_ALPHA)
    elif payload["search_mode"] == "graph":
        for key in Neo4jGraphRetrievalConfig.keys():
            if key in retrieval_params:
                payload[key] = retrieval_params[key]
    return payload
