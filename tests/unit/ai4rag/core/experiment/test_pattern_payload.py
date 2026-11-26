# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.core.experiment.pattern_payload import build_pattern_payload
from ai4rag.core.experiment.results import EvaluationResult
from ai4rag.evaluator.base_evaluator import build_aggregate_metric
from ai4rag.evaluator.metric import Metrics


@pytest.mark.parametrize(
    ("ranker_strategy", "ranker_settings"),
    [("rrf", {"ranker_k": 60}), ("weighted", {"ranker_alpha": 0.25})],
)
def test_build_pattern_payload_preserves_hybrid_settings_and_objective(ranker_strategy, ranker_settings):
    unitxt_metric = build_aggregate_metric(Metrics.FAITHFULNESS, (None, None), mean=0.6)
    ragas_metric = build_aggregate_metric(Metrics.RAGAS_FAITHFULNESS, (None, None), mean=0.72)
    embedding = {"model_id": "embedding", "embedding_params": {"embedding_dimension": 384}}
    generation = {"model_id": "generator", "temperature": 0.2}
    result = EvaluationResult(
        pattern_name="PatternB-warm-start",
        collection="ai4rag_test",
        indexing_params={
            "chunking": {"chunking_method": "hybrid", "chunk_size": 512, "chunk_overlap": 0, "include_metadata": True},
            "embedding": embedding,
        },
        rag_params={
            "template_id": "agentic_rag",
            "retrieval": {
                "retrieval_method": "window",
                "window_size": 2,
                "number_of_chunks": 5,
                "search_mode": "hybrid",
                "ranker_strategy": ranker_strategy,
                "ranker_k": 60,
                "ranker_alpha": 0.25,
            },
            "generation": generation,
        },
        scores={"metrics": [unitxt_metric, ragas_metric], "question_scores": []},
        execution_time=12.75,
        final_score=0.72,
    )

    payload = build_pattern_payload(
        result,
        vector_store_provider="pgvector",
        optimization_metric=Metrics.RAGAS_FAITHFULNESS,
        max_combinations=20,
        pattern_name="PatternA",
        iteration=0,
    )

    assert payload == {
        "name": "PatternA",
        "template_id": "agentic_rag",
        "max_combinations": 20,
        "evaluation": {"metrics": [unitxt_metric, {**ragas_metric, "optimization_metric": True}]},
        "duration_seconds": 12,
        "settings": {
            "store_binding": {"provider_type": "pgvector", "collection_name": "ai4rag_test"},
            "chunking": {"method": "hybrid", "chunk_size": 512, "chunk_overlap": 0, "include_metadata": True},
            "embedding": embedding,
            "retrieval": {
                "method": "window",
                "number_of_chunks": 5,
                "search_mode": "hybrid",
                "window_size": 2,
                "ranker_strategy": ranker_strategy,
                **ranker_settings,
            },
            "generation": generation,
        },
        "iteration": 0,
    }
    assert all("optimization_metric" not in metric for metric in result.scores["metrics"])
