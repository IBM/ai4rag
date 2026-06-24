# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

import copy

import pytest

from ai4rag.components.assets_generator import build_pattern_json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pattern(**overrides) -> dict:
    """Build a minimal pattern dict matching the schema expected by build_pattern_json."""
    base = {
        "name": "pattern_001",
        "settings": {
            "vector_store_binding": {
                "provider_id": "provider-123",
                "provider_type": "milvus",
                "vector_store_id": "test_collection_001",
            },
            "chunking": {
                "method": "recursive",
                "chunk_size": 512,
                "chunk_overlap": 50,
            },
            "embedding": {
                "model_id": "ibm/slate-125m-english-rtrvr",
                "distance_metric": "cosine",
                "embedding_params": {"embedding_dimension": 768},
            },
            "retrieval": {
                "method": "simple",
                "number_of_chunks": 5,
            },
            "generation": {
                "model_id": "ibm/granite-3-8b-instruct",
                "temperature": 0.7,
                "max_completion_tokens": 1024,
                "system_message_text": "Answer based on context only.",
                "user_message_text": "Context: {reference_documents}\nQ: {question}",
                "context_template_text": "{document}",
            },
        },
    }
    for key, value in overrides.items():
        keys = key.split(".")
        target = base
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    return base


# ---------------------------------------------------------------------------
# build_pattern_json -- responses_template generation
# ---------------------------------------------------------------------------


class TestBuildPatternJson:
    """Verify that build_pattern_json populates responses_template correctly."""

    def test_adds_responses_template(self):
        """A responses_template section must be added to settings."""
        pattern = _make_pattern()
        result = build_pattern_json(pattern)

        rt = result["settings"]["responses_template"]
        assert rt["model"] == "ibm/granite-3-8b-instruct"
        assert rt["stream"] is False
        assert rt["store"] is False
        assert rt["input"] == [
            {
                "content": [{"text": "Answer based on context only.", "type": "input_text"}],
                "role": "system",
            },
            {"content": [{"text": "<user_query_placeholder>", "type": "input_text"}], "role": "user"},
        ]
        assert rt["max_output_tokens"] == 1024
        assert rt["temperature"] == 0.7
        assert rt["tool_choice"] == {"mode": "required", "tools": [{}], "type": "file_search"}
        assert len(rt["tools"]) == 1
        assert rt["tools"][0]["type"] == "file_search"
        assert "test_collection_001" in rt["tools"][0]["vector_store_ids"]
        assert rt["tools"][0]["ranking_options"]["max_num_results"] == 5
        assert rt["tools"][0]["max_num_results"] == 5
        assert rt["include"] == ["file_search_call.results"]

    def test_returns_same_dict(self):
        """The function must return the same dict it received (mutated in place)."""
        pattern = _make_pattern()
        result = build_pattern_json(pattern)
        assert result is pattern

    def test_no_detected_language_by_default(self):
        """When detected_language is None, no detected_language key appears in generation."""
        pattern = _make_pattern()
        build_pattern_json(pattern)
        assert "detected_language" not in pattern["settings"]["generation"]

    def test_detected_language_injected(self):
        """Non-English language detection must inject detected_language into generation."""
        pattern = _make_pattern()
        lang = {"code": "de", "name": "German"}
        build_pattern_json(pattern, detected_language=lang)
        assert pattern["settings"]["generation"]["detected_language"] == lang

    def test_hybrid_rrf_ranking_options(self):
        """Hybrid search with RRF ranker must merge impact_factor into ranking_options."""
        pattern = _make_pattern()
        pattern["settings"]["retrieval"]["search_mode"] = "hybrid"
        pattern["settings"]["retrieval"]["ranker_strategy"] = "rrf"
        pattern["settings"]["retrieval"]["ranker_k"] = 60

        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro["impact_factor"] == 60
        assert ro["max_num_results"] == 5

    def test_hybrid_weighted_ranking_options(self):
        """Hybrid search with weighted ranker must merge alpha into ranking_options."""
        pattern = _make_pattern()
        pattern["settings"]["retrieval"]["search_mode"] = "hybrid"
        pattern["settings"]["retrieval"]["ranker_strategy"] = "weighted"
        pattern["settings"]["retrieval"]["ranker_alpha"] = 0.7

        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro["alpha"] == 0.7
        assert ro["max_num_results"] == 5

    def test_simple_retrieval_default_ranking_options(self):
        """Simple retrieval must have default ranker and weights in ranking_options."""
        pattern = _make_pattern()
        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro == {
            "max_num_results": 5,
            "ranker": "auto",
            "weights": {"vector": 1.0, "neural": 0.0, "keyword": 0.0},
        }
        assert pattern["settings"]["responses_template"]["tools"][0]["max_num_results"] == 5

    def test_preserves_existing_pattern_fields(self):
        """Existing pattern fields (name, chunking, embedding, etc.) must not be altered."""
        pattern = _make_pattern()
        original_name = pattern["name"]
        original_chunking = copy.deepcopy(pattern["settings"]["chunking"])

        build_pattern_json(pattern)

        assert pattern["name"] == original_name
        assert pattern["settings"]["chunking"] == original_chunking
