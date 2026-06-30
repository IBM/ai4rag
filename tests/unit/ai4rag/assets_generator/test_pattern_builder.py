# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

import copy

import pytest

from ai4rag.components.assets_generator import build_pattern_json
from ai4rag.components.assets_generator.pattern_builder import (
    _is_placeholder_only_export,
    build_responses_system_input,
)
from ai4rag.components.assets_generator.prompt_filters import normalize_answer_scaffold
from ai4rag.search_space.src.model_props import get_system_message_text, get_user_message_text

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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("   ", True),
        ("{reference_documents}", True),
        ("{reference_documents}\n{question}", True),
        ("foo {reference_documents}", False),
        ("You are a helpful assistant.", False),
    ],
)
def test_is_placeholder_only_export(text: str, expected: bool):
    """Placeholder-only export text must trigger the empty-input fallback path."""
    assert _is_placeholder_only_export(text) == expected


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
        generation = result["settings"]["generation"]
        expected_system = build_responses_system_input(generation)

        assert rt["model"] == "ibm/granite-3-8b-instruct"
        assert rt["stream"] is False
        assert rt["store"] is False
        assert rt["input"] == [
            {
                "content": [{"text": expected_system, "type": "input_text"}],
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
        assert rt["tools"][0]["max_num_results"] == 5
        assert rt["include"] == ["file_search_call.results"]

    def test_returns_same_dict(self):
        """The function must return the same dict it received (mutated in place)."""
        pattern = _make_pattern()
        result = build_pattern_json(pattern)
        assert result is pattern

    def test_hybrid_rrf_ranking_options(self):
        """Hybrid search with RRF ranker must set ranker and impact_factor in ranking_options."""
        pattern = _make_pattern()
        pattern["settings"]["retrieval"]["search_mode"] = "hybrid"
        pattern["settings"]["retrieval"]["ranker_strategy"] = "rrf"
        pattern["settings"]["retrieval"]["ranker_k"] = 60

        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro == {"ranker": "rrf", "impact_factor": 60}
        assert pattern["settings"]["responses_template"]["tools"][0]["max_num_results"] == 5

    def test_hybrid_weighted_ranking_options(self):
        """Hybrid search with weighted ranker must set ranker and alpha in ranking_options."""
        pattern = _make_pattern()
        pattern["settings"]["retrieval"]["search_mode"] = "hybrid"
        pattern["settings"]["retrieval"]["ranker_strategy"] = "weighted"
        pattern["settings"]["retrieval"]["ranker_alpha"] = 0.7

        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro == {"ranker": "weighted", "alpha": 0.7}
        assert pattern["settings"]["responses_template"]["tools"][0]["max_num_results"] == 5

    def test_simple_retrieval_default_ranking_options(self):
        """Vector-only search simulates semantic retrieval via weighted ranker alpha=1.0."""
        pattern = _make_pattern()
        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro == {"ranker": "weighted", "alpha": 1.0}
        assert pattern["settings"]["responses_template"]["tools"][0]["max_num_results"] == 5

    def test_hybrid_weighted_alpha_one_uses_default_ranking(self):
        """Hybrid weighted with alpha=1.0 uses the default semantic-only simulation branch."""
        pattern = _make_pattern()
        pattern["settings"]["retrieval"]["search_mode"] = "hybrid"
        pattern["settings"]["retrieval"]["ranker_strategy"] = "weighted"
        pattern["settings"]["retrieval"]["ranker_alpha"] = 1.0

        build_pattern_json(pattern)

        ro = pattern["settings"]["responses_template"]["tools"][0]["ranking_options"]
        assert ro == {"ranker": "weighted", "alpha": 1.0}

    def test_export_system_input_merges_non_redundant_user_rules(self):
        """Legacy user supplements merge; redundant grounding and citations are omitted."""
        pattern = _make_pattern()
        pattern["settings"]["generation"][
            "system_message_text"
        ] = "You are a retrieval-augmented assistant. Answer using ONLY the provided documents."
        pattern["settings"]["generation"]["user_message_text"] = (
            "Answer ONLY using information from the documents below. "
            "Do not use outside knowledge.\n"
            "You MUST cite sources using [1], [2], etc.\n\n"
            "Documents:\n{reference_documents}\n\n"
            "Question: {question}\n\n"
            "Answer (max 150 words, with citations):\n"
            "You MUST write your entire answer in English only."
        )

        build_pattern_json(pattern)

        system_text = pattern["settings"]["responses_template"]["input"][0]["content"][0]["text"]
        assert "retrieval-augmented assistant" in system_text
        assert "retrieved via file search" not in system_text
        assert "provided documents" not in system_text.lower()
        assert "Answer ONLY using information from the documents below" not in system_text
        assert "must cite sources" not in system_text.lower()
        assert "file citations" not in system_text.lower()
        assert "max 150 words" in system_text
        assert "with citations" not in system_text.lower()
        assert "English only" in system_text
        assert "{reference_documents}" not in system_text
        assert "{question}" not in system_text

    def test_export_system_input_skips_duplicate_citation_and_keeps_answer_scaffold(self):
        """Citation lines are stripped; answer scaffold and language policy still merge."""
        pattern = _make_pattern()
        pattern["settings"]["generation"][
            "system_message_text"
        ] = "You are a retrieval-augmented assistant. You MUST cite sources using [1], [2]."
        pattern["settings"]["generation"]["user_message_text"] = (
            "You MUST cite sources using [1], [2], etc.\n\n"
            "Documents:\n{reference_documents}\n\n"
            "Question: {question}\n\n"
            "Answer (max 150 words, with citations):\n"
            "You MUST write your entire answer in English only."
        )

        build_pattern_json(pattern)

        system_text = pattern["settings"]["responses_template"]["input"][0]["content"][0]["text"]
        assert "must cite sources" not in system_text.lower()
        assert "max 150 words" in system_text
        assert "English only" in system_text

    def test_build_responses_system_input_strips_ogx_prefix(self):
        """Legacy grounding and citation lines are omitted; persona supplements are kept."""
        generation = {
            "system_message_text": "Short system prefix.",
            "user_message_text": (
                "Answer ONLY using information from the documents below.\n"
                "You MUST cite sources using [1], [2].\n\n"
                "Context: {reference_documents}\n\n"
                "Question: {question}\n"
            ),
        }
        system_input = build_responses_system_input(generation)
        assert system_input == "Short system prefix."
        assert "retrieved via file search" not in system_input
        assert "must cite sources" not in system_input.lower()
        assert "documents below" not in system_input

    def test_build_pattern_json_uses_export_parity_system_input(self):
        """build_pattern_json must use build_responses_system_input(), not raw system text."""
        model_id = "ibm/granite-3-8b-instruct"
        expected = build_responses_system_input(
            {
                "system_message_text": get_system_message_text(model_id),
                "user_message_text": get_user_message_text(model_id, language="English"),
            }
        )

        pattern = _make_pattern()
        pattern["settings"]["generation"]["model_id"] = model_id
        pattern["settings"]["generation"]["system_message_text"] = get_system_message_text(model_id)
        pattern["settings"]["generation"]["user_message_text"] = get_user_message_text(model_id, language="English")

        build_pattern_json(pattern)

        actual = pattern["settings"]["responses_template"]["input"][0]["content"][0]["text"]
        assert actual == expected
        assert actual != pattern["settings"]["generation"]["system_message_text"]
        assert "Granite Chat" in actual
        assert "retrieval-augmented assistant" in actual
        assert "You MUST respond in English" in actual

    @pytest.mark.parametrize(
        "model_id",
        [
            "unknown-model",
            "ibm/granite-3-8b-instruct",
            "meta-llama/llama-3-1-8b-instruct",
            "mistralai/mistral-large",
            "openai/gpt-oss-120b",
        ],
    )
    def test_export_omits_ogx_duplicative_prompt_text(self, model_id: str):
        """Export must not duplicate citation/retrieval text that OGX injects at file_search runtime."""
        generation = {
            "system_message_text": get_system_message_text(model_id),
            "user_message_text": get_user_message_text(model_id, language="English"),
        }
        system_text = build_responses_system_input(generation)

        assert "[1], [2]" not in system_text
        assert "must cite sources" not in system_text.lower()
        assert "file citations" not in system_text.lower()
        assert "documents below" not in system_text
        assert "retrieved via file search" not in system_text
        assert "retrieved to help answer" not in system_text.lower()
        assert "<|file-id|>" not in system_text
        assert "cite sources immediately" not in system_text.lower()
        assert "supporting information only" not in system_text.lower()
        assert "{reference_documents}" not in system_text
        assert "{question}" not in system_text
        assert "[End]" not in system_text

    def test_export_omits_ogx_config_yaml_instruction_text(self):
        """Export must not contain verbatim OGX annotation/context template phrases."""
        generation = {
            "system_message_text": (
                "You are a retrieval-augmented assistant. "
                "Cite sources immediately at the end of sentences before punctuation."
            ),
            "user_message_text": (
                "The above results were retrieved to help answer the user's query. "
                "Use them as supporting information only in answering this query.\n"
                "Documents:\n{reference_documents}\n\n"
                "Question: {question}\n"
            ),
        }
        system_text = build_responses_system_input(generation)
        assert system_text == "You are a retrieval-augmented assistant."

    @pytest.mark.parametrize(
        "model_id",
        [
            "meta-llama/llama-3-1-8b-instruct",
            "mistralai/mistral-large",
            "openai/gpt-oss-120b",
            "ibm/granite-3-8b-instruct",
        ],
    )
    def test_export_merges_unified_rag_instructions(self, model_id: str):
        """All model families use unified RAG instructions after PR #81."""
        generation = {
            "system_message_text": get_system_message_text(model_id),
            "user_message_text": get_user_message_text(model_id, language="English"),
        }
        system_text = build_responses_system_input(generation)
        # All models now use unified RAG structure from PR #81
        assert "retrieval-augmented assistant" in system_text
        assert "max 150 words" in system_text
        assert "You MUST respond in English" in system_text

    def test_build_responses_system_input_handles_empty_inputs(self):
        """When both system and user are empty or contain only placeholders, return fallback."""
        # Case 1: Completely empty
        generation = {
            "system_message_text": "",
            "user_message_text": "",
        }
        system_text = build_responses_system_input(generation)
        assert system_text == "You are a helpful assistant."

        # Case 2: Only placeholders in user template
        generation = {
            "system_message_text": "",
            "user_message_text": "{reference_documents}\n{question}",
        }
        system_text = build_responses_system_input(generation)
        assert system_text == "You are a helpful assistant."

        # Case 4: Unresolved question slot only (cookbook-style minimal user template)
        generation = {
            "system_message_text": "",
            "user_message_text": "{question}",
        }
        system_text = build_responses_system_input(generation)
        assert system_text == "You are a helpful assistant."

        # Case 3: Only OGX-duplicative content that gets stripped
        generation = {
            "system_message_text": "Answer using ONLY the provided documents.",
            "user_message_text": (
                "Answer ONLY using information from the documents below.\n"
                "You MUST cite sources using [1], [2].\n"
                "{reference_documents}\n{question}"
            ),
        }
        system_text = build_responses_system_input(generation)
        assert system_text == "You are a helpful assistant."

    def test_strip_ogx_runtime_partial_sentence_removal(self):
        """OGX sentences in a multi-sentence system prompt are removed; others are kept."""
        generation = {
            "system_message_text": (
                "You are an expert assistant. " "Answer using ONLY the provided documents. " "Be concise."
            ),
            "user_message_text": "",
        }
        result = build_responses_system_input(generation)
        assert "You are an expert assistant" in result
        assert "Be concise" in result
        assert "provided documents" not in result.lower()

    def test_user_grounding_merges_when_system_is_persona_only(self):
        """Persona-only system must not suppress non-OGX user supplements (e.g. RAG block)."""
        generation = {
            "system_message_text": "You are a retrieval-augmented assistant. Use your best judgment.",
            "user_message_text": (
                "You are a specialized Retrieval Augmented Generation (RAG) assistant. "
                "Prioritize correctness and ensure your response is grounded in the documents.\n"
                "{reference_documents}\n{question}"
            ),
        }
        result = build_responses_system_input(generation)
        assert "retrieval-augmented assistant" in result
        assert "specialized Retrieval Augmented Generation" in result

    def test_extract_static_user_pure_text_no_slots(self):
        """Templates without {reference_documents} are invalid and return empty user text."""
        generation = {
            "system_message_text": "Short system.",
            "user_message_text": "Always respond in a formal tone.",
        }
        result = build_responses_system_input(generation)
        # Invalid template (no {reference_documents}) → system only
        assert result == "Short system."

    def test_normalize_answer_scaffold_strips_with_citations(self):
        """Answer scaffolds must not retain citation hints owned by OGX."""
        assert normalize_answer_scaffold("Answer (max 150 words, with citations):") == "Answer (max 150 words):"

    def test_build_pattern_json_requires_generation_model_id(self):
        """Malformed generation payloads must raise KeyError for required fields."""
        pattern = _make_pattern()
        del pattern["settings"]["generation"]["model_id"]
        with pytest.raises(KeyError):
            build_pattern_json(pattern)

    def test_system_grounding_detection_requires_explicit_policy(self):
        """Grounding detection must require explicit 'ONLY' constraint, not just persona."""
        generation_persona_only = {
            "system_message_text": "You are a retrieval-augmented assistant. Use your best judgment.",
            "user_message_text": "Answer ONLY using information from the documents below.\n{reference_documents}\n{question}",
        }
        result_persona = build_responses_system_input(generation_persona_only)
        # Persona-only system should NOT trigger grounding suppression
        assert "retrieval-augmented assistant" in result_persona
        assert "use your best judgment" in result_persona.lower()

        generation_explicit = {
            "system_message_text": "Answer using ONLY the provided documents.",
            "user_message_text": "Answer ONLY using information from the documents below.\n{reference_documents}\n{question}",
        }
        result_explicit = build_responses_system_input(generation_explicit)
        # Explicit grounding system SHOULD suppress redundant user grounding
        # Both prompts are OGX-duplicative, so fallback is used
        assert result_explicit == "You are a helpful assistant."

    def test_system_grounding_detection_uses_grounding_prefixes(self):
        """Grounding detection must cover all ``_GROUNDING_PREFIXES`` entries."""
        generation = {
            "system_message_text": "Answer ONLY using information from documents retrieved via file search.",
            "user_message_text": (
                "Answer ONLY using information from the documents below.\n{reference_documents}\n{question}"
            ),
        }
        result = build_responses_system_input(generation)
        assert result == "You are a helpful assistant."

    def test_preserves_existing_pattern_fields(self):
        """Existing pattern fields (name, chunking, embedding, etc.) must not be altered."""
        pattern = _make_pattern()
        original_name = pattern["name"]
        original_chunking = copy.deepcopy(pattern["settings"]["chunking"])

        build_pattern_json(pattern)

        assert pattern["name"] == original_name
        assert pattern["settings"]["chunking"] == original_chunking

    def test_omits_temperature_when_none(self):
        """Temperature field must be omitted when None to avoid sending null to API."""
        pattern = _make_pattern()
        pattern["settings"]["generation"]["temperature"] = None

        build_pattern_json(pattern)

        assert "temperature" not in pattern["settings"]["responses_template"]
        # max_output_tokens should still be present
        assert "max_output_tokens" in pattern["settings"]["responses_template"]

    def test_omits_max_output_tokens_when_none(self):
        """max_output_tokens field must be omitted when None to avoid sending null to API."""
        pattern = _make_pattern()
        pattern["settings"]["generation"]["max_completion_tokens"] = None

        build_pattern_json(pattern)

        assert "max_output_tokens" not in pattern["settings"]["responses_template"]
        # temperature should still be present
        assert "temperature" in pattern["settings"]["responses_template"]

    def test_system_grounding_detection_no_false_positive_on_embedded_substring(self):
        """Grounding detection must not match embedded substrings, only sentence prefixes."""
        generation = {
            "system_message_text": "Use only relevant information. All documents do not contain PII.",
            "user_message_text": "Answer ONLY using information from the documents below.\n{reference_documents}\n{question}",
        }
        result = build_responses_system_input(generation)

        # "documents do not contain" is in _GROUNDING_PREFIXES but appears mid-sentence
        # Should NOT suppress user grounding since system doesn't start with a grounding prefix
        assert "Use only relevant information" in result
        assert "All documents do not contain PII" in result
