# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ai4rag.components.optimization.rag_templates_optimization import (
    DEFAULT_MAX_RAG_PATTERNS,
    MIN_MAX_RAG_PATTERNS_RANGE,
    SUPPORTED_OPTIMIZATION_METRICS,
    _inject_language_instructions,
    _validate_optimization_settings,
    run_rag_optimization,
)
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ogx_client() -> MagicMock:
    """Return a bare MagicMock standing in for OgxClient."""
    return MagicMock()


@pytest.fixture()
def foundation_model_stub() -> MagicMock:
    """Return a MagicMock that acts as a foundation model with message attributes."""
    fm = MagicMock()
    fm.system_message_text = "You are a helpful assistant."
    fm.user_message_text = "Answer: {question}"
    return fm


# ---------------------------------------------------------------------------
# _validate_optimization_settings
# ---------------------------------------------------------------------------


class TestValidateOptimizationSettings:
    """Tests for the _validate_optimization_settings helper."""

    def test_none_returns_empty_dict(self):
        """None input (no settings provided) must return an empty dict."""
        assert _validate_optimization_settings(None) == {}

    def test_non_dict_raises_type_error(self):
        """A non-dict value must raise TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            _validate_optimization_settings("not-a-dict")  # type: ignore[arg-type]

    def test_list_raises_type_error(self):
        """A list (common mis-use) must raise TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            _validate_optimization_settings([1, 2, 3])  # type: ignore[arg-type]

    def test_valid_settings_with_int_max_patterns(self):
        """A valid dict with an integer max_number_of_rag_patterns must pass."""
        settings = {"max_number_of_rag_patterns": 10}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_string_max_patterns_parsed_to_int(self):
        """The pipeline UI sends strings; they must be parsed without error."""
        settings = {"max_number_of_rag_patterns": "12"}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_string_with_whitespace_parsed(self):
        """Whitespace around the string value must be tolerated."""
        settings = {"max_number_of_rag_patterns": "  8  "}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_invalid_string_raises_value_error(self):
        """A non-numeric string must raise ValueError."""
        with pytest.raises(ValueError, match="valid integer"):
            _validate_optimization_settings({"max_number_of_rag_patterns": "abc"})

    def test_below_range_raises_value_error(self):
        """A value below the minimum allowed range must raise ValueError."""
        below_min = MIN_MAX_RAG_PATTERNS_RANGE[0] - 1
        with pytest.raises(ValueError, match="must be in range"):
            _validate_optimization_settings({"max_number_of_rag_patterns": below_min})

    def test_above_range_raises_value_error(self):
        """A value above the maximum allowed range must raise ValueError."""
        above_max = MIN_MAX_RAG_PATTERNS_RANGE[1] + 1
        with pytest.raises(ValueError, match="must be in range"):
            _validate_optimization_settings({"max_number_of_rag_patterns": above_max})

    def test_boundary_min_accepted(self):
        """The exact minimum boundary must be accepted."""
        settings = {"max_number_of_rag_patterns": MIN_MAX_RAG_PATTERNS_RANGE[0]}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_boundary_max_accepted(self):
        """The exact maximum boundary must be accepted."""
        settings = {"max_number_of_rag_patterns": MIN_MAX_RAG_PATTERNS_RANGE[1]}
        result = _validate_optimization_settings(settings)
        assert result is settings

    def test_default_max_patterns_within_range(self):
        """The DEFAULT_MAX_RAG_PATTERNS constant must fall inside the allowed range."""
        lo, hi = MIN_MAX_RAG_PATTERNS_RANGE
        assert lo <= DEFAULT_MAX_RAG_PATTERNS <= hi

    def test_empty_dict_passes(self):
        """An empty dict (no overrides) must be accepted and returned."""
        result = _validate_optimization_settings({})
        assert result == {}

    def test_float_raises_type_error(self):
        """A float value must raise TypeError (after range check, it is not int)."""
        with pytest.raises(TypeError, match="must be an integer"):
            _validate_optimization_settings({"max_number_of_rag_patterns": 8.5})

    def test_extra_keys_preserved(self):
        """Settings with additional keys besides max_number_of_rag_patterns must pass through."""
        settings = {"max_number_of_rag_patterns": 10, "metric": "faithfulness"}
        result = _validate_optimization_settings(settings)
        assert result["metric"] == "faithfulness"


# ---------------------------------------------------------------------------
# _inject_language_instructions
# ---------------------------------------------------------------------------


class TestInjectLanguageInstructions:
    """Tests for injecting language-specific instructions into foundation models."""

    @staticmethod
    def _make_search_space(foundation_models: list) -> AI4RAGSearchSpace:
        """Build a minimal AI4RAGSearchSpace with required parameters."""
        dummy_embedding = MagicMock()
        dummy_embedding.context_length = 8192
        return AI4RAGSearchSpace(
            params=[
                Parameter("foundation_model", "C", values=foundation_models),
                Parameter("embedding_model", "C", values=[dummy_embedding]),
            ]
        )

    def test_injects_language_into_system_message(self, foundation_model_stub):
        """The system message must be prefixed with the language instruction."""
        search_space = self._make_search_space([foundation_model_stub])

        _inject_language_instructions(search_space, {"code": "ja", "name": "Japanese"})

        fm = search_space["foundation_model"].values[0]
        assert fm.system_message_text.startswith("You MUST respond in Japanese.")

    def test_injects_language_into_user_message(self, foundation_model_stub):
        """The user message must be appended with the language instruction."""
        search_space = self._make_search_space([foundation_model_stub])

        _inject_language_instructions(search_space, {"code": "ja", "name": "Japanese"})

        fm = search_space["foundation_model"].values[0]
        assert fm.user_message_text.endswith("You MUST respond in Japanese.")

    def test_english_code_skips_injection(self, foundation_model_stub):
        """English language detection must not modify the messages."""
        original_sys = foundation_model_stub.system_message_text
        original_usr = foundation_model_stub.user_message_text

        search_space = self._make_search_space([foundation_model_stub])

        _inject_language_instructions(search_space, {"code": "en", "name": "English"})

        fm = search_space["foundation_model"].values[0]
        assert fm.system_message_text == original_sys
        assert fm.user_message_text == original_usr

    def test_empty_name_skips_injection(self, foundation_model_stub):
        """An empty language name must not inject any instruction."""
        original_sys = foundation_model_stub.system_message_text

        search_space = self._make_search_space([foundation_model_stub])

        _inject_language_instructions(search_space, {"code": "ja", "name": ""})

        fm = search_space["foundation_model"].values[0]
        assert fm.system_message_text == original_sys

    def test_multiple_foundation_models(self):
        """All foundation models in the search space must be updated."""
        fm1 = MagicMock()
        fm1.system_message_text = "sys1"
        fm1.user_message_text = "usr1"

        fm2 = MagicMock()
        fm2.system_message_text = "sys2"
        fm2.user_message_text = "usr2"

        search_space = self._make_search_space([fm1, fm2])

        _inject_language_instructions(search_space, {"code": "ko", "name": "Korean"})

        for fm in search_space["foundation_model"].values:
            assert "You MUST respond in Korean." in fm.system_message_text
            assert "You MUST respond in Korean." in fm.user_message_text

    def test_none_system_message_gets_instruction_prepended(self):
        """A model with None system_message_text must still get the language instruction."""
        fm = MagicMock()
        fm.system_message_text = None
        fm.user_message_text = None

        search_space = self._make_search_space([fm])

        _inject_language_instructions(search_space, {"code": "de", "name": "German"})

        assert fm.system_message_text.startswith("You MUST respond in German.")


# ---------------------------------------------------------------------------
# run_rag_optimization -- input validation only
# ---------------------------------------------------------------------------


class TestRunRagOptimizationValidation:
    """Test input validation in run_rag_optimization.

    These tests verify that the function rejects bad inputs before
    reaching any heavy I/O or OGX calls.
    """

    def test_empty_vector_io_provider_id_raises(self, mock_ogx_client):
        """An empty vector_io_provider_id must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="",
                test_data_key="data.json",
            )

    def test_whitespace_vector_io_provider_id_raises(self, mock_ogx_client):
        """A whitespace-only vector_io_provider_id must raise ValueError."""
        with pytest.raises(ValueError, match="non-empty string"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="   ",
                test_data_key="data.json",
            )

    def test_test_data_key_not_json_raises(self, mock_ogx_client):
        """A test_data_key not ending in .json must raise ValueError."""
        with pytest.raises(ValueError, match="JSON file"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="provider-1",
                test_data_key="data.csv",
            )

    def test_empty_test_data_key_raises(self, mock_ogx_client):
        """An empty test_data_key must raise ValueError."""
        with pytest.raises(ValueError, match="JSON file"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="provider-1",
                test_data_key="",
            )

    def test_invalid_optimization_metric_raises(self, mock_ogx_client):
        """An unsupported metric in optimization_settings must raise ValueError."""
        with pytest.raises(ValueError, match="is not supported"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="provider-1",
                test_data_key="bench.json",
                optimization_settings={"metric": "nonexistent_metric"},
            )

    def test_supported_optimization_metrics_constant(self):
        """SUPPORTED_OPTIMIZATION_METRICS must contain the three canonical metrics."""
        assert "faithfulness" in SUPPORTED_OPTIMIZATION_METRICS
        assert "answer_correctness" in SUPPORTED_OPTIMIZATION_METRICS
        assert "context_correctness" in SUPPORTED_OPTIMIZATION_METRICS

    def test_invalid_optimization_settings_type_raises(self, mock_ogx_client):
        """Non-dict optimization_settings must raise TypeError."""
        with pytest.raises(TypeError, match="must be a dictionary"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="provider-1",
                test_data_key="bench.json",
                optimization_settings="bad",  # type: ignore[arg-type]
            )

    def test_out_of_range_max_patterns_raises(self, mock_ogx_client):
        """max_number_of_rag_patterns outside the allowed range must raise ValueError."""
        with pytest.raises(ValueError, match="must be in range"):
            run_rag_optimization(
                extracted_text_path="dummy",
                test_data_path="dummy.json",
                search_space_report_path="dummy.yaml",
                output_dir="out",
                ogx_client=mock_ogx_client,
                vector_io_provider_id="provider-1",
                test_data_key="bench.json",
                optimization_settings={"max_number_of_rag_patterns": 50},
            )
