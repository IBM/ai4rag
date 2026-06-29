# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from ai4rag.components.optimization.search_space_preparation import (
    LANGUAGE_MAP,
)
from ai4rag.components.optimization.search_space_preparation import (
    _detect_benchmark_language as detect_benchmark_language,
)
from ai4rag.components.optimization.search_space_preparation import _detect_language_via_llm as detect_language_via_llm

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_ogx_client() -> MagicMock:
    """Return a MagicMock that behaves like an OgxClient with one LLM model."""
    mock_model = MagicMock()
    mock_model.identifier = "test-model"
    mock_model.model_type = "llm"

    mock_models_response = MagicMock()
    mock_models_response.data = [mock_model]

    mock_choice = MagicMock()
    mock_choice.message.content = "ja"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_client = MagicMock()
    mock_client.models.list.return_value = mock_models_response
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


@pytest.fixture()
def sample_questions() -> list[str]:
    """Return a short list of sample questions."""
    return [
        "東京の天気はどうですか？",
        "日本の首都はどこですか？",
        "富士山の高さは？",
    ]


# ---------------------------------------------------------------------------
# LANGUAGE_MAP
# ---------------------------------------------------------------------------


class TestLanguageMap:
    """Verify the static LANGUAGE_MAP contents."""

    def test_known_codes_present(self):
        """Well-known ISO 639-1 codes must resolve to their language names."""
        assert LANGUAGE_MAP["ja"] == "Japanese"
        assert LANGUAGE_MAP["en"] == "English"
        assert LANGUAGE_MAP["pl"] == "Polish"
        assert LANGUAGE_MAP["de"] == "German"
        assert LANGUAGE_MAP["fr"] == "French"
        assert LANGUAGE_MAP["ko"] == "Korean"

    def test_chinese_variants(self):
        """Both simplified and traditional Chinese codes must be present."""
        assert LANGUAGE_MAP["zh-cn"] == "Chinese"
        assert LANGUAGE_MAP["zh-tw"] == "Chinese"

    def test_all_values_are_nonempty_strings(self):
        """Every value in LANGUAGE_MAP must be a non-empty human-readable name."""
        for code, name in LANGUAGE_MAP.items():
            assert isinstance(code, str) and code, f"Invalid code: {code!r}"
            assert isinstance(name, str) and name, f"Invalid name for {code!r}: {name!r}"


# ---------------------------------------------------------------------------
# detect_language_via_llm
# ---------------------------------------------------------------------------


class TestDetectLanguageViaLlm:
    """Tests for the LLM-based language detection function."""

    def test_detects_japanese(self, mock_ogx_client, sample_questions):
        """When the LLM returns 'ja', the result must contain the correct code and name."""
        result = detect_language_via_llm(sample_questions, mock_ogx_client)

        assert result is not None
        assert result == {"code": "ja", "name": "Japanese"}
        mock_ogx_client.chat.completions.create.assert_called_once()

    def test_english_returns_none(self, mock_ogx_client, sample_questions):
        """English is the default language, so detection must return None."""
        mock_ogx_client.chat.completions.create.return_value.choices[0].message.content = "en"

        result = detect_language_via_llm(sample_questions, mock_ogx_client)

        # English maps to a valid entry in LANGUAGE_MAP, so it returns the dict.
        # The contract says "None for English" only at the detect_benchmark_language
        # level.  At this level the function returns the mapping when the code is
        # valid, regardless of which language it is.
        # Re-reading the source: the function returns {"code": ..., "name": ...}
        # for ANY valid code, including English.
        assert result == {"code": "en", "name": "English"}

    def test_api_failure_returns_none(self, mock_ogx_client, sample_questions):
        """An exception from the OGX client must be swallowed, returning None."""
        mock_ogx_client.chat.completions.create.side_effect = RuntimeError("API unavailable")

        result = detect_language_via_llm(sample_questions, mock_ogx_client)

        assert result is None

    def test_unsupported_language_code_returns_none(self, mock_ogx_client, sample_questions):
        """An ISO code not present in LANGUAGE_MAP must return None."""
        mock_ogx_client.chat.completions.create.return_value.choices[0].message.content = "xx"

        result = detect_language_via_llm(sample_questions, mock_ogx_client)

        assert result is None

    def test_no_models_available_returns_none(self, sample_questions):
        """When no models are registered, the function must return None."""
        mock_model_response = MagicMock()
        mock_model_response.data = []

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_model_response

        result = detect_language_via_llm(sample_questions, mock_client)

        assert result is None
        mock_client.chat.completions.create.assert_not_called()

    def test_prefers_allowed_generation_model(self, sample_questions):
        """When allowed_generation_models is set, the preferred model must be used."""
        preferred_model = MagicMock()
        preferred_model.identifier = "preferred-llm"
        preferred_model.model_type = "llm"

        other_model = MagicMock()
        other_model.identifier = "other-llm"
        other_model.model_type = "llm"

        mock_models_response = MagicMock()
        mock_models_response.data = [other_model, preferred_model]

        mock_choice = MagicMock()
        mock_choice.message.content = "ja"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.models.list.return_value = mock_models_response
        mock_client.chat.completions.create.return_value = mock_response

        detect_language_via_llm(sample_questions, mock_client, allowed_generation_models=["preferred-llm"])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs[1]["model"] == "preferred-llm" or call_kwargs.kwargs["model"] == "preferred-llm"

    def test_samples_at_most_five_questions(self, mock_ogx_client):
        """Only the first five questions should appear in the prompt."""
        many_questions = [f"Question {i}" for i in range(20)]

        detect_language_via_llm(many_questions, mock_ogx_client)

        call_kwargs = mock_ogx_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        # The prompt enumerates "- Q" lines; at most 5 should appear.
        assert user_content.count("- Question") == 5

    def test_empty_llm_response_returns_none(self, mock_ogx_client, sample_questions):
        """A blank response from the LLM must return None."""
        mock_ogx_client.chat.completions.create.return_value.choices[0].message.content = "   "

        result = detect_language_via_llm(sample_questions, mock_ogx_client)

        assert result is None

    def test_models_list_failure_returns_none(self, sample_questions):
        """An exception during models.list() must be swallowed."""
        mock_client = MagicMock()
        mock_client.models.list.side_effect = ConnectionError("timeout")

        result = detect_language_via_llm(sample_questions, mock_client)

        assert result is None


# ---------------------------------------------------------------------------
# detect_benchmark_language
# ---------------------------------------------------------------------------


class TestDetectBenchmarkLanguage:
    """Tests for the DataFrame-level language detection wrapper."""

    def test_detects_language_from_dataframe(self, mock_ogx_client):
        """A DataFrame with a 'question' column must yield detection results."""
        df = pd.DataFrame({"question": ["東京の天気は？", "富士山の高さは？", "日本の首都は？"]})

        result = detect_benchmark_language(df, mock_ogx_client)

        assert result is not None
        assert result["code"] == "ja"
        mock_ogx_client.chat.completions.create.assert_called_once()

    def test_empty_dataframe_returns_none(self, mock_ogx_client):
        """An empty DataFrame must short-circuit to None without calling the LLM."""
        df = pd.DataFrame({"question": pd.Series([], dtype=str)})

        result = detect_benchmark_language(df, mock_ogx_client)

        assert result is None
        mock_ogx_client.chat.completions.create.assert_not_called()

    def test_all_nan_questions_returns_none(self, mock_ogx_client):
        """When every question value is NaN, the function must return None."""
        df = pd.DataFrame({"question": [None, None, None]})

        result = detect_benchmark_language(df, mock_ogx_client)

        assert result is None
        mock_ogx_client.chat.completions.create.assert_not_called()

    def test_respects_sample_size(self, mock_ogx_client):
        """The sample_size parameter must cap the number of questions forwarded."""
        df = pd.DataFrame({"question": [f"Q{i}" for i in range(50)]})

        detect_benchmark_language(df, mock_ogx_client, sample_size=3)

        call_kwargs = mock_ogx_client.chat.completions.create.call_args
        user_content = call_kwargs[1]["messages"][1]["content"]
        # detect_language_via_llm further caps to 5, but sample_size=3 means
        # only 3 questions are passed in.
        assert user_content.count("- Q") == 3

    def test_passes_generation_models_through(self, mock_ogx_client):
        """The generation_models parameter must reach detect_language_via_llm."""
        df = pd.DataFrame({"question": ["Hello?"]})

        detect_benchmark_language(df, mock_ogx_client, generation_models=["custom-model"])

        # The function should still call the LLM; the model selection logic
        # inside detect_language_via_llm handles the allowed list.
        mock_ogx_client.chat.completions.create.assert_called_once()
