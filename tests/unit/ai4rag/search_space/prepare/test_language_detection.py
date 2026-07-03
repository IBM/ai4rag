# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from ai4rag.search_space.prepare.language_detection import (
    LANGUAGE_MAP,
    detect_language_with_llm,
)


@pytest.fixture()
def mock_generation_model() -> MagicMock:
    """Return a MagicMock that behaves like an OGXFoundationModel."""
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps({"code": "ja"})

    model = MagicMock()
    model.chat.return_value = [mock_choice]
    return model


@pytest.fixture()
def sample_questions() -> list[str]:
    return [
        "東京の天気はどうですか？",
        "日本の首都はどこですか？",
        "富士山の高さは？",
    ]


class TestLanguageMap:

    def test_known_codes_present(self):
        assert LANGUAGE_MAP["ja"] == "Japanese"
        assert LANGUAGE_MAP["en"] == "English"
        assert LANGUAGE_MAP["pl"] == "Polish"
        assert LANGUAGE_MAP["de"] == "German"
        assert LANGUAGE_MAP["fr"] == "French"
        assert LANGUAGE_MAP["ko"] == "Korean"

    def test_chinese_variants(self):
        assert LANGUAGE_MAP["zh-cn"] == "Chinese"
        assert LANGUAGE_MAP["zh-tw"] == "Chinese"
        assert LANGUAGE_MAP["zh"] == "Chinese"

    def test_all_values_are_nonempty_strings(self):
        for code, name in LANGUAGE_MAP.items():
            assert isinstance(code, str) and code, f"Invalid code: {code!r}"
            assert isinstance(name, str) and name, f"Invalid name for {code!r}: {name!r}"


class TestDetectLanguageWithLlm:

    def test_detects_japanese(self, mock_generation_model, sample_questions):
        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is not None
        assert result == {"code": "ja", "name": "Japanese"}
        mock_generation_model.chat.assert_called_once()

    def test_detects_english(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"code": "en"})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result == {"code": "en", "name": "English"}

    def test_api_failure_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.side_effect = RuntimeError("API unavailable")

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None

    def test_unsupported_language_code_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"code": "xx"})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None

    def test_samples_at_most_five_questions(self, mock_generation_model):
        many_questions = [f"Question {i}" for i in range(20)]

        detect_language_with_llm(many_questions, mock_generation_model)

        call_kwargs = mock_generation_model.chat.call_args
        user_content = call_kwargs.kwargs["messages"][1]["content"]
        assert user_content.count("- Question") == 5

    def test_malformed_json_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = "not json"

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None

    def test_passes_expected_chat_params(self, mock_generation_model, sample_questions):
        detect_language_with_llm(sample_questions, mock_generation_model)

        call_kwargs = mock_generation_model.chat.call_args.kwargs
        assert call_kwargs["max_completion_tokens"] == 15
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["strict"] is True

    def test_malformed_code_format_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"code": "123"})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None

    def test_missing_code_key_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"language": "en"})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None

    def test_extracts_code_with_region_suffix(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"code": "zh-cn"})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result == {"code": "zh", "name": "Chinese"}

    def test_normalises_uppercase_code(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"code": "FR"})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result == {"code": "fr", "name": "French"}

    def test_strips_whitespace_from_code(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = json.dumps({"code": " de "})

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result == {"code": "de", "name": "German"}

    def test_none_content_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value[0].message.content = None

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None

    def test_empty_response_list_returns_none(self, mock_generation_model, sample_questions):
        mock_generation_model.chat.return_value = []

        result = detect_language_with_llm(sample_questions, mock_generation_model)

        assert result is None
