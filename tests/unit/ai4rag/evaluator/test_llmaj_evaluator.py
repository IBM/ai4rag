# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, patch

import pytest

from ai4rag import logger
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData, MetricType
from ai4rag.evaluator.llmaj_evaluator import (
    LLMaJConfig,
    LLMaJEvaluator,
    _extract_json,
    _llmaj_log_io_enabled,
    _normalize_score,
    _parse_score,
)


@pytest.fixture
def sample_evaluation_data() -> list[EvaluationData]:
    return [
        EvaluationData(
            question="What is Python?",
            answer="Python is a programming language.",
            contexts=["Python is a high-level programming language."],
            context_ids=["doc1"],
            ground_truths=["Python is a programming language."],
            ground_truths_context_ids=["doc1"],
            question_id="q1",
        ),
        EvaluationData(
            question="What is AI?",
            answer="AI is Artificial Intelligence.",
            contexts=["AI stands for Artificial Intelligence."],
            context_ids=["doc2"],
            ground_truths=["AI is Artificial Intelligence."],
            ground_truths_context_ids=["doc2"],
            question_id="q2",
        ),
    ]


def _make_chat_response(score: int) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps({"score": score, "rationale": "OK"})
    return resp


@pytest.fixture
def llmaj_config() -> LLMaJConfig:
    return LLMaJConfig(
        base_url="https://ogx.example.com/v1",
        api_key="test-api-key",
        model="judge-model",
    )


class TestLLMaJConfig:
    def test_required_fields(self, llmaj_config):
        assert llmaj_config.base_url == "https://ogx.example.com/v1"
        assert llmaj_config.api_key == "test-api-key"
        assert llmaj_config.model == "judge-model"
        assert llmaj_config.temperature == 0.0

    def test_missing_required_fields_raise(self):
        with pytest.raises(TypeError):
            LLMaJConfig()  # type: ignore[call-arg]

        with pytest.raises(ValueError, match="base_url is required"):
            LLMaJConfig(base_url="", api_key="key", model="model")

        with pytest.raises(ValueError, match="api_key is required"):
            LLMaJConfig(base_url="https://ogx.example.com/v1", api_key="", model="model")

        with pytest.raises(ValueError, match="model is required"):
            LLMaJConfig(base_url="https://ogx.example.com/v1", api_key="key", model="")


class TestLLMaJEvaluator:
    @patch("ai4rag.evaluator.llmaj_evaluator.OpenAI")
    def test_supported_metrics(self, _mock_openai, llmaj_config):
        evaluator = LLMaJEvaluator(llmaj_config)
        supported = evaluator.get_supported_metrics()
        assert supported == [MetricType.ANSWER_RELEVANCE]

    @patch("ai4rag.evaluator.llmaj_evaluator.OpenAI")
    def test_evaluate_metrics(self, mock_openai_cls, sample_evaluation_data, llmaj_config):
        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.side_effect = [
            _make_chat_response(5),
            _make_chat_response(4),
        ]

        evaluator = LLMaJEvaluator(llmaj_config)
        result = evaluator.evaluate_metrics(sample_evaluation_data, [MetricType.ANSWER_RELEVANCE])

        assert result["scores"][MetricType.ANSWER_RELEVANCE]["mean"] == 0.875
        assert result["question_scores"][MetricType.ANSWER_RELEVANCE]["q1"] == 1.0
        assert result["question_scores"][MetricType.ANSWER_RELEVANCE]["q2"] == 0.75

    @patch("ai4rag.evaluator.llmaj_evaluator.OpenAI")
    def test_is_base_evaluator_subclass(self, _mock_openai):
        assert issubclass(LLMaJEvaluator, BaseEvaluator)


class TestParsingHelpers:
    def test_extract_json_plain(self):
        assert _extract_json('{"score": 4}') == {"score": 4}

    def test_extract_json_from_surrounding_text(self):
        assert _extract_json('Here is the result: {"score": 3, "rationale": "ok"} done') == {
            "score": 3,
            "rationale": "ok",
        }

    def test_parse_score_valid(self):
        assert _parse_score('{"score": 3}') == 3

    def test_normalize_score(self):
        assert _normalize_score(1) == 0.0
        assert _normalize_score(5) == 1.0

    def test_llmaj_log_io_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("AI4RAG_LLMAJ_LOG_IO", raising=False)
        assert _llmaj_log_io_enabled() is True

    def test_llmaj_log_io_can_be_disabled(self, monkeypatch):
        monkeypatch.setenv("AI4RAG_LLMAJ_LOG_IO", "0")
        assert _llmaj_log_io_enabled() is False

    @patch("ai4rag.evaluator.llmaj_evaluator.OpenAI")
    def test_judge_row_logs_prompt_and_response(self, mock_openai_cls, llmaj_config, caplog):
        import logging

        caplog.set_level(logging.INFO, logger="ai4rag")

        client = MagicMock()
        mock_openai_cls.return_value = client
        client.chat.completions.create.return_value = _make_chat_response(4)

        evaluator = LLMaJEvaluator(llmaj_config)
        evaluation_data = EvaluationData(
            question="What is Python?",
            answer="A language.",
            question_id="q-log",
        )
        result = evaluator._judge_row(evaluation_data, "Check relevance.")

        assert result == 0.75
        assert "LLM judge request" in caplog.text
        assert "--- PROMPT ---" in caplog.text
        assert "What is Python?" in caplog.text
        assert "LLM judge response" in caplog.text
        assert "--- RESPONSE ---" in caplog.text
