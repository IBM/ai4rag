# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, PropertyMock

import pytest

from ai4rag import logger
from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData
from ai4rag.evaluator.llmaj_evaluator import (
    JUDGE_RESPONSE_FORMAT,
    LLMaJEvaluator,
    _normalize_score,
    _parse_judge_response,
)
from ai4rag.evaluator.metric import Metrics


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


def _make_chat_choice(score: int) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps({"score": score, "rationale": "OK"})
    return choice


def _make_judge_model() -> MagicMock:
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value="judge-model")
    return model


class TestLLMaJEvaluator:
    def test_supported_metrics(self):
        evaluator = LLMaJEvaluator(model=_make_judge_model())
        supported = evaluator.get_supported_metrics()
        assert supported == [Metrics.JUDGE_ANSWER_RELEVANCE.name]

    def test_evaluate_metrics(self, sample_evaluation_data):
        model = _make_judge_model()
        model.chat.side_effect = [
            [_make_chat_choice(5)],
            [_make_chat_choice(4)],
        ]

        evaluator = LLMaJEvaluator(model=model)
        result = evaluator.evaluate_metrics(sample_evaluation_data, [Metrics.JUDGE_ANSWER_RELEVANCE])

        assert len(result["metrics"]) == 1
        agg = result["metrics"][0]
        assert agg["name"] == Metrics.JUDGE_ANSWER_RELEVANCE.name
        assert agg["evaluator"] == "judge"
        assert agg["scores"]["mean"] == 0.875

        q_scores = {qs["question_id"]: qs for qs in result["question_scores"]}
        q1_val = q_scores["q1"]["metrics"][0]["value"]
        q2_val = q_scores["q2"]["metrics"][0]["value"]
        assert q1_val == 1.0
        assert q2_val == 0.75

    def test_structured_output_format_passed(self):
        model = _make_judge_model()
        model.chat.return_value = [_make_chat_choice(3)]

        evaluator = LLMaJEvaluator(model=model)
        ed = EvaluationData(question="Q?", answer="A.", question_id="q0")
        evaluator._judge_row(ed, "Check relevance.")

        call_kwargs = model.chat.call_args.kwargs
        assert call_kwargs["response_format"] == JUDGE_RESPONSE_FORMAT

    def test_is_base_evaluator_subclass(self):
        assert issubclass(LLMaJEvaluator, BaseEvaluator)


class TestParseJudgeResponse:
    def test_valid_json(self):
        assert _parse_judge_response('{"score": 3, "rationale": "OK"}') == {"score": 3, "rationale": "OK"}

    def test_single_quotes(self):
        assert _parse_judge_response("{'score': 4, 'rationale': 'Good'}") == {"score": 4, "rationale": "Good"}

    def test_markdown_fenced(self):
        content = '```json\n{"score": 5, "rationale": "Great"}\n```'
        assert _parse_judge_response(content) == {"score": 5, "rationale": "Great"}

    def test_surrounding_prose(self):
        content = 'My evaluation: {"score": 2, "rationale": "Weak"} end.'
        assert _parse_judge_response(content) == {"score": 2, "rationale": "Weak"}

    def test_single_quotes_in_extracted_fragment(self):
        content = "Here is the result: {'score': 1, 'rationale': 'Bad'}"
        assert _parse_judge_response(content) == {"score": 1, "rationale": "Bad"}

    def test_irreparable_content_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_judge_response("completely unparseable gibberish")


class TestHelpers:
    def test_normalize_score(self):
        assert _normalize_score(1) == 0.0
        assert _normalize_score(5) == 1.0

    def test_judge_row_logs_warning_on_failure(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="ai4rag")

        model = _make_judge_model()
        model.chat.return_value = [_make_chat_choice(4)]
        model.chat.return_value[0].message.content = "not json"

        evaluator = LLMaJEvaluator(model=model)
        evaluation_data = EvaluationData(
            question="What is Python?",
            answer="A language.",
            question_id="q-log",
        )
        result = evaluator._judge_row(evaluation_data, "Check relevance.")

        assert result is None
        assert "LLM judge response is not valid JSON" in caplog.text
        assert "judge-model" in caplog.text
        assert "q-log" in caplog.text
        assert "not json" in caplog.text

    def test_judge_row_recovers_single_quoted_json(self):
        model = _make_judge_model()
        choice = MagicMock()
        choice.message.content = "{'score': 4, 'rationale': 'Good answer'}"
        model.chat.return_value = [choice]

        evaluator = LLMaJEvaluator(model=model)
        ed = EvaluationData(question="Q?", answer="A.", question_id="q0")
        result = evaluator._judge_row(ed, "Check relevance.")

        assert result == 0.75

    def test_judge_row_recovers_markdown_fenced_json(self):
        model = _make_judge_model()
        choice = MagicMock()
        choice.message.content = '```json\n{"score": 5, "rationale": "Perfect"}\n```'
        model.chat.return_value = [choice]

        evaluator = LLMaJEvaluator(model=model)
        ed = EvaluationData(question="Q?", answer="A.", question_id="q0")
        result = evaluator._judge_row(ed, "Check relevance.")

        assert result == 1.0

    def test_judge_row_recovers_json_with_surrounding_prose(self):
        model = _make_judge_model()
        choice = MagicMock()
        choice.message.content = 'Here is my evaluation: {"score": 2, "rationale": "Weak"} Hope that helps.'
        model.chat.return_value = [choice]

        evaluator = LLMaJEvaluator(model=model)
        ed = EvaluationData(question="Q?", answer="A.", question_id="q0")
        result = evaluator._judge_row(ed, "Check relevance.")

        assert result == 0.25

    def test_judge_row_logs_warning_on_call_failure(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="ai4rag")

        model = _make_judge_model()
        model.chat.side_effect = RuntimeError("connection refused")

        evaluator = LLMaJEvaluator(model=model)
        evaluation_data = EvaluationData(
            question="What is Python?",
            answer="A language.",
            question_id="q-err",
        )
        result = evaluator._judge_row(evaluation_data, "Check relevance.")

        assert result is None
        assert "LLM judge call failed" in caplog.text
        assert "q-err" in caplog.text

    def test_judge_row_logs_warning_on_out_of_range_score(self, caplog):
        import logging

        caplog.set_level(logging.WARNING, logger="ai4rag")

        model = _make_judge_model()
        model.chat.return_value = [_make_chat_choice(99)]

        evaluator = LLMaJEvaluator(model=model)
        evaluation_data = EvaluationData(
            question="What is Python?",
            answer="A language.",
            question_id="q-range",
        )
        result = evaluator._judge_row(evaluation_data, "Check relevance.")

        assert result is None
        assert "out-of-range score=99" in caplog.text
