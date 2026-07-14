# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from ai4rag.evaluator.base_evaluator import (
    AggregateMetric,
    ConfidenceInterval,
    EvaluationMetricsResult,
    QuestionMetric,
    QuestionScore,
)
from ai4rag.evaluator.judge_selection import (
    _ordered_question_scores,
    _score_judge_candidates,
    _spread_and_stability_score,
    calibration_subset_size,
    select_judge_model,
)
from ai4rag.evaluator.metric import Metrics

# ---------------------------------------------------------------------------
# calibration_subset_size
# ---------------------------------------------------------------------------


class TestCalibrationSubsetSize:
    def test_small_dataset(self):
        assert calibration_subset_size(5) == 1

    def test_medium_dataset(self):
        assert calibration_subset_size(100) == 10

    def test_large_dataset_capped_at_20(self):
        assert calibration_subset_size(500) == 20

    def test_zero_rows(self):
        assert calibration_subset_size(0) == 0

    def test_negative_rows(self):
        assert calibration_subset_size(-10) == 0

    def test_one_row(self):
        assert calibration_subset_size(1) == 1


# ---------------------------------------------------------------------------
# _spread_and_stability_score
# ---------------------------------------------------------------------------


class TestSpreadAndStabilityScore:
    def test_all_valid_scores(self):
        score = _spread_and_stability_score([0.0, 0.5, 1.0])
        assert score > 0.0

    def test_all_identical_scores_zero_spread(self):
        score = _spread_and_stability_score([0.5, 0.5, 0.5])
        assert score == 0.0

    def test_fewer_than_two_valid_returns_negative(self):
        assert _spread_and_stability_score([0.5]) == -1.0
        assert _spread_and_stability_score([]) == -1.0

    def test_all_none_returns_negative(self):
        assert _spread_and_stability_score([None, None, None]) == -1.0

    def test_some_none_reduces_stability(self):
        all_valid = _spread_and_stability_score([0.0, 0.5, 1.0])
        with_failures = _spread_and_stability_score([0.0, 0.5, 1.0, None, None])
        assert with_failures < all_valid

    def test_two_valid_scores(self):
        score = _spread_and_stability_score([0.0, 1.0])
        assert score > 0.0

    def test_one_none_one_valid_returns_negative(self):
        assert _spread_and_stability_score([0.5, None]) == -1.0


# ---------------------------------------------------------------------------
# _ordered_question_scores
# ---------------------------------------------------------------------------


def _make_eval_result(scores_by_qid: dict[str, float | None]) -> EvaluationMetricsResult:
    """Build a minimal EvaluationMetricsResult for testing."""
    metric_name = Metrics.JUDGE_ANSWER_RELEVANCE.name
    return EvaluationMetricsResult(
        metrics=[
            AggregateMetric(
                name=metric_name,
                evaluator="judge",
                description="",
                scores=ConfidenceInterval(mean=0.5, ci_low=None, ci_high=None),
            )
        ],
        question_scores=[
            QuestionScore(
                question_id=qid,
                metrics=[QuestionMetric(name=metric_name, evaluator="judge", value=val)] if val is not None else [],
            )
            for qid, val in scores_by_qid.items()
        ],
    )


class TestOrderedQuestionScores:
    def test_scores_sorted_by_question_id(self):
        result = _make_eval_result({"q3": 0.3, "q1": 0.1, "q2": 0.2})
        scores = _ordered_question_scores(result, Metrics.JUDGE_ANSWER_RELEVANCE.name)
        assert scores == [0.1, 0.2, 0.3]

    def test_missing_metric_returns_none(self):
        result = _make_eval_result({"q1": None})
        scores = _ordered_question_scores(result, Metrics.JUDGE_ANSWER_RELEVANCE.name)
        assert scores == [None]

    def test_empty_result(self):
        result = EvaluationMetricsResult(metrics=[], question_scores=[])
        scores = _ordered_question_scores(result, Metrics.JUDGE_ANSWER_RELEVANCE.name)
        assert scores == []

    def test_wrong_metric_name_returns_none_entries(self):
        result = _make_eval_result({"q1": 0.5})
        scores = _ordered_question_scores(result, "nonexistent_metric")
        assert scores == [None]


# ---------------------------------------------------------------------------
# _score_judge_candidates
# ---------------------------------------------------------------------------


def _make_model(model_id: str, chat_scores: list[int]) -> MagicMock:
    model = MagicMock()
    type(model).model_id = PropertyMock(return_value=model_id)
    model.chat.side_effect = [[_make_chat_choice(s)] for s in chat_scores]
    return model


def _make_chat_choice(score: int) -> MagicMock:
    choice = MagicMock()
    choice.message.content = json.dumps({"score": score, "rationale": "OK"})
    return choice


def _make_eval_data(n: int = 3) -> list:
    from ai4rag.evaluator.base_evaluator import EvaluationData

    return [
        EvaluationData(
            question=f"Q{i}?",
            answer=f"A{i}.",
            contexts=[f"C{i}"],
            context_ids=[f"d{i}"],
            ground_truths=[f"A{i}."],
            ground_truths_context_ids=[f"d{i}"],
            question_id=f"q{i}",
        )
        for i in range(n)
    ]


class TestScoreJudgeCandidates:
    def test_candidates_ranked_by_score_descending(self):
        model_a = _make_model("model-a", [3, 3, 3])
        model_b = _make_model("model-b", [1, 3, 5])
        eval_data = _make_eval_data(3)

        rankings = _score_judge_candidates(
            candidates=[model_a, model_b],
            eval_data=eval_data,
            reference_model_id="model-a",
        )

        assert rankings[0]["model"].model_id == "model-b"
        assert rankings[0]["score"] > rankings[1]["score"]

    def test_reference_model_preferred_on_tie(self):
        model_a = _make_model("model-a", [1, 5])
        model_b = _make_model("model-b", [1, 5])
        eval_data = _make_eval_data(2)

        rankings = _score_judge_candidates(
            candidates=[model_a, model_b],
            eval_data=eval_data,
            reference_model_id="model-a",
        )

        assert rankings[0]["model"].model_id == "model-a"


# ---------------------------------------------------------------------------
# select_judge_model
# ---------------------------------------------------------------------------


class TestSelectJudgeModel:
    def test_empty_models_raises(self):
        with pytest.raises(ValueError, match="At least one"):
            select_judge_model(
                generation_models=[],
                embedding_models=[],
                benchmark_data=MagicMock(),
                documents=[],
            )

    def test_single_model_returned_directly(self):
        model = MagicMock()
        result = select_judge_model(
            generation_models=[model],
            embedding_models=[MagicMock()],
            benchmark_data=MagicMock(),
            documents=[],
        )
        assert result is model

    @patch("ai4rag.evaluator.judge_selection._run_reference_rag")
    @patch("ai4rag.evaluator.judge_selection._score_judge_candidates")
    def test_multi_model_runs_calibration(self, mock_score, mock_rag):
        model_a = MagicMock()
        model_a.model_id = "model-a"
        model_b = MagicMock()
        model_b.model_id = "model-b"

        mock_rag.return_value = _make_eval_data(2)
        mock_score.return_value = [
            {"model": model_b, "score": 0.5},
            {"model": model_a, "score": 0.1},
        ]

        benchmark = MagicMock()
        benchmark.questions = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]
        benchmark.get_random_sample.return_value = benchmark

        result = select_judge_model(
            generation_models=[model_a, model_b],
            embedding_models=[MagicMock()],
            benchmark_data=benchmark,
            documents=[],
        )

        assert result is model_b
        mock_score.assert_called_once()
        mock_rag.assert_called_once()
