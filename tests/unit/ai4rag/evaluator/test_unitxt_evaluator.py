# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from ai4rag.core.experiment.exception_handler import EvaluationError
from ai4rag.evaluator.base_evaluator import EvaluationData, MetricType
from ai4rag.evaluator.unitxt_evaluator import UnitxtEvaluator


@pytest.fixture
def sample_evaluation_data_list() -> list[EvaluationData]:
    """Fixture providing a list of EvaluationData instances."""
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


@pytest.fixture
def sample_scores_df() -> pd.DataFrame:
    """Fixture providing a sample scores DataFrame from unitxt."""
    return pd.DataFrame(
        {
            "question_id": ["q1", "q2"],
            "metrics.rag.external_rag.answer_correctness": [0.95, 0.87],
            "metrics.rag.external_rag.faithfulness": [0.92, 0.88],
            "metrics.rag.external_rag.context_correctness": [0.89, 0.91],
        }
    )


@pytest.fixture
def sample_ci_table() -> pd.DataFrame:
    """Fixture providing a sample confidence interval table from unitxt."""
    return pd.DataFrame(
        {
            "metrics.rag.external_rag.answer_correctness": {"score": 0.91, "score_ci_low": 0.85, "score_ci_high": 0.97},
            "metrics.rag.external_rag.faithfulness": {"score": 0.90, "score_ci_low": 0.84, "score_ci_high": 0.96},
            "metrics.rag.external_rag.context_correctness": {
                "score": 0.90,
                "score_ci_low": 0.86,
                "score_ci_high": 0.94,
            },
        }
    )


class TestUnitxtEvaluatorMetricMapping:
    """Test suite for UnitxtEvaluator metric type mapping."""

    def test_get_metric_types_single_metric(self):
        """Test get_metric_types with a single metric."""
        result = UnitxtEvaluator.get_metric_types([MetricType.ANSWER_CORRECTNESS])
        assert result == ["metrics.rag.external_rag.answer_correctness"]

    def test_get_metric_types_multiple_metrics(self):
        """Test get_metric_types with multiple metrics."""
        result = UnitxtEvaluator.get_metric_types(
            [MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS, MetricType.CONTEXT_CORRECTNESS]
        )
        assert len(result) == 3
        assert "metrics.rag.external_rag.answer_correctness" in result
        assert "metrics.rag.external_rag.faithfulness" in result
        assert "metrics.rag.external_rag.context_correctness" in result

    def test_get_metric_types_empty_list(self):
        """Test get_metric_types with empty list."""
        result = UnitxtEvaluator.get_metric_types([])
        assert result == []

    def test_get_metric_types_invalid_metric(self):
        """Test get_metric_types with invalid metric type."""
        result = UnitxtEvaluator.get_metric_types(["invalid_metric", MetricType.FAITHFULNESS])
        assert len(result) == 1
        assert result == ["metrics.rag.external_rag.faithfulness"]

    def test_get_metric_types_all_invalid(self):
        """Test get_metric_types with all invalid metrics."""
        result = UnitxtEvaluator.get_metric_types(["invalid1", "invalid2"])
        assert result == []

    def test_decode_unitxt_metric_single(self):
        """Test decode_unitxt_metric with a single metric."""
        result = UnitxtEvaluator.decode_unitxt_metric(["metrics.rag.external_rag.answer_correctness"])
        assert result == [MetricType.ANSWER_CORRECTNESS]

    def test_decode_unitxt_metric_multiple(self):
        """Test decode_unitxt_metric with multiple metrics."""
        unitxt_metrics = [
            "metrics.rag.external_rag.answer_correctness",
            "metrics.rag.external_rag.faithfulness",
            "metrics.rag.external_rag.context_correctness",
        ]
        result = UnitxtEvaluator.decode_unitxt_metric(unitxt_metrics)
        assert len(result) == 3
        assert MetricType.ANSWER_CORRECTNESS in result
        assert MetricType.FAITHFULNESS in result
        assert MetricType.CONTEXT_CORRECTNESS in result

    def test_decode_unitxt_metric_empty_list(self):
        """Test decode_unitxt_metric with empty list."""
        result = UnitxtEvaluator.decode_unitxt_metric([])
        assert result == []


class TestUnitxtEvaluatorHandleCICalculations:
    """Test suite for _handle_ci_calculations method."""

    def test_handle_ci_calculations_basic(self, sample_ci_table):
        """Test _handle_ci_calculations with basic input."""
        evaluator = UnitxtEvaluator()
        result = evaluator._handle_ci_calculations(sample_ci_table)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert MetricType.ANSWER_CORRECTNESS in result
        assert MetricType.FAITHFULNESS in result
        assert MetricType.CONTEXT_CORRECTNESS in result

    def test_handle_ci_calculations_structure(self, sample_ci_table):
        """Test that _handle_ci_calculations returns correct structure."""
        evaluator = UnitxtEvaluator()
        result = evaluator._handle_ci_calculations(sample_ci_table)

        for metric_key in result:
            assert "mean" in result[metric_key]
            assert "ci_low" in result[metric_key]
            assert "ci_high" in result[metric_key]

    def test_handle_ci_calculations_rounding(self, sample_ci_table):
        """Test that _handle_ci_calculations rounds values correctly."""
        evaluator = UnitxtEvaluator()
        result = evaluator._handle_ci_calculations(sample_ci_table)

        answer_correctness = result[MetricType.ANSWER_CORRECTNESS]
        assert answer_correctness["mean"] == 0.91
        assert answer_correctness["ci_low"] == 0.85
        assert answer_correctness["ci_high"] == 0.97

    def test_handle_ci_calculations_with_nan(self):
        """Test _handle_ci_calculations with NaN values."""
        ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {
                    "score": 0.91,
                    "score_ci_low": 0.85,
                    "score_ci_high": 0.97,
                },
                "metrics.rag.external_rag.faithfulness": {
                    "score": np.nan,
                    "score_ci_low": np.nan,
                    "score_ci_high": np.nan,
                },
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._handle_ci_calculations(ci_table)

        assert result[MetricType.ANSWER_CORRECTNESS]["mean"] == 0.91
        assert result[MetricType.FAITHFULNESS]["mean"] is None
        assert result[MetricType.FAITHFULNESS]["ci_low"] is None
        assert result[MetricType.FAITHFULNESS]["ci_high"] is None

    def test_handle_ci_calculations_missing_ci_columns(self):
        """Test _handle_ci_calculations when CI columns are missing."""
        ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {"score": 0.91},
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._handle_ci_calculations(ci_table)

        assert result[MetricType.ANSWER_CORRECTNESS]["mean"] == 0.91
        assert result[MetricType.ANSWER_CORRECTNESS]["ci_low"] is None
        assert result[MetricType.ANSWER_CORRECTNESS]["ci_high"] is None

    def test_handle_ci_calculations_precision(self):
        """Test that _handle_ci_calculations rounds to 4 decimal places."""
        ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {
                    "score": 0.123456789,
                    "score_ci_low": 0.987654321,
                    "score_ci_high": 0.555555555,
                },
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._handle_ci_calculations(ci_table)

        assert result[MetricType.ANSWER_CORRECTNESS]["mean"] == 0.1235
        assert result[MetricType.ANSWER_CORRECTNESS]["ci_low"] == 0.9877
        assert result[MetricType.ANSWER_CORRECTNESS]["ci_high"] == 0.5556


class TestUnitxtEvaluatorHandleQuestionsScores:
    """Test suite for _handle_questions_scores method."""

    def test_handle_questions_scores_basic(self, sample_scores_df):
        """Test _handle_questions_scores with basic input."""
        evaluator = UnitxtEvaluator()
        result = evaluator._handle_questions_scores(sample_scores_df)

        assert isinstance(result, dict)
        assert len(result) == 3
        assert MetricType.ANSWER_CORRECTNESS in result
        assert MetricType.FAITHFULNESS in result
        assert MetricType.CONTEXT_CORRECTNESS in result

    def test_handle_questions_scores_structure(self, sample_scores_df):
        """Test that _handle_questions_scores returns correct structure."""
        evaluator = UnitxtEvaluator()
        result = evaluator._handle_questions_scores(sample_scores_df)

        for metric_key in result:
            assert isinstance(result[metric_key], dict)
            assert "q1" in result[metric_key]
            assert "q2" in result[metric_key]

    def test_handle_questions_scores_values(self, sample_scores_df):
        """Test that _handle_questions_scores returns correct values."""
        evaluator = UnitxtEvaluator()
        result = evaluator._handle_questions_scores(sample_scores_df)

        assert result[MetricType.ANSWER_CORRECTNESS]["q1"] == 0.95
        assert result[MetricType.ANSWER_CORRECTNESS]["q2"] == 0.87
        assert result[MetricType.FAITHFULNESS]["q1"] == 0.92
        assert result[MetricType.FAITHFULNESS]["q2"] == 0.88

    def test_handle_questions_scores_rounding(self):
        """Test that _handle_questions_scores rounds to 4 decimal places."""
        scores_df = pd.DataFrame(
            {
                "question_id": ["q1"],
                "metrics.rag.external_rag.answer_correctness": [0.123456789],
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._handle_questions_scores(scores_df)

        assert result[MetricType.ANSWER_CORRECTNESS]["q1"] == 0.1235

    def test_handle_questions_scores_with_empty_strings(self):
        """Test _handle_questions_scores replaces empty strings with NaN."""
        scores_df = pd.DataFrame(
            {
                "question_id": ["q1", "q2"],
                "metrics.rag.external_rag.answer_correctness": [0.95, ""],
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._handle_questions_scores(scores_df)

        assert result[MetricType.ANSWER_CORRECTNESS]["q1"] == 0.95
        assert pd.isna(result[MetricType.ANSWER_CORRECTNESS]["q2"])

    def test_handle_questions_scores_filters_irrelevant_columns(self):
        """Test that _handle_questions_scores only includes relevant metrics."""
        scores_df = pd.DataFrame(
            {
                "question_id": ["q1"],
                "metrics.rag.external_rag.answer_correctness": [0.95],
                "some_other_column": [0.50],
                "another_column": ["text"],
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._handle_questions_scores(scores_df)

        assert len(result) == 1
        assert MetricType.ANSWER_CORRECTNESS in result
        assert "some_other_column" not in str(result)


class TestUnitxtEvaluatorEvaluateMetrics:
    """Test suite for evaluate_metrics method."""

    def test_evaluate_metrics_success(self, mocker, sample_evaluation_data_list):
        """Test successful evaluation with mocked unitxt.evaluate."""
        mock_scores_df = pd.DataFrame(
            {
                "question_id": ["q1", "q2"],
                "metrics.rag.external_rag.answer_correctness": [0.95, 0.87],
                "metrics.rag.external_rag.faithfulness": [0.92, 0.88],
            }
        )
        mock_ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {
                    "score": 0.91,
                    "score_ci_low": 0.85,
                    "score_ci_high": 0.97,
                },
                "metrics.rag.external_rag.faithfulness": {"score": 0.90, "score_ci_low": 0.84, "score_ci_high": 0.96},
            }
        )

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics(
            sample_evaluation_data_list,
            [MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS],
        )

        assert "scores" in result
        assert "question_scores" in result
        assert isinstance(result["scores"], dict)
        assert isinstance(result["question_scores"], dict)

        mock_evaluate.assert_called_once()
        call_args = mock_evaluate.call_args
        assert call_args[1]["compute_conf_intervals"] is True
        assert call_args[1]["metric_names"] == [
            "metrics.rag.external_rag.answer_correctness",
            "metrics.rag.external_rag.faithfulness",
        ]

    def test_evaluate_metrics_converts_evaluation_data_to_dict(self, mocker, sample_evaluation_data_list):
        """Test that evaluate_metrics converts EvaluationData to dictionaries."""
        mock_scores_df = pd.DataFrame({"question_id": ["q1"]})
        mock_ci_table = pd.DataFrame({"metrics.rag.external_rag.answer_correctness": {"score": 0.9}})

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        evaluator.evaluate_metrics(sample_evaluation_data_list, [MetricType.ANSWER_CORRECTNESS])

        call_args = mock_evaluate.call_args
        df_arg = call_args[0][0]
        assert isinstance(df_arg, pd.DataFrame)
        assert "question" in df_arg.columns
        assert "answer" in df_arg.columns

    def test_evaluate_metrics_raises_evaluation_error_on_exception(self, mocker, sample_evaluation_data_list):
        """Test that evaluate_metrics raises EvaluationError on exception."""
        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.side_effect = ValueError("Unitxt evaluation failed")

        evaluator = UnitxtEvaluator()

        with pytest.raises(EvaluationError):
            evaluator.evaluate_metrics(sample_evaluation_data_list, [MetricType.ANSWER_CORRECTNESS])

    def test_evaluate_metrics_with_single_metric(self, mocker, sample_evaluation_data_list):
        """Test evaluate_metrics with a single metric."""
        mock_scores_df = pd.DataFrame(
            {
                "question_id": ["q1", "q2"],
                "metrics.rag.external_rag.faithfulness": [0.92, 0.88],
            }
        )
        mock_ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.faithfulness": {"score": 0.90, "score_ci_low": 0.84, "score_ci_high": 0.96},
            }
        )

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics(sample_evaluation_data_list, [MetricType.FAITHFULNESS])

        assert MetricType.FAITHFULNESS in result["scores"]
        assert MetricType.FAITHFULNESS in result["question_scores"]

    def test_evaluate_metrics_with_all_metrics(self, mocker, sample_evaluation_data_list):
        """Test evaluate_metrics with all three metrics."""
        mock_scores_df = pd.DataFrame(
            {
                "question_id": ["q1", "q2"],
                "metrics.rag.external_rag.answer_correctness": [0.95, 0.87],
                "metrics.rag.external_rag.faithfulness": [0.92, 0.88],
                "metrics.rag.external_rag.context_correctness": [0.89, 0.91],
            }
        )
        mock_ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {
                    "score": 0.91,
                    "score_ci_low": 0.85,
                    "score_ci_high": 0.97,
                },
                "metrics.rag.external_rag.faithfulness": {"score": 0.90, "score_ci_low": 0.84, "score_ci_high": 0.96},
                "metrics.rag.external_rag.context_correctness": {
                    "score": 0.90,
                    "score_ci_low": 0.86,
                    "score_ci_high": 0.94,
                },
            }
        )

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics(
            sample_evaluation_data_list,
            [MetricType.ANSWER_CORRECTNESS, MetricType.FAITHFULNESS, MetricType.CONTEXT_CORRECTNESS],
        )

        assert len(result["scores"]) == 4
        assert "overall_score" in result["scores"]
        assert "overall_score" in result["question_scores"]
        assert len(result["question_scores"]) == 4


class TestUnitxtEvaluatorIntegration:
    """Integration tests for UnitxtEvaluator."""

    def test_evaluator_is_base_evaluator_instance(self):
        """Test that UnitxtEvaluator is an instance of BaseEvaluator."""
        from ai4rag.evaluator.base_evaluator import BaseEvaluator

        evaluator = UnitxtEvaluator()
        assert isinstance(evaluator, BaseEvaluator)

    def test_full_evaluation_workflow(self, mocker):
        """Test complete evaluation workflow from data to results."""
        evaluation_data = [
            EvaluationData(
                question="What is RAG?",
                answer="RAG is Retrieval-Augmented Generation.",
                contexts=["RAG stands for Retrieval-Augmented Generation."],
                context_ids=["doc1"],
                ground_truths=["RAG is Retrieval-Augmented Generation."],
                ground_truths_context_ids=["doc1"],
                question_id="q1",
            )
        ]

        mock_scores_df = pd.DataFrame(
            {
                "question_id": ["q1"],
                "metrics.rag.external_rag.answer_correctness": [0.98],
            }
        )
        mock_ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {
                    "score": 0.98,
                    "score_ci_low": 0.95,
                    "score_ci_high": 1.0,
                },
            }
        )

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics(evaluation_data, [MetricType.ANSWER_CORRECTNESS])

        assert result["scores"][MetricType.ANSWER_CORRECTNESS]["mean"] == 0.98
        assert result["scores"][MetricType.ANSWER_CORRECTNESS]["ci_low"] == 0.95
        assert result["scores"][MetricType.ANSWER_CORRECTNESS]["ci_high"] == 1.0
        assert result["question_scores"][MetricType.ANSWER_CORRECTNESS]["q1"] == 0.98


class TestUnitxtEvaluatorEdgeCases:
    """Test suite for edge cases in UnitxtEvaluator."""

    def test_empty_evaluation_data(self, mocker):
        """Test evaluate_metrics with empty evaluation data list."""
        mock_scores_df = pd.DataFrame()
        mock_ci_table = pd.DataFrame()

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics([], [MetricType.ANSWER_CORRECTNESS])

        assert "scores" in result
        assert "question_scores" in result

    def test_metric_name_case_sensitivity(self):
        """Test that metric names are case-sensitive."""
        evaluator = UnitxtEvaluator()
        result = evaluator.get_metric_types(["ANSWER_CORRECTNESS", "answer_correctness"])
        assert result == ["metrics.rag.external_rag.answer_correctness"]

    def test_decode_with_invalid_unitxt_metric(self):
        """Test decode_unitxt_metric with invalid metric raises KeyError."""
        with pytest.raises(KeyError):
            UnitxtEvaluator.decode_unitxt_metric(["invalid.metric.name"])
