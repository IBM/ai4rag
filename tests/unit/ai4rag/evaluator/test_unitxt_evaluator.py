# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
import pytest

from ai4rag.core.experiment.exception_handler import EvaluationError
from ai4rag.evaluator.base_evaluator import EvaluationData
from ai4rag.evaluator.metric import Metrics, RAGMetric
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


@pytest.fixture
def sample_metric_lookup() -> dict[str, RAGMetric]:
    """Fixture providing a unitxt-name → RAGMetric lookup for the three standard metrics."""
    return {
        UnitxtEvaluator.METRIC_TYPE_MAP[m.name]: m
        for m in (Metrics.ANSWER_CORRECTNESS, Metrics.FAITHFULNESS, Metrics.CONTEXT_CORRECTNESS)
    }


class TestUnitxtEvaluatorMetricMapping:
    """Test suite for UnitxtEvaluator metric type mapping."""

    def test_get_metric_types_single_metric(self):
        result = UnitxtEvaluator.get_metric_types([Metrics.ANSWER_CORRECTNESS])
        assert result == ["metrics.rag.external_rag.answer_correctness"]

    def test_get_metric_types_multiple_metrics(self):
        result = UnitxtEvaluator.get_metric_types(
            [Metrics.ANSWER_CORRECTNESS, Metrics.FAITHFULNESS, Metrics.CONTEXT_CORRECTNESS]
        )
        assert len(result) == 3
        assert "metrics.rag.external_rag.answer_correctness" in result
        assert "metrics.rag.external_rag.faithfulness" in result
        assert "metrics.rag.external_rag.context_correctness" in result

    def test_get_metric_types_empty_list(self):
        result = UnitxtEvaluator.get_metric_types([])
        assert result == []

    def test_get_metric_types_unknown_metric_skipped(self):
        unknown = RAGMetric(name="not_in_map", evaluator="unitxt", description="")
        result = UnitxtEvaluator.get_metric_types([unknown, Metrics.FAITHFULNESS])
        assert result == ["metrics.rag.external_rag.faithfulness"]

    def test_get_metric_types_all_unknown(self):
        unknown = RAGMetric(name="nope", evaluator="unitxt", description="")
        result = UnitxtEvaluator.get_metric_types([unknown])
        assert result == []

    def test_decode_unitxt_metric_single(self):
        """Test decode_unitxt_metric with a single metric."""
        result = UnitxtEvaluator.decode_unitxt_metric(["metrics.rag.external_rag.answer_correctness"])
        assert result == [Metrics.ANSWER_CORRECTNESS.name]

    def test_decode_unitxt_metric_multiple(self):
        """Test decode_unitxt_metric with multiple metrics."""
        unitxt_metrics = [
            "metrics.rag.external_rag.answer_correctness",
            "metrics.rag.external_rag.faithfulness",
            "metrics.rag.external_rag.context_correctness",
        ]
        result = UnitxtEvaluator.decode_unitxt_metric(unitxt_metrics)
        assert len(result) == 3
        assert Metrics.ANSWER_CORRECTNESS.name in result
        assert Metrics.FAITHFULNESS.name in result
        assert Metrics.CONTEXT_CORRECTNESS.name in result

    def test_decode_unitxt_metric_empty_list(self):
        """Test decode_unitxt_metric with empty list."""
        result = UnitxtEvaluator.decode_unitxt_metric([])
        assert result == []


class TestBuildAggregateMetrics:
    """Test suite for _build_aggregate_metrics method."""

    def test_basic(self, sample_ci_table, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_aggregate_metrics(sample_ci_table, sample_metric_lookup)

        assert isinstance(result, list)
        assert len(result) == 3
        names = {entry["name"] for entry in result}
        assert names == {Metrics.ANSWER_CORRECTNESS.name, Metrics.FAITHFULNESS.name, Metrics.CONTEXT_CORRECTNESS.name}

    def test_structure(self, sample_ci_table, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_aggregate_metrics(sample_ci_table, sample_metric_lookup)

        for entry in result:
            assert "name" in entry
            assert "evaluator" in entry
            assert "description" in entry
            assert "scores" in entry
            assert "mean" in entry["scores"]
            assert "ci_low" in entry["scores"]
            assert "ci_high" in entry["scores"]

    def test_evaluator_and_description_populated(self, sample_ci_table, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_aggregate_metrics(sample_ci_table, sample_metric_lookup)

        for entry in result:
            assert entry["evaluator"] == "unitxt"
            assert isinstance(entry["description"], str)

    def test_rounding(self, sample_ci_table, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_aggregate_metrics(sample_ci_table, sample_metric_lookup)

        ac = next(e for e in result if e["name"] == Metrics.ANSWER_CORRECTNESS.name)
        assert ac["scores"]["mean"] == 0.91
        assert ac["scores"]["ci_low"] == 0.85
        assert ac["scores"]["ci_high"] == 0.97

    def test_with_nan(self, sample_metric_lookup):
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
        result = evaluator._build_aggregate_metrics(ci_table, sample_metric_lookup)

        ac = next(e for e in result if e["name"] == Metrics.ANSWER_CORRECTNESS.name)
        faith = next(e for e in result if e["name"] == Metrics.FAITHFULNESS.name)
        assert ac["scores"]["mean"] == 0.91
        assert faith["scores"]["mean"] is None
        assert faith["scores"]["ci_low"] is None
        assert faith["scores"]["ci_high"] is None

    def test_missing_ci_columns(self, sample_metric_lookup):
        ci_table = pd.DataFrame(
            {
                "metrics.rag.external_rag.answer_correctness": {"score": 0.91},
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._build_aggregate_metrics(ci_table, sample_metric_lookup)

        ac = result[0]
        assert ac["scores"]["mean"] == 0.91
        assert ac["scores"]["ci_low"] is None
        assert ac["scores"]["ci_high"] is None

    def test_precision(self, sample_metric_lookup):
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
        result = evaluator._build_aggregate_metrics(ci_table, sample_metric_lookup)

        ac = result[0]
        assert ac["scores"]["mean"] == 0.1235
        assert ac["scores"]["ci_low"] == 0.9877
        assert ac["scores"]["ci_high"] == 0.5556


class TestBuildQuestionScores:
    """Test suite for _build_question_scores method."""

    def test_basic(self, sample_scores_df, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(sample_scores_df, sample_metric_lookup)

        assert isinstance(result, list)
        assert len(result) == 2
        question_ids = {entry["question_id"] for entry in result}
        assert question_ids == {"q1", "q2"}

    def test_structure(self, sample_scores_df, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(sample_scores_df, sample_metric_lookup)

        for entry in result:
            assert "question_id" in entry
            assert "metrics" in entry
            assert isinstance(entry["metrics"], list)
            for m in entry["metrics"]:
                assert "name" in m
                assert "evaluator" in m
                assert "value" in m

    def test_values(self, sample_scores_df, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(sample_scores_df, sample_metric_lookup)

        q1 = next(e for e in result if e["question_id"] == "q1")
        q2 = next(e for e in result if e["question_id"] == "q2")

        q1_metrics = {m["name"]: m["value"] for m in q1["metrics"]}
        q2_metrics = {m["name"]: m["value"] for m in q2["metrics"]}

        assert q1_metrics[Metrics.ANSWER_CORRECTNESS.name] == 0.95
        assert q2_metrics[Metrics.ANSWER_CORRECTNESS.name] == 0.87
        assert q1_metrics[Metrics.FAITHFULNESS.name] == 0.92
        assert q2_metrics[Metrics.FAITHFULNESS.name] == 0.88

    def test_evaluator_populated(self, sample_scores_df, sample_metric_lookup):
        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(sample_scores_df, sample_metric_lookup)

        for entry in result:
            for m in entry["metrics"]:
                assert m["evaluator"] == "unitxt"

    def test_rounding(self, sample_metric_lookup):
        scores_df = pd.DataFrame(
            {
                "question_id": ["q1"],
                "metrics.rag.external_rag.answer_correctness": [0.123456789],
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(scores_df, sample_metric_lookup)

        assert result[0]["metrics"][0]["value"] == 0.1235

    def test_unevaluable_values_are_omitted(self, sample_metric_lookup):
        """Empty strings and NaN mark unevaluable records: they get no metric entry."""
        scores_df = pd.DataFrame(
            {
                "question_id": ["q1", "q2", "q3"],
                "metrics.rag.external_rag.answer_correctness": [0.95, "", np.nan],
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(scores_df, sample_metric_lookup)

        q1 = next(e for e in result if e["question_id"] == "q1")
        q2 = next(e for e in result if e["question_id"] == "q2")
        q3 = next(e for e in result if e["question_id"] == "q3")
        assert q1["metrics"][0]["value"] == 0.95
        assert q2["metrics"] == []
        assert q3["metrics"] == []

    def test_filters_irrelevant_columns(self, sample_metric_lookup):
        scores_df = pd.DataFrame(
            {
                "question_id": ["q1"],
                "metrics.rag.external_rag.answer_correctness": [0.95],
                "some_other_column": [0.50],
                "another_column": ["text"],
            }
        )

        evaluator = UnitxtEvaluator()
        result = evaluator._build_question_scores(scores_df, sample_metric_lookup)

        assert len(result) == 1
        assert len(result[0]["metrics"]) == 1
        assert result[0]["metrics"][0]["name"] == Metrics.ANSWER_CORRECTNESS.name


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
            [Metrics.ANSWER_CORRECTNESS, Metrics.FAITHFULNESS],
        )

        assert "metrics" in result
        assert "question_scores" in result
        assert isinstance(result["metrics"], list)
        assert isinstance(result["question_scores"], list)

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
        evaluator.evaluate_metrics(sample_evaluation_data_list, [Metrics.ANSWER_CORRECTNESS])

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
            evaluator.evaluate_metrics(sample_evaluation_data_list, [Metrics.ANSWER_CORRECTNESS])

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
        result = evaluator.evaluate_metrics(sample_evaluation_data_list, [Metrics.FAITHFULNESS])

        names = {m["name"] for m in result["metrics"]}
        assert Metrics.FAITHFULNESS.name in names
        q_names = {m["name"] for q in result["question_scores"] for m in q["metrics"]}
        assert Metrics.FAITHFULNESS.name in q_names

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
            [Metrics.ANSWER_CORRECTNESS, Metrics.FAITHFULNESS, Metrics.CONTEXT_CORRECTNESS],
        )

        assert len(result["metrics"]) == 3
        assert len(result["question_scores"]) == 2


class TestUnitxtEvaluatorEmptyReferences:
    """Records lacking a metric's references must be excluded, not crash the run.

    These exercise the *real* unitxt ``evaluate`` (reference-based token-overlap
    metrics need no LLM, so they run offline). They reproduce the production
    ``TokenOverlap`` failure (``max() iterable argument is empty``) that occurred
    when a record had no ``contexts``/``ground_truths`` for a metric.
    """

    @staticmethod
    def _data() -> list[EvaluationData]:
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
            # No contexts and no ground_truths: faithfulness/answer_correctness
            # cannot be computed for this record and used to crash unitxt.
            EvaluationData(
                question="What is AI?",
                answer="AI is Artificial Intelligence.",
                contexts=[],
                context_ids=[],
                ground_truths=[],
                ground_truths_context_ids=[],
                question_id="q2",
            ),
        ]

    def test_missing_references_excluded_without_crash(self):
        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics(
            self._data(),
            [Metrics.FAITHFULNESS, Metrics.ANSWER_CORRECTNESS, Metrics.CONTEXT_CORRECTNESS],
        )

        names = {m["name"] for m in result["metrics"]}
        assert names == {
            Metrics.FAITHFULNESS.name,
            Metrics.ANSWER_CORRECTNESS.name,
            Metrics.CONTEXT_CORRECTNESS.name,
        }

        # The reference-bearing record still produces a mean for both
        # reference-based metrics (computed over q1 alone).
        by_name = {m["name"]: m for m in result["metrics"]}
        assert by_name[Metrics.FAITHFULNESS.name]["scores"]["mean"] is not None
        assert by_name[Metrics.ANSWER_CORRECTNESS.name]["scores"]["mean"] is not None

        # The record without references gets no faithfulness/answer_correctness
        # per-question score, while q1 does.
        scores_by_qid = {q["question_id"]: {m["name"] for m in q["metrics"]} for q in result["question_scores"]}
        assert Metrics.FAITHFULNESS.name in scores_by_qid["q1"]
        assert Metrics.ANSWER_CORRECTNESS.name in scores_by_qid["q1"]
        assert Metrics.FAITHFULNESS.name not in scores_by_qid["q2"]
        assert Metrics.ANSWER_CORRECTNESS.name not in scores_by_qid["q2"]

    def test_all_records_missing_references_yield_none_mean(self):
        data = [
            EvaluationData(
                question="What is AI?",
                answer="AI is Artificial Intelligence.",
                contexts=[],
                context_ids=[],
                ground_truths=[],
                ground_truths_context_ids=[],
                question_id="q1",
            )
        ]

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics(data, [Metrics.FAITHFULNESS])

        faith = result["metrics"][0]
        assert faith["name"] == Metrics.FAITHFULNESS.name
        assert faith["scores"]["mean"] is None
        assert result["question_scores"][0]["metrics"] == []


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
        result = evaluator.evaluate_metrics(evaluation_data, [Metrics.ANSWER_CORRECTNESS])

        ac_metric = result["metrics"][0]
        assert ac_metric["name"] == Metrics.ANSWER_CORRECTNESS.name
        assert ac_metric["scores"]["mean"] == 0.98
        assert ac_metric["scores"]["ci_low"] == 0.95
        assert ac_metric["scores"]["ci_high"] == 1.0

        q1 = result["question_scores"][0]
        assert q1["question_id"] == "q1"
        assert q1["metrics"][0]["value"] == 0.98


class TestUnitxtEvaluatorEdgeCases:
    """Test suite for edge cases in UnitxtEvaluator."""

    def test_empty_evaluation_data(self, mocker):
        """Test evaluate_metrics with empty evaluation data list."""
        mock_scores_df = pd.DataFrame()
        mock_ci_table = pd.DataFrame()

        mock_evaluate = mocker.patch("ai4rag.evaluator.unitxt_evaluator.evaluate")
        mock_evaluate.return_value = (mock_scores_df, mock_ci_table)

        evaluator = UnitxtEvaluator()
        result = evaluator.evaluate_metrics([], [Metrics.ANSWER_CORRECTNESS])

        assert "metrics" in result
        assert "question_scores" in result
        assert result["metrics"] == []
        assert result["question_scores"] == []

    def test_name_case_sensitivity(self):
        """Test that metric names are case-sensitive in METRIC_TYPE_MAP lookup."""
        upper = RAGMetric(name="ANSWER_CORRECTNESS", evaluator="unitxt", description="")
        lower = Metrics.ANSWER_CORRECTNESS
        result = UnitxtEvaluator.get_metric_types([upper, lower])
        assert result == ["metrics.rag.external_rag.answer_correctness"]

    def test_decode_with_invalid_unitxt_metric(self):
        """Test decode_unitxt_metric with invalid metric raises KeyError."""
        with pytest.raises(KeyError):
            UnitxtEvaluator.decode_unitxt_metric(["invalid.metric.name"])
