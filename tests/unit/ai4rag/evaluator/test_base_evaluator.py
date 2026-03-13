# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.evaluator.base_evaluator import BaseEvaluator, EvaluationData, MetricType


@pytest.fixture
def sample_evaluation_data() -> EvaluationData:
    """Fixture providing a sample EvaluationData instance."""
    return EvaluationData(
        question="What is Python?",
        answer="Python is a programming language.",
        contexts=["Python is a high-level programming language.", "It was created by Guido van Rossum."],
        context_ids=["doc1", "doc2"],
        ground_truths=["Python is a programming language.", "Python is a high-level language."],
        ground_truths_context_ids=["doc1", "doc3"],
        question_id="q1",
        additional_data=["some", "data"],
    )


class TestEvaluationDataInitialization:
    """Test suite for EvaluationData initialization."""

    def test_init_with_all_fields(self, sample_evaluation_data):
        """Test initialization with all fields populated."""
        assert sample_evaluation_data.question == "What is Python?"
        assert sample_evaluation_data.answer == "Python is a programming language."
        assert len(sample_evaluation_data.contexts) == 2
        assert len(sample_evaluation_data.context_ids) == 2
        assert len(sample_evaluation_data.ground_truths) == 2
        assert len(sample_evaluation_data.ground_truths_context_ids) == 2
        assert sample_evaluation_data.question_id == "q1"
        assert sample_evaluation_data.additional_data == ["some", "data"]

    def test_init_with_default_values(self):
        """Test initialization with default values (all None)."""
        eval_data = EvaluationData()
        assert eval_data.question is None
        assert eval_data.answer is None
        assert eval_data.contexts is None
        assert eval_data.context_ids is None
        assert eval_data.ground_truths is None
        assert eval_data.ground_truths_context_ids is None
        assert eval_data.question_id is None
        assert eval_data.additional_data is None

    def test_init_with_partial_fields(self):
        """Test initialization with partial fields."""
        eval_data = EvaluationData(
            question="What is AI?",
            answer="AI stands for Artificial Intelligence.",
        )
        assert eval_data.question == "What is AI?"
        assert eval_data.answer == "AI stands for Artificial Intelligence."
        assert eval_data.contexts is None
        assert eval_data.context_ids is None

    def test_init_with_empty_lists(self):
        """Test initialization with empty lists."""
        eval_data = EvaluationData(
            question="Test question?",
            answer="Test answer.",
            contexts=[],
            context_ids=[],
            ground_truths=[],
            ground_truths_context_ids=[],
        )
        assert eval_data.contexts == []
        assert eval_data.context_ids == []
        assert eval_data.ground_truths == []
        assert eval_data.ground_truths_context_ids == []


class TestEvaluationDataToDict:
    """Test suite for EvaluationData to_dict method."""

    def test_to_dict_with_all_fields(self, sample_evaluation_data):
        """Test to_dict conversion with all fields populated."""
        result = sample_evaluation_data.to_dict()
        assert isinstance(result, dict)
        assert result["question"] == "What is Python?"
        assert result["answer"] == "Python is a programming language."
        assert result["contexts"] == [
            "Python is a high-level programming language.",
            "It was created by Guido van Rossum.",
        ]
        assert result["context_ids"] == ["doc1", "doc2"]
        assert result["ground_truths"] == ["Python is a programming language.", "Python is a high-level language."]
        assert result["ground_truths_context_ids"] == ["doc1", "doc3"]
        assert result["question_id"] == "q1"
        assert result["additional_data"] == ["some", "data"]

    def test_to_dict_with_default_values(self):
        """Test to_dict conversion with default values."""
        eval_data = EvaluationData()
        result = eval_data.to_dict()
        assert isinstance(result, dict)
        assert result["question"] is None
        assert result["answer"] is None
        assert result["contexts"] is None
        assert result["context_ids"] is None
        assert result["ground_truths"] is None
        assert result["ground_truths_context_ids"] is None
        assert result["question_id"] is None
        assert result["additional_data"] is None

    def test_to_dict_contains_all_keys(self, sample_evaluation_data):
        """Test that to_dict includes all expected keys."""
        result = sample_evaluation_data.to_dict()
        expected_keys = {
            "question",
            "answer",
            "contexts",
            "context_ids",
            "ground_truths",
            "ground_truths_context_ids",
            "question_id",
            "additional_data",
        }
        assert set(result.keys()) == expected_keys


class TestMetricType:
    """Test suite for MetricType constants."""

    def test_answer_correctness_constant(self):
        """Test ANSWER_CORRECTNESS constant value."""
        assert MetricType.ANSWER_CORRECTNESS == "answer_correctness"

    def test_faithfulness_constant(self):
        """Test FAITHFULNESS constant value."""
        assert MetricType.FAITHFULNESS == "faithfulness"

    def test_context_correctness_constant(self):
        """Test CONTEXT_CORRECTNESS constant value."""
        assert MetricType.CONTEXT_CORRECTNESS == "context_correctness"

    def test_all_metrics_are_strings(self):
        """Test that all metric constants are strings."""
        assert isinstance(MetricType.ANSWER_CORRECTNESS, str)
        assert isinstance(MetricType.FAITHFULNESS, str)
        assert isinstance(MetricType.CONTEXT_CORRECTNESS, str)

    def test_metrics_are_unique(self):
        """Test that all metric constants have unique values."""
        metrics = [
            MetricType.ANSWER_CORRECTNESS,
            MetricType.FAITHFULNESS,
            MetricType.CONTEXT_CORRECTNESS,
        ]
        assert len(metrics) == len(set(metrics))


class TestBaseEvaluator:
    """Test suite for BaseEvaluator abstract class."""

    def test_cannot_instantiate_base_evaluator(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEvaluator()

    def test_evaluate_metrics_is_abstract(self):
        """Test that evaluate_metrics is an abstract method."""
        assert hasattr(BaseEvaluator, "evaluate_metrics")
        assert getattr(BaseEvaluator.evaluate_metrics, "__isabstractmethod__", False)

    def test_subclass_must_implement_evaluate_metrics(self):
        """Test that subclass must implement evaluate_metrics."""

        class IncompleteEvaluator(BaseEvaluator):
            pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteEvaluator()

    def test_valid_subclass_implementation(self):
        """Test that a valid subclass can be instantiated."""

        class ValidEvaluator(BaseEvaluator):
            def evaluate_metrics(self, evaluation_data, metrics):
                return {"test": "result"}

        evaluator = ValidEvaluator()
        assert isinstance(evaluator, BaseEvaluator)
        result = evaluator.evaluate_metrics([], [])
        assert result == {"test": "result"}


class TestEvaluationDataEdgeCases:
    """Test suite for EvaluationData edge cases."""

    def test_very_long_strings(self):
        """Test EvaluationData with very long strings."""
        long_text = "A" * 10000
        eval_data = EvaluationData(
            question=long_text,
            answer=long_text,
        )
        assert len(eval_data.question) == 10000
        assert len(eval_data.answer) == 10000

    def test_many_contexts(self):
        """Test EvaluationData with many contexts."""
        many_contexts = [f"Context {i}" for i in range(100)]
        many_ids = [f"doc{i}" for i in range(100)]
        eval_data = EvaluationData(
            contexts=many_contexts,
            context_ids=many_ids,
        )
        assert len(eval_data.contexts) == 100
        assert len(eval_data.context_ids) == 100

    def test_special_characters_in_fields(self):
        """Test EvaluationData with special characters."""
        eval_data = EvaluationData(
            question="What is 'AI' & <ML>?",
            answer='AI & ML are "technologies".',
            question_id="q-123_test",
        )
        assert eval_data.question == "What is 'AI' & <ML>?"
        assert eval_data.answer == 'AI & ML are "technologies".'
        assert eval_data.question_id == "q-123_test"

    def test_unicode_characters(self):
        """Test EvaluationData with unicode characters."""
        eval_data = EvaluationData(
            question="Qu'est-ce que l'IA? 什么是人工智能?",
            answer="L'IA est... 人工智能是...",
        )
        assert "Qu'est-ce" in eval_data.question
        assert "人工智能" in eval_data.question

    def test_whitespace_preservation(self):
        """Test that whitespace is preserved in fields."""
        eval_data = EvaluationData(
            question="  Question with    spaces  ",
            answer="\tAnswer with\ttabs\t",
        )
        assert eval_data.question == "  Question with    spaces  "
        assert eval_data.answer == "\tAnswer with\ttabs\t"

    def test_newlines_in_fields(self):
        """Test EvaluationData with newlines in fields."""
        eval_data = EvaluationData(
            question="Line 1\nLine 2\nLine 3",
            answer="Answer\nwith\nnewlines",
        )
        assert "\n" in eval_data.question
        assert "\n" in eval_data.answer
        assert eval_data.question.count("\n") == 2
        assert eval_data.answer.count("\n") == 2

    def test_mixed_type_additional_data(self):
        """Test EvaluationData with mixed type additional_data."""
        mixed_data = [123, "string", {"key": "value"}, [1, 2, 3], None]
        eval_data = EvaluationData(additional_data=mixed_data)
        assert eval_data.additional_data == mixed_data
        assert len(eval_data.additional_data) == 5
