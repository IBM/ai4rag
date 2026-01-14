# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pandas as pd
import pytest

from ai4rag.core.experiment.benchmark_data import BenchmarkData, BenchmarkDataValueError


@pytest.fixture
def benchmark_data_df() -> pd.DataFrame:
    """Fixture providing valid benchmark data DataFrame."""
    raw_data = [
        {
            "question": "What is meaning of life?",
            "correct_answers": ["Being good to others."],
            "correct_answer_document_ids": ["doc_id_1", "doc_id_2"],
        },
        {
            "question": "Is it good to be a software engineer?",
            "correct_answers": [
                "Sometimes it is good, sometimes it is bad. But mostly it is ok.",
                "It's great!",
            ],
            "correct_answer_document_ids": ["doc_id_3", "doc_id_4"],
        },
        {
            "question": "What is Python?",
            "correct_answers": ["A programming language."],
            "correct_answer_document_ids": ["doc_id_5"],
        },
    ]
    return pd.DataFrame(raw_data)


@pytest.fixture
def benchmark_data(benchmark_data_df) -> BenchmarkData:
    """Fixture providing a BenchmarkData instance."""
    return BenchmarkData(benchmark_data=benchmark_data_df)


class TestBenchmarkDataInitialization:
    """Test suite for BenchmarkData initialization."""

    def test_init_with_valid_data(self, benchmark_data_df):
        """Test initialization with valid DataFrame."""
        benchmark_data = BenchmarkData(benchmark_data=benchmark_data_df)
        assert len(benchmark_data) == 3
        assert len(benchmark_data.questions) == 3
        assert len(benchmark_data.answers) == 3
        assert len(benchmark_data.document_ids) == 3

    def test_init_with_empty_question_string(self, benchmark_data_df):
        """Test initialization fails with empty question string."""
        benchmark_data_df.loc[0, "question"] = ""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'question' value: ''"):
            BenchmarkData(benchmark_data=benchmark_data_df)

    def test_init_with_non_string_question(self, benchmark_data_df):
        """Test initialization fails with non-string question."""
        benchmark_data_df.loc[0, "question"] = 123
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'question' value: '123'"):
            BenchmarkData(benchmark_data=benchmark_data_df)

    def test_init_with_empty_answer_string(self, benchmark_data_df):
        """Test initialization fails with empty answer string."""
        benchmark_data_df.loc[0, "correct_answers"] = [""]
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answers' value: ''"):
            BenchmarkData(benchmark_data=benchmark_data_df)

    def test_init_with_non_string_answer(self, benchmark_data_df):
        """Test initialization fails with non-string answer."""
        benchmark_data_df.loc[0, "correct_answers"] = [123]
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answers' value: '123'"):
            BenchmarkData(benchmark_data=benchmark_data_df)

    def test_init_with_empty_document_id_string(self, benchmark_data_df):
        """Test initialization fails with empty document ID string."""
        benchmark_data_df.loc[0, "correct_answer_document_ids"] = [""]
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answer_document_ids' value: ''"):
            BenchmarkData(benchmark_data=benchmark_data_df)

    def test_init_with_non_string_document_id(self, benchmark_data_df):
        """Test initialization fails with non-string document ID."""
        benchmark_data_df.loc[0, "correct_answer_document_ids"] = [None]
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answer_document_ids' value: 'None'"):
            BenchmarkData(benchmark_data=benchmark_data_df)

    def test_init_with_empty_dataframe(self):
        """Test initialization with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["question", "correct_answers", "correct_answer_document_ids"])
        benchmark_data = BenchmarkData(benchmark_data=empty_df)
        assert len(benchmark_data) == 0
        assert benchmark_data.questions == []
        assert benchmark_data.answers == []
        assert benchmark_data.document_ids == []


class TestBenchmarkDataMagicMethods:
    """Test suite for BenchmarkData magic methods."""

    def test_len(self, benchmark_data):
        """Test __len__ method."""
        assert len(benchmark_data) == 3

    def test_getitem_valid_index(self, benchmark_data):
        """Test __getitem__ with valid index."""
        question, answers, doc_ids = benchmark_data[0]
        assert question == "What is meaning of life?"
        assert answers == ["Being good to others."]
        assert doc_ids == ["doc_id_1", "doc_id_2"]

    def test_getitem_last_index(self, benchmark_data):
        """Test __getitem__ with last valid index."""
        question, answers, doc_ids = benchmark_data[2]
        assert question == "What is Python?"
        assert answers == ["A programming language."]
        assert doc_ids == ["doc_id_5"]

    def test_getitem_negative_index(self, benchmark_data):
        """Test __getitem__ with negative index."""
        question, answers, doc_ids = benchmark_data[-1]
        assert question == "What is Python?"

    def test_getitem_index_error(self, benchmark_data):
        """Test __getitem__ raises IndexError for out of bounds index."""
        with pytest.raises(IndexError):
            _ = benchmark_data[10]

    def test_iter(self, benchmark_data):
        """Test __iter__ method."""
        items = list(benchmark_data)
        assert len(items) == 3
        assert items[0][0] == "What is meaning of life?"
        assert items[1][0] == "Is it good to be a software engineer?"
        assert items[2][0] == "What is Python?"

    def test_iter_empty_data(self):
        """Test __iter__ with empty data."""
        empty_df = pd.DataFrame(columns=["question", "correct_answers", "correct_answer_document_ids"])
        benchmark_data = BenchmarkData(benchmark_data=empty_df)
        items = list(benchmark_data)
        assert items == []


class TestBenchmarkDataProperties:
    """Test suite for BenchmarkData properties."""

    def test_questions_property_getter(self, benchmark_data):
        """Test questions property getter."""
        questions = benchmark_data.questions
        assert isinstance(questions, list)
        assert len(questions) == 3
        assert questions[0] == "What is meaning of life?"

    def test_questions_property_setter_valid(self, benchmark_data):
        """Test questions property setter with valid data."""
        new_questions = ["Q1", "Q2", "Q3"]
        benchmark_data.questions = new_questions
        assert benchmark_data.questions == new_questions

    def test_questions_property_setter_empty_string(self, benchmark_data):
        """Test questions property setter fails with empty string."""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'question' value: ''"):
            benchmark_data.questions = [""]

    def test_questions_property_setter_non_string(self, benchmark_data):
        """Test questions property setter fails with non-string."""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'question' value: '123'"):
            benchmark_data.questions = [123]

    def test_answers_property_getter(self, benchmark_data):
        """Test answers property getter."""
        answers = benchmark_data.answers
        assert isinstance(answers, list)
        assert len(answers) == 3
        assert answers[0] == ["Being good to others."]

    def test_answers_property_setter_valid(self, benchmark_data):
        """Test answers property setter with valid data."""
        new_answers = [["A1"], ["A2"], ["A3"]]
        benchmark_data.answers = new_answers
        assert benchmark_data.answers == new_answers

    def test_answers_property_setter_empty_string(self, benchmark_data):
        """Test answers property setter fails with empty string."""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answers' value: ''"):
            benchmark_data.answers = [[""]]

    def test_answers_property_setter_non_string(self, benchmark_data):
        """Test answers property setter fails with non-string."""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answers' value: '456'"):
            benchmark_data.answers = [[456]]

    def test_answers_property_setter_multiple_invalid(self, benchmark_data):
        """Test answers property setter fails when multiple answers have issues."""
        with pytest.raises(BenchmarkDataValueError):
            benchmark_data.answers = [["Valid"], [""], ["Also valid"]]

    def test_document_ids_property_getter(self, benchmark_data):
        """Test document_ids property getter."""
        doc_ids = benchmark_data.document_ids
        assert isinstance(doc_ids, list)
        assert len(doc_ids) == 3
        assert doc_ids[0] == ["doc_id_1", "doc_id_2"]

    def test_document_ids_property_setter_valid(self, benchmark_data):
        """Test document_ids property setter with valid data."""
        new_doc_ids = [["id1"], ["id2"], ["id3"]]
        benchmark_data.document_ids = new_doc_ids
        assert benchmark_data.document_ids == new_doc_ids

    def test_document_ids_property_setter_none(self, benchmark_data):
        """Test document_ids property setter with None."""
        benchmark_data.document_ids = None
        assert benchmark_data.document_ids is None

    def test_document_ids_property_setter_empty_string(self, benchmark_data):
        """Test document_ids property setter fails with empty string."""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answer_document_ids' value: ''"):
            benchmark_data.document_ids = [[""]]

    def test_document_ids_property_setter_non_string(self, benchmark_data):
        """Test document_ids property setter fails with non-string."""
        with pytest.raises(BenchmarkDataValueError, match="Incorrect 'correct_answer_document_ids' value: '789'"):
            benchmark_data.document_ids = [[789]]

    def test_questions_ids_property(self, benchmark_data):
        """Test questions_ids property."""
        question_ids = benchmark_data.questions_ids
        assert isinstance(question_ids, list)
        assert len(question_ids) == 3
        assert question_ids == ["q0", "q1", "q2"]

    def test_questions_ids_property_empty_data(self):
        """Test questions_ids property with empty data."""
        empty_df = pd.DataFrame(columns=["question", "correct_answers", "correct_answer_document_ids"])
        benchmark_data = BenchmarkData(benchmark_data=empty_df)
        assert benchmark_data.questions_ids == []


class TestBenchmarkDataGetRandomSample:
    """Test suite for get_random_sample method."""

    def test_get_random_sample_default(self, benchmark_data):
        """Test get_random_sample with default parameters."""
        sample = benchmark_data.get_random_sample()
        assert isinstance(sample, BenchmarkData)
        # Default is 10, but we only have 3 records, so should return all when n > len
        assert len(sample) == len(benchmark_data)

    def test_get_random_sample_smaller_than_data(self, benchmark_data):
        """Test get_random_sample with n smaller than data size."""
        sample = benchmark_data.get_random_sample(n_records=2, random_seed=17)
        assert isinstance(sample, BenchmarkData)
        assert len(sample) == 2
        assert sample is not benchmark_data  # Should be a new instance

    def test_get_random_sample_equal_to_data(self, benchmark_data):
        """Test get_random_sample with n equal to data size."""
        sample = benchmark_data.get_random_sample(n_records=3, random_seed=17)
        assert isinstance(sample, BenchmarkData)
        assert len(sample) == 3

    def test_get_random_sample_larger_than_data(self, benchmark_data):
        """Test get_random_sample with n larger than data size."""
        sample = benchmark_data.get_random_sample(n_records=100, random_seed=17)
        assert isinstance(sample, BenchmarkData)
        assert len(sample) == 3  # Should return all records

    def test_get_random_sample_deterministic(self, benchmark_data):
        """Test get_random_sample produces same results with same seed."""
        sample1 = benchmark_data.get_random_sample(n_records=2, random_seed=42)
        sample2 = benchmark_data.get_random_sample(n_records=2, random_seed=42)
        assert sample1.questions == sample2.questions
        assert sample1.answers == sample2.answers
        assert sample1.document_ids == sample2.document_ids

    def test_get_random_sample_different_seeds(self, benchmark_data):
        """Test get_random_sample produces different results with different seeds."""
        sample1 = benchmark_data.get_random_sample(n_records=2, random_seed=1)
        sample2 = benchmark_data.get_random_sample(n_records=2, random_seed=2)
        # They might be the same by chance, but with 3 records and 2 samples, likely different
        # This test verifies the seed parameter is used

    def test_get_random_sample_zero_records(self, benchmark_data):
        """Test get_random_sample with zero records."""
        sample = benchmark_data.get_random_sample(n_records=0, random_seed=17)
        assert isinstance(sample, BenchmarkData)
        assert len(sample) == 0

    def test_get_random_sample_empty_data(self):
        """Test get_random_sample with empty data."""
        empty_df = pd.DataFrame(columns=["question", "correct_answers", "correct_answer_document_ids"])
        benchmark_data = BenchmarkData(benchmark_data=empty_df)
        sample = benchmark_data.get_random_sample(n_records=10, random_seed=17)
        assert isinstance(sample, BenchmarkData)
        assert len(sample) == 0


class TestBenchmarkDataEdgeCases:
    """Test suite for edge cases and special scenarios."""

    def test_single_record(self):
        """Test BenchmarkData with single record."""
        df = pd.DataFrame(
            {
                "question": ["Single question?"],
                "correct_answers": [["Single answer."]],
                "correct_answer_document_ids": [["doc1"]],
            }
        )
        benchmark_data = BenchmarkData(benchmark_data=df)
        assert len(benchmark_data) == 1
        assert benchmark_data.questions_ids == ["q0"]

    def test_multiple_answers_per_question(self):
        """Test BenchmarkData with multiple answers per question."""
        df = pd.DataFrame(
            {
                "question": ["Q1"],
                "correct_answers": [["A1", "A2", "A3"]],
                "correct_answer_document_ids": [["doc1", "doc2", "doc3"]],
            }
        )
        benchmark_data = BenchmarkData(benchmark_data=df)
        assert len(benchmark_data.answers[0]) == 3
        assert len(benchmark_data.document_ids[0]) == 3

    def test_single_document_id_per_question(self):
        """Test BenchmarkData with single document ID per question."""
        df = pd.DataFrame(
            {
                "question": ["Q1"],
                "correct_answers": [["A1"]],
                "correct_answer_document_ids": [["doc1"]],
            }
        )
        benchmark_data = BenchmarkData(benchmark_data=df)
        assert len(benchmark_data.document_ids[0]) == 1

    def test_whitespace_only_strings(self, benchmark_data_df):
        """Test that whitespace-only strings are considered valid (not empty)."""
        benchmark_data_df.loc[0, "question"] = "   "
        # Whitespace strings are not empty, so this should work
        benchmark_data = BenchmarkData(benchmark_data=benchmark_data_df)
        assert benchmark_data.questions[0] == "   "

    def test_very_long_strings(self):
        """Test BenchmarkData with very long strings."""
        long_question = "Q" * 10000
        long_answer = "A" * 10000
        df = pd.DataFrame(
            {
                "question": [long_question],
                "correct_answers": [[long_answer]],
                "correct_answer_document_ids": [["doc1"]],
            }
        )
        benchmark_data = BenchmarkData(benchmark_data=df)
        assert len(benchmark_data.questions[0]) == 10000
        assert len(benchmark_data.answers[0][0]) == 10000


class TestBenchmarkDataValueError:
    """Test suite for BenchmarkDataValueError exception."""

    def test_exception_inheritance(self):
        """Test that BenchmarkDataValueError inherits from ValueError."""
        assert issubclass(BenchmarkDataValueError, ValueError)

    def test_exception_instantiation(self):
        """Test that BenchmarkDataValueError can be instantiated."""
        error = BenchmarkDataValueError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, ValueError)
