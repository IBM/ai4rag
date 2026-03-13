# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import pytest

from ai4rag.core.experiment.results import EvaluationResult, ExperimentResults
from ai4rag.evaluator.base_evaluator import EvaluationData


class TestEvaluationResult:
    """Test suite for EvaluationResult dataclass."""

    @pytest.fixture
    def sample_scores(self):
        """Create sample scores dictionary."""
        return {
            "scores": {
                "answer_correctness": {"mean": 0.75, "ci_low": 0.65, "ci_high": 0.85},
                "context_correctness": {"mean": 0.80, "ci_low": 0.70, "ci_high": 0.90},
            },
            "question_scores": {
                "answer_correctness": {"q0": 0.75, "q1": 0.80},
                "context_correctness": {"q0": 0.85, "q1": 0.75},
            },
        }

    @pytest.fixture
    def sample_evaluation_result(self, sample_scores):
        """Create a sample EvaluationResult."""
        return EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params={"chunk_size": 512, "chunk_overlap": 128},
            rag_params={"retrieval_method": "simple", "number_of_chunks": 5},
            scores=sample_scores,
            execution_time=45.5,
            final_score=0.77,
        )

    def test_evaluation_result_creation(self, sample_evaluation_result):
        """Test creating an EvaluationResult instance."""
        assert sample_evaluation_result.pattern_name == "Pattern1"
        assert sample_evaluation_result.collection == "collection_1"
        assert sample_evaluation_result.indexing_params == {"chunk_size": 512, "chunk_overlap": 128}
        assert sample_evaluation_result.rag_params == {"retrieval_method": "simple", "number_of_chunks": 5}
        assert sample_evaluation_result.execution_time == 45.5
        assert sample_evaluation_result.final_score == 0.77

    def test_evaluation_result_is_frozen(self, sample_evaluation_result):
        """Test that EvaluationResult is immutable (frozen dataclass)."""
        with pytest.raises(AttributeError):
            sample_evaluation_result.pattern_name = "Pattern2"

    def test_evaluation_result_to_dict(self, sample_evaluation_result):
        """Test converting EvaluationResult to dictionary."""
        result_dict = sample_evaluation_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["pattern_name"] == "Pattern1"
        assert result_dict["collection"] == "collection_1"
        assert result_dict["indexing_params"] == {"chunk_size": 512, "chunk_overlap": 128}
        assert result_dict["rag_params"] == {"retrieval_method": "simple", "number_of_chunks": 5}
        assert result_dict["execution_time"] == 45.5
        assert result_dict["final_score"] == 0.77

    def test_evaluation_result_to_dict_contains_all_fields(self, sample_evaluation_result):
        """Test that to_dict includes all fields."""
        result_dict = sample_evaluation_result.to_dict()
        expected_keys = {
            "pattern_name",
            "collection",
            "indexing_params",
            "rag_params",
            "scores",
            "execution_time",
            "final_score",
            "rag_pattern",
        }
        assert set(result_dict.keys()) == expected_keys

    def test_evaluation_result_with_none_indexing_params(self, sample_scores):
        """Test EvaluationResult with None indexing_params."""
        result = EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params=None,
            rag_params={"retrieval_method": "simple"},
            scores=sample_scores,
            execution_time=10.0,
            final_score=0.5,
        )
        assert result.indexing_params is None

    def test_evaluation_result_equality(self, sample_scores):
        """Test equality of EvaluationResult instances."""
        result1 = EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params={"chunk_size": 512},
            rag_params={"retrieval_method": "simple"},
            scores=sample_scores,
            execution_time=10.0,
            final_score=0.5,
        )
        result2 = EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params={"chunk_size": 512},
            rag_params={"retrieval_method": "simple"},
            scores=sample_scores,
            execution_time=10.0,
            final_score=0.5,
        )
        assert result1 == result2


class TestExperimentResultsInitialization:
    """Test suite for ExperimentResults initialization."""

    def test_init_creates_empty_lists(self):
        """Test that initialization creates empty evaluation and evaluation_data lists."""
        results = ExperimentResults()
        assert results.evaluations == []
        assert results.evaluation_data == []

    def test_len_on_empty_results(self):
        """Test __len__ on empty ExperimentResults."""
        results = ExperimentResults()
        assert len(results) == 0

    def test_bool_on_empty_results(self):
        """Test __bool__ on empty ExperimentResults."""
        results = ExperimentResults()
        assert not results

    def test_iter_on_empty_results(self):
        """Test __iter__ on empty ExperimentResults."""
        results = ExperimentResults()
        assert list(results) == []


class TestExperimentResultsAddEvaluation:
    """Test suite for ExperimentResults.add_evaluation method."""

    @pytest.fixture
    def results(self):
        """Create an ExperimentResults instance."""
        return ExperimentResults()

    @pytest.fixture
    def sample_evaluation_result(self):
        """Create a sample EvaluationResult."""
        return EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params={"chunk_size": 512},
            rag_params={"retrieval_method": "simple"},
            scores={"scores": {}, "question_scores": {}},
            execution_time=10.0,
            final_score=0.75,
        )

    @pytest.fixture
    def sample_evaluation_data(self):
        """Create sample evaluation data."""
        return [
            EvaluationData(
                question="What is AI?",
                answer="AI is artificial intelligence.",
                contexts=["Context 1"],
                context_ids=["doc1"],
                ground_truths=["Answer 1"],
                question_id="q0",
                ground_truths_context_ids=["doc1"],
            )
        ]

    def test_add_evaluation_single(self, results, sample_evaluation_result, sample_evaluation_data, mocker):
        """Test adding a single evaluation."""
        mock_logger = mocker.patch("ai4rag.core.experiment.results.logger")

        results.add_evaluation(sample_evaluation_data, sample_evaluation_result)

        assert len(results.evaluations) == 1
        assert len(results.evaluation_data) == 1
        assert results.evaluations[0] == sample_evaluation_result
        assert results.evaluation_data[0] == sample_evaluation_data
        mock_logger.info.assert_called_once()

    def test_add_evaluation_multiple(self, results, mocker):
        """Test adding multiple evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")

        for i in range(3):
            eval_result = EvaluationResult(
                pattern_name=f"Pattern{i}",
                collection=f"collection_{i}",
                indexing_params={"chunk_size": 512},
                rag_params={"retrieval_method": "simple"},
                scores={"scores": {}, "question_scores": {}},
                execution_time=10.0,
                final_score=0.5 + i * 0.1,
            )
            eval_data = [
                EvaluationData(
                    question=f"Question {i}",
                    answer=f"Answer {i}",
                    contexts=[f"Context {i}"],
                    context_ids=[f"doc{i}"],
                    ground_truths=[f"Ground truth {i}"],
                    question_id=f"q{i}",
                    ground_truths_context_ids=[f"doc{i}"],
                )
            ]
            results.add_evaluation(eval_data, eval_result)

        assert len(results.evaluations) == 3
        assert len(results.evaluation_data) == 3

    def test_len_after_adding_evaluations(self, results, sample_evaluation_result, sample_evaluation_data, mocker):
        """Test __len__ after adding evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")

        results.add_evaluation(sample_evaluation_data, sample_evaluation_result)
        results.add_evaluation(sample_evaluation_data, sample_evaluation_result)

        assert len(results) == 2

    def test_bool_after_adding_evaluations(self, results, sample_evaluation_result, sample_evaluation_data, mocker):
        """Test __bool__ after adding evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")

        results.add_evaluation(sample_evaluation_data, sample_evaluation_result)

        assert results

    def test_iter_after_adding_evaluations(self, results, sample_evaluation_result, sample_evaluation_data, mocker):
        """Test __iter__ after adding evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")

        results.add_evaluation(sample_evaluation_data, sample_evaluation_result)
        results.add_evaluation(sample_evaluation_data, sample_evaluation_result)

        evaluations_list = list(results)
        assert len(evaluations_list) == 2
        assert all(isinstance(e, EvaluationResult) for e in evaluations_list)


class TestExperimentResultsEvaluationExploredOrCached:
    """Test suite for ExperimentResults.evaluation_explored_or_cached method."""

    @pytest.fixture
    def results_with_data(self, mocker):
        """Create ExperimentResults with some evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")
        results = ExperimentResults()

        for i in range(3):
            eval_result = EvaluationResult(
                pattern_name=f"Pattern{i}",
                collection=f"collection_{i}",
                indexing_params={"chunk_size": 512 + i * 100, "chunk_overlap": 128},
                rag_params={"retrieval_method": "simple", "number_of_chunks": 5 + i},
                scores={"scores": {}, "question_scores": {}},
                execution_time=10.0,
                final_score=0.5 + i * 0.1,
            )
            eval_data = []
            results.add_evaluation(eval_data, eval_result)

        return results

    def test_evaluation_explored_returns_score_when_found(self, results_with_data):
        """Test that evaluation_explored_or_cached returns score when params match."""
        indexing_params = {"chunk_size": 512, "chunk_overlap": 128}
        rag_params = {"retrieval_method": "simple", "number_of_chunks": 5}

        score = results_with_data.evaluation_explored_or_cached(indexing_params, rag_params)

        assert score == 0.5

    def test_evaluation_explored_returns_none_when_not_found(self, results_with_data):
        """Test that evaluation_explored_or_cached returns None when params don't match."""
        indexing_params = {"chunk_size": 999, "chunk_overlap": 999}
        rag_params = {"retrieval_method": "window", "number_of_chunks": 10}

        score = results_with_data.evaluation_explored_or_cached(indexing_params, rag_params)

        assert score is None

    def test_evaluation_explored_with_none_indexing_params(self, mocker):
        """Test evaluation_explored_or_cached with None indexing_params."""
        mocker.patch("ai4rag.core.experiment.results.logger")
        results = ExperimentResults()

        eval_result = EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params=None,
            rag_params={"retrieval_method": "simple", "number_of_chunks": 5},
            scores={"scores": {}, "question_scores": {}},
            execution_time=10.0,
            final_score=0.8,
        )
        results.add_evaluation([], eval_result)

        score = results.evaluation_explored_or_cached(None, {"retrieval_method": "simple", "number_of_chunks": 5})

        assert score == 0.8

    def test_evaluation_explored_partial_match_returns_none(self, results_with_data):
        """Test that partial parameter match returns None."""
        indexing_params = {"chunk_size": 512, "chunk_overlap": 128}
        rag_params = {"retrieval_method": "simple", "number_of_chunks": 999}  # Different

        score = results_with_data.evaluation_explored_or_cached(indexing_params, rag_params)

        assert score is None


class TestExperimentResultsCollectionExists:
    """Test suite for ExperimentResults.collection_exists method."""

    @pytest.fixture
    def results_with_collections(self, mocker):
        """Create ExperimentResults with different collections."""
        mocker.patch("ai4rag.core.experiment.results.logger")
        results = ExperimentResults()

        for i in range(3):
            eval_result = EvaluationResult(
                pattern_name=f"Pattern{i}",
                collection=f"collection_{i}",
                indexing_params={"chunk_size": 512 + i * 100, "chunk_overlap": 128},
                rag_params={"retrieval_method": "simple", "number_of_chunks": 5},
                scores={"scores": {}, "question_scores": {}},
                execution_time=10.0,
                final_score=0.5,
            )
            results.add_evaluation([], eval_result)

        return results

    def test_collection_exists_returns_name_when_found(self, results_with_collections):
        """Test that collection_exists returns collection name when indexing params match."""
        indexing_params = {"chunk_size": 512, "chunk_overlap": 128}

        collection = results_with_collections.get_existing_collection(indexing_params)

        assert collection == "collection_0"

    def test_collection_exists_returns_none_when_not_found(self, results_with_collections):
        """Test that collection_exists returns None when indexing params don't match."""
        indexing_params = {"chunk_size": 999, "chunk_overlap": 999}

        collection = results_with_collections.get_existing_collection(indexing_params)

        assert collection is None

    def test_collection_exists_with_empty_results(self):
        """Test collection_exists on empty results."""
        results = ExperimentResults()
        indexing_params = {"chunk_size": 512, "chunk_overlap": 128}

        collection = results.get_existing_collection(indexing_params)

        assert collection is None


class TestExperimentResultsProperties:
    """Test suite for ExperimentResults properties."""

    @pytest.fixture
    def results_with_data(self, mocker):
        """Create ExperimentResults with some evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")
        results = ExperimentResults()

        # Add evaluations with some duplicate collections
        for i in range(5):
            eval_result = EvaluationResult(
                pattern_name=f"Pattern{i}",
                collection=f"collection_{i % 3}",  # Will create 3 unique collections
                indexing_params={"chunk_size": 512},
                rag_params={"retrieval_method": "simple", "number_of_chunks": 5},
                scores={"scores": {}, "question_scores": {}},
                execution_time=10.0 + i,
                final_score=0.5 + i * 0.05,
            )
            results.add_evaluation([], eval_result)

        return results

    def test_collection_names_property(self, results_with_data):
        """Test collection_names property returns unique collection names."""
        collection_names = results_with_data.collection_names

        assert isinstance(collection_names, list)
        assert len(collection_names) == 3
        assert set(collection_names) == {"collection_0", "collection_1", "collection_2"}

    def test_collection_names_on_empty_results(self):
        """Test collection_names on empty results."""
        results = ExperimentResults()

        assert results.collection_names == []


class TestExperimentResultsSorted:
    """Test suite for ExperimentResults.sorted_ method."""

    @pytest.fixture
    def unsorted_results(self, mocker):
        """Create ExperimentResults with unsorted evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")
        results = ExperimentResults()

        scores = [0.5, 0.8, 0.3, 0.9, 0.6]
        for i, score in enumerate(scores):
            eval_result = EvaluationResult(
                pattern_name=f"Pattern{i}",
                collection=f"collection_{i}",
                indexing_params={"chunk_size": 512},
                rag_params={"retrieval_method": "simple", "number_of_chunks": 5},
                scores={"scores": {}, "question_scores": {}},
                execution_time=10.0,
                final_score=score,
            )
            results.add_evaluation([], eval_result)

        return results


class TestExperimentResultsGetBestEvaluations:
    """Test suite for ExperimentResults.get_best_evaluations method."""

    @pytest.fixture
    def results_with_data(self, mocker):
        """Create ExperimentResults with evaluations."""
        mocker.patch("ai4rag.core.experiment.results.logger")
        results = ExperimentResults()

        scores = [0.5, 0.8, 0.3, 0.9, 0.6]
        for i, score in enumerate(scores):
            eval_result = EvaluationResult(
                pattern_name=f"Pattern{i}",
                collection=f"collection_{i}",
                indexing_params={"chunk_size": 512},
                rag_params={"retrieval_method": "simple", "number_of_chunks": 5},
                scores={"scores": {}, "question_scores": {}},
                execution_time=10.0,
                final_score=score,
            )
            results.add_evaluation([], eval_result)

        return results

    def test_get_best_evaluations_with_k(self, results_with_data):
        """Test get_best_evaluations with specific k value."""
        best = results_with_data.get_best_evaluations(k=3)

        assert isinstance(best, tuple)
        assert len(best) == 3
        assert [ev.final_score for ev in best] == [0.9, 0.8, 0.6]

    def test_get_best_evaluations_with_k_none(self, results_with_data):
        """Test get_best_evaluations with k=None returns all."""
        best = results_with_data.get_best_evaluations(k=None)

        assert len(best) == 5
        assert [ev.final_score for ev in best] == [0.9, 0.8, 0.6, 0.5, 0.3]

    def test_get_best_evaluations_k_larger_than_available(self, results_with_data):
        """Test get_best_evaluations when k is larger than available evaluations."""
        best = results_with_data.get_best_evaluations(k=10)

        assert len(best) == 5

    def test_get_best_evaluations_k_zero(self, results_with_data):
        """Test get_best_evaluations with k=0."""
        best = results_with_data.get_best_evaluations(k=0)

        assert best == ()

    def test_get_best_evaluations_on_empty_results(self):
        """Test get_best_evaluations on empty results."""
        results = ExperimentResults()

        best = results.get_best_evaluations(k=5)

        assert best == ()


class TestExperimentResultsCreateEvaluationResultsJson:
    """Test suite for ExperimentResults.create_evaluation_results_json static method."""

    @pytest.fixture
    def evaluation_data_list(self):
        """Create sample evaluation data list."""
        return [
            EvaluationData(
                question="What is AI?",
                answer="AI is artificial intelligence.",
                contexts=["Context about AI", "More AI context"],
                context_ids=["doc1", "doc2"],
                ground_truths=["AI is artificial intelligence"],
                question_id="q0",
                ground_truths_context_ids=["doc1"],
            ),
            EvaluationData(
                question="What is ML?",
                answer="ML is machine learning.",
                contexts=["ML context", "Another ML context"],
                context_ids=["doc3", "doc4"],
                ground_truths=["ML is machine learning"],
                question_id="q1",
                ground_truths_context_ids=["doc3"],
            ),
        ]

    @pytest.fixture
    def evaluation_result(self):
        """Create sample evaluation result."""
        return EvaluationResult(
            pattern_name="Pattern1",
            collection="collection_1",
            indexing_params={"chunk_size": 512},
            rag_params={"retrieval_method": "simple"},
            scores={
                "scores": {
                    "answer_correctness": {"mean": 0.75},
                    "context_correctness": {"mean": 0.80},
                },
                "question_scores": {
                    "answer_correctness": {"q0": 0.70, "q1": 0.80},
                    "context_correctness": {"q0": 0.85, "q1": 0.75},
                },
            },
            execution_time=10.0,
            final_score=0.77,
        )

    def test_create_evaluation_results_json_structure(self, evaluation_data_list, evaluation_result):
        """Test that create_evaluation_results_json returns correct structure."""
        result = ExperimentResults.create_evaluation_results_json(evaluation_data_list, evaluation_result)

        assert isinstance(result, list)
        assert len(result) == 2

    def test_create_evaluation_results_json_first_entry(self, evaluation_data_list, evaluation_result):
        """Test first entry in the results json."""
        result = ExperimentResults.create_evaluation_results_json(evaluation_data_list, evaluation_result)

        first_entry = result[0]
        assert first_entry["question"] == "What is AI?"
        assert first_entry["correct_answers"] == ["AI is artificial intelligence"]
        assert first_entry["answer"] == "AI is artificial intelligence."

    def test_create_evaluation_results_json_answer_contexts(self, evaluation_data_list, evaluation_result):
        """Test answer_contexts structure in results json."""
        result = ExperimentResults.create_evaluation_results_json(evaluation_data_list, evaluation_result)

        first_entry = result[0]
        assert len(first_entry["answer_contexts"]) == 2
        assert first_entry["answer_contexts"][0] == {"text": "Context about AI", "document_id": "doc1"}
        assert first_entry["answer_contexts"][1] == {"text": "More AI context", "document_id": "doc2"}

    def test_create_evaluation_results_json_scores(self, evaluation_data_list, evaluation_result):
        """Test scores structure in results json."""
        result = ExperimentResults.create_evaluation_results_json(evaluation_data_list, evaluation_result)

        first_entry = result[0]
        assert first_entry["scores"]["answer_correctness"] == 0.70
        assert first_entry["scores"]["context_correctness"] == 0.85

        second_entry = result[1]
        assert second_entry["scores"]["answer_correctness"] == 0.80
        assert second_entry["scores"]["context_correctness"] == 0.75

    def test_create_evaluation_results_json_all_fields(self, evaluation_data_list, evaluation_result):
        """Test that all expected fields are present."""
        result = ExperimentResults.create_evaluation_results_json(evaluation_data_list, evaluation_result)

        expected_fields = {"question", "correct_answers", "answer", "answer_contexts", "scores"}
        for entry in result:
            assert set(entry.keys()) == expected_fields

    def test_create_evaluation_results_json_empty_list(self, evaluation_result):
        """Test with empty evaluation data list."""
        result = ExperimentResults.create_evaluation_results_json([], evaluation_result)

        assert result == []
