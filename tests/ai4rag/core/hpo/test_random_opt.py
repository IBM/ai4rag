# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock

from ai4rag.core.hpo.random_opt import RandomOptimiser, RandomOptSettings
from ai4rag.core.hpo.base_optimiser import OptimisationError, FailedIterationError
from ai4rag.search_space.src.search_space import SearchSpace


class TestRandomOptSettings:
    """Test the RandomOptSettings dataclass."""

    def test_random_opt_settings_creation(self):
        """Test that RandomOptSettings can be instantiated."""
        settings = RandomOptSettings(max_evals=15)

        assert settings.max_evals == 15

    def test_random_opt_settings_inherits_from_optimiser_settings(self):
        """Test that RandomOptSettings inherits from OptimiserSettings."""
        from ai4rag.core.hpo.base_optimiser import OptimiserSettings

        settings = RandomOptSettings(max_evals=10)

        assert isinstance(settings, OptimiserSettings)

    def test_random_opt_settings_to_dict(self):
        """Test the to_dict method inherited from OptimiserSettings."""
        settings = RandomOptSettings(max_evals=25)
        settings_dict = settings.to_dict()

        assert settings_dict == {"max_evals": 25}


class TestRandomOptimiser:
    """Test the RandomOptimiser class."""

    @pytest.fixture
    def mock_search_space(self):
        """Create a mock search space with predefined combinations."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [
            {"param1": 1, "param2": "a"},
            {"param1": 2, "param2": "b"},
            {"param1": 3, "param2": "c"},
            {"param1": 4, "param2": "d"},
            {"param1": 5, "param2": "e"},
        ]
        return mock_space

    @pytest.fixture
    def optimiser_settings(self):
        """Create RandomOptSettings."""
        return RandomOptSettings(max_evals=3)

    def test_random_optimiser_initialization(self, mock_search_space, optimiser_settings):
        """Test that RandomOptimiser initializes correctly."""
        objective_func = MagicMock(return_value=0.5)

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        assert optimiser.objective_function == objective_func
        assert optimiser._search_space == mock_search_space
        assert optimiser.settings == optimiser_settings
        assert optimiser._evaluated_combinations == []

    def test_search_successful_evaluations(self, mock_search_space, optimiser_settings, mocker):
        """Test the search method with successful evaluations."""
        # Mock the objective function to return different scores
        objective_func = MagicMock(side_effect=[0.3, 0.7, 0.5])

        # Mock random.shuffle to make the test deterministic
        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        result = optimiser.search()

        # Should return the combination with the highest score (0.7)
        assert result["param1"] == 2
        assert result["param2"] == "b"
        assert result["score"] == 0.7
        assert len(optimiser._evaluated_combinations) == 3

    def test_search_with_some_failed_iterations(self, mock_search_space, mocker):
        """Test search when some iterations fail but some succeed."""
        settings = RandomOptSettings(max_evals=4)

        # First and third succeed, second and fourth fail
        objective_func = MagicMock(side_effect=[0.3, FailedIterationError("Failed"), 0.8, FailedIterationError("Failed")])

        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimiser.search()

        # Should return the best successful evaluation (0.8)
        assert result["score"] == 0.8
        assert len(optimiser._evaluated_combinations) == 4

    def test_search_all_iterations_failed(self, mock_search_space, optimiser_settings, mocker):
        """Test search when all iterations fail."""
        # All evaluations fail
        objective_func = MagicMock(side_effect=[
            FailedIterationError("Failed 1"),
            FailedIterationError("Failed 2"),
            FailedIterationError("Failed 3"),
        ])

        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        with pytest.raises(OptimisationError) as exc_info:
            optimiser.search()

        assert "Number of evaluations has reached limit" in str(exc_info.value)
        assert "All iterations have failed" in str(exc_info.value)

    def test_objective_function_wrapper_success(self, mock_search_space, optimiser_settings):
        """Test the _objective_function wrapper with successful execution."""
        objective_func = MagicMock(return_value=0.42)

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        params = {"param1": 10, "param2": "test"}
        result = optimiser._objective_function(params)

        assert result == 0.42
        objective_func.assert_called_once_with(params)

    def test_objective_function_wrapper_catches_failed_iteration_error(
        self, mock_search_space, optimiser_settings
    ):
        """Test that _objective_function catches FailedIterationError and returns None."""
        objective_func = MagicMock(side_effect=FailedIterationError("Iteration failed"))

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        params = {"param1": 10, "param2": "test"}
        result = optimiser._objective_function(params)

        assert result is None
        objective_func.assert_called_once_with(params)

    def test_search_shuffles_combinations(self, mock_search_space, optimiser_settings, mocker):
        """Test that search shuffles combinations before evaluation."""
        objective_func = MagicMock(return_value=0.5)
        mock_shuffle = mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        optimiser.search()

        # Verify shuffle was called
        mock_shuffle.assert_called_once()

    def test_search_respects_max_evals(self, mock_search_space, mocker):
        """Test that search respects the max_evals setting."""
        settings = RandomOptSettings(max_evals=2)
        objective_func = MagicMock(return_value=0.5)

        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimiser.search()

        # Should only evaluate max_evals times
        assert objective_func.call_count == 2
        assert len(optimiser._evaluated_combinations) == 2

    def test_search_returns_highest_score(self, mock_search_space, mocker):
        """Test that search returns the configuration with the highest score."""
        settings = RandomOptSettings(max_evals=5)
        # Return different scores, highest should be 0.95
        objective_func = MagicMock(side_effect=[0.3, 0.95, 0.5, 0.2, 0.7])

        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimiser.search()

        assert result["score"] == 0.95
        assert result["param1"] == 2
        assert result["param2"] == "b"

    def test_evaluated_combinations_stores_all_evaluations(self, mock_search_space, optimiser_settings, mocker):
        """Test that _evaluated_combinations stores all evaluated parameters."""
        objective_func = MagicMock(side_effect=[0.3, 0.7, 0.5])

        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        optimiser.search()

        # Should store all evaluations with scores
        assert len(optimiser._evaluated_combinations) == 3
        for evaluation in optimiser._evaluated_combinations:
            assert "param1" in evaluation
            assert "param2" in evaluation
            assert "score" in evaluation

    def test_search_filters_out_none_scores(self, mock_search_space, mocker):
        """Test that search correctly filters out evaluations with None scores."""
        settings = RandomOptSettings(max_evals=4)
        # Mix of successful and failed evaluations
        objective_func = MagicMock(side_effect=[
            0.3,
            FailedIterationError("Failed"),
            0.8,
            FailedIterationError("Failed"),
        ])

        mocker.patch("ai4rag.core.hpo.random_opt.random.shuffle")

        optimiser = RandomOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimiser.search()

        # Should only consider successful evaluations (0.3 and 0.8)
        assert result["score"] == 0.8
        # But _evaluated_combinations should include all attempts
        assert len(optimiser._evaluated_combinations) == 4
