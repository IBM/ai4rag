# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from ai4rag.core.hpo.gam_opt import GAMOptimiser, GAMOptSettings
from ai4rag.core.hpo.base_optimiser import OptimisationError, FailedIterationError
from ai4rag.search_space.src.search_space import SearchSpace


class TestGAMOptSettings:
    """Test the GAMOptSettings dataclass."""

    def test_gam_opt_settings_creation_with_defaults(self):
        """Test that GAMOptSettings can be instantiated with default values."""
        settings = GAMOptSettings(max_evals=20)

        assert settings.max_evals == 20
        assert settings.n_random_nodes == 4
        assert settings.evals_per_trial == 1
        assert settings.random_state == 64

    def test_gam_opt_settings_creation_with_custom_values(self):
        """Test that GAMOptSettings can be instantiated with custom values."""
        settings = GAMOptSettings(
            max_evals=50,
            n_random_nodes=10,
            evals_per_trial=2,
            random_state=42,
        )

        assert settings.max_evals == 50
        assert settings.n_random_nodes == 10
        assert settings.evals_per_trial == 2
        assert settings.random_state == 42

    def test_gam_opt_settings_post_init_limits_n_random_nodes(self):
        """Test that __post_init__ limits n_random_nodes to max_evals."""
        settings = GAMOptSettings(max_evals=5, n_random_nodes=10)

        # n_random_nodes should be capped at max_evals
        assert settings.n_random_nodes == 5

    def test_gam_opt_settings_post_init_keeps_n_random_nodes_if_smaller(self):
        """Test that __post_init__ keeps n_random_nodes if it's smaller than max_evals."""
        settings = GAMOptSettings(max_evals=20, n_random_nodes=5)

        assert settings.n_random_nodes == 5

    def test_gam_opt_settings_inherits_from_optimiser_settings(self):
        """Test that GAMOptSettings inherits from OptimiserSettings."""
        from ai4rag.core.hpo.base_optimiser import OptimiserSettings

        settings = GAMOptSettings(max_evals=10)

        assert isinstance(settings, OptimiserSettings)


class TestGAMOptimiser:
    """Test the GAMOptimiser class."""

    @pytest.fixture
    def mock_search_space(self):
        """Create a mock search space with predefined combinations."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
            {"param1": "c", "param2": 3},
            {"param1": "d", "param2": 4},
            {"param1": "e", "param2": 5},
            {"param1": "f", "param2": 6},
        ]
        mock_space.max_combinations = 6
        return mock_space

    @pytest.fixture
    def optimiser_settings(self):
        """Create GAMOptSettings."""
        return GAMOptSettings(max_evals=6, n_random_nodes=3)

    def test_gam_optimiser_initialization(self, mock_search_space, optimiser_settings):
        """Test that GAMOptimiser initializes correctly."""
        objective_func = MagicMock(return_value=0.5)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        assert optimiser.objective_function == objective_func
        assert optimiser._search_space == mock_search_space
        assert optimiser.settings == optimiser_settings
        assert optimiser.evaluations == []
        assert optimiser._evaluated_combinations == []
        assert optimiser._encoders_with_columns == []

    def test_max_iterations_getter(self, mock_search_space, optimiser_settings):
        """Test the max_iterations property getter."""
        objective_func = MagicMock(return_value=0.5)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        assert optimiser.max_iterations == 6

    def test_max_iterations_setter_within_limit(self, mock_search_space):
        """Test setting max_iterations when it's within the search space limit."""
        settings = GAMOptSettings(max_evals=4)
        objective_func = MagicMock(return_value=0.5)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimiser.max_iterations = 3
        assert optimiser.max_iterations == 3

    def test_max_iterations_setter_exceeds_limit(self, mock_search_space):
        """Test setting max_iterations when it exceeds the search space combinations."""
        settings = GAMOptSettings(max_evals=10)
        objective_func = MagicMock(return_value=0.5)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        # Should be capped at max_combinations (6)
        assert optimiser.max_iterations == 6

    def test_get_remaining_evaluations(self):
        """Test the _get_remaining_evaluations static method."""
        all_combinations = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
            {"param1": "c", "param2": 3},
        ]

        evaluated = [
            {"param1": "a", "param2": 1},
        ]

        remaining = GAMOptimiser._get_remaining_evaluations(all_combinations, evaluated)

        assert len(remaining) == 2
        assert {"param1": "b", "param2": 2} in remaining
        assert {"param1": "c", "param2": 3} in remaining
        assert {"param1": "a", "param2": 1} not in remaining

    def test_get_remaining_evaluations_all_evaluated(self):
        """Test _get_remaining_evaluations when all combinations are evaluated."""
        all_combinations = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
        ]

        evaluated = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
        ]

        remaining = GAMOptimiser._get_remaining_evaluations(all_combinations, evaluated)

        assert len(remaining) == 0

    def test_get_remaining_evaluations_none_evaluated(self):
        """Test _get_remaining_evaluations when no combinations are evaluated."""
        all_combinations = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
        ]

        evaluated = []

        remaining = GAMOptimiser._get_remaining_evaluations(all_combinations, evaluated)

        assert len(remaining) == 2

    def test_objective_function_wrapper_success(self, mock_search_space, optimiser_settings):
        """Test the _objective_function wrapper with successful execution."""
        objective_func = MagicMock(return_value=0.42)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        params = {"param1": "test", "param2": 10}
        result = optimiser._objective_function(params)

        assert result == 0.42
        objective_func.assert_called_once_with(params)

    def test_objective_function_wrapper_catches_failed_iteration_error(
        self, mock_search_space, optimiser_settings
    ):
        """Test that _objective_function catches FailedIterationError and returns None."""
        objective_func = MagicMock(side_effect=FailedIterationError("Iteration failed"))

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        params = {"param1": "test", "param2": 10}
        result = optimiser._objective_function(params)

        assert result is None
        objective_func.assert_called_once_with(params)

    def test_get_iterations_limit(self, mock_search_space, optimiser_settings):
        """Test the _get_iterations_limit method."""
        objective_func = MagicMock(return_value=0.5)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        # With max_iterations=6, n_random_nodes=3, evals_per_trial=1
        # iterations_limit = ceil((6 - 0) / 1) = 6
        iterations_limit = optimiser._get_iterations_limit()
        assert iterations_limit == 6

        # After evaluating 3 random nodes
        optimiser.evaluations = [{"score": 0.5}] * 3
        iterations_limit = optimiser._get_iterations_limit()
        # ceil((6 - 3) / 1) = 3
        assert iterations_limit == 3

    def test_get_iterations_limit_with_multiple_evals_per_trial(self, mock_search_space):
        """Test _get_iterations_limit with evals_per_trial > 1."""
        settings = GAMOptSettings(max_evals=10, evals_per_trial=2)
        objective_func = MagicMock(return_value=0.5)

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimiser.evaluations = [{"score": 0.5}] * 4
        iterations_limit = optimiser._get_iterations_limit()
        # ceil((6 - 4) / 2) = ceil(1) = 1
        assert iterations_limit == 1

    def test_evaluate_initial_random_nodes(self, mock_search_space, optimiser_settings, mocker):
        """Test the evaluate_initial_random_nodes method."""
        objective_func = MagicMock(side_effect=[0.3, 0.7, 0.5])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        optimiser.evaluate_initial_random_nodes()

        # Should evaluate n_random_nodes=3 combinations
        assert len(optimiser.evaluations) == 3
        assert len(optimiser._evaluated_combinations) == 3
        assert objective_func.call_count == 3

        # Check that scores are stored
        for evaluation in optimiser.evaluations:
            assert "score" in evaluation

    def test_evaluate_initial_random_nodes_with_failures(self, mock_search_space, optimiser_settings, mocker):
        """Test evaluate_initial_random_nodes with some failed iterations."""
        # First fails, second succeeds, third succeeds
        objective_func = MagicMock(side_effect=[
            FailedIterationError("Failed"),
            0.7,
            FailedIterationError("Failed"),
            0.5,
            0.3,
        ])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        optimiser.evaluate_initial_random_nodes()

        # Should continue until n_random_nodes=3 successful evaluations
        # or until max_iterations is reached
        successful_evals = [e for e in optimiser.evaluations if e["score"] is not None]
        assert len(successful_evals) == 3
        assert len(optimiser.evaluations) >= 3

    def test_evaluate_initial_random_nodes_stops_at_max_iterations(self, mock_search_space, mocker):
        """Test that evaluate_initial_random_nodes stops at max_iterations."""
        settings = GAMOptSettings(max_evals=4, n_random_nodes=10)
        # All fail
        objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimiser.evaluate_initial_random_nodes()

        # Should stop at max_iterations=4, not n_random_nodes=10
        assert len(optimiser.evaluations) == 4

    def test_prepare_encoder(self, mock_search_space, optimiser_settings, mocker):
        """Test the _prepare_encoder method."""
        objective_func = MagicMock(return_value=0.5)

        # Mock handle_missing_values to return dataframe as is
        mock_handle_missing = mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        # Initially no encoders
        assert len(optimiser._encoders_with_columns) == 0

        optimiser._prepare_encoder()

        # Should have created encoders for each column
        assert len(optimiser._encoders_with_columns) == 2  # param1 and param2
        column_names = [col for col, enc in optimiser._encoders_with_columns]
        assert "param1" in column_names
        assert "param2" in column_names

    def test_prepare_encoder_called_only_once(self, mock_search_space, optimiser_settings, mocker):
        """Test that _prepare_encoder only prepares encoders once."""
        objective_func = MagicMock(return_value=0.5)

        mock_handle_missing = mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        optimiser._prepare_encoder()
        first_encoders = optimiser._encoders_with_columns.copy()

        # Call again
        optimiser._prepare_encoder()

        # Should not recreate encoders
        assert optimiser._encoders_with_columns == first_encoders

    @patch("ai4rag.core.hpo.gam_opt.LinearGAM")
    def test_run_iteration(self, mock_gam_class, mock_search_space, mocker):
        """Test the _run_iteration method."""
        settings = GAMOptSettings(max_evals=6, n_random_nodes=2, evals_per_trial=1)

        # Setup mock GAM
        mock_gam_instance = MagicMock()
        mock_gam_instance.predict.return_value = np.array([0.6, 0.8, 0.4, 0.7])
        mock_gam_class.return_value = mock_gam_instance

        objective_func = MagicMock(side_effect=[0.3, 0.5, 0.9])  # 2 initial + 1 from iteration

        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")
        mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        # First evaluate initial nodes
        optimiser.evaluate_initial_random_nodes()

        # Then run one iteration
        optimiser._run_iteration()

        # Should have evaluated one more combination
        assert len(optimiser.evaluations) == 3
        assert mock_gam_instance.fit.called
        assert mock_gam_instance.predict.called

    def test_search_successful(self, mock_search_space, mocker):
        """Test the search method with successful optimization."""
        settings = GAMOptSettings(max_evals=5, n_random_nodes=2, evals_per_trial=1)

        # Mock LinearGAM - need to return predictions for remaining combinations
        # After 2 initial random nodes, there will be 4 remaining combinations
        # After iteration 1: 3 remaining, after iteration 2: 2 remaining, after iteration 3: 1 remaining
        mock_gam = MagicMock()
        mock_gam.predict.side_effect = [
            np.array([0.6, 0.7, 0.5, 0.4]),  # First iteration: 4 remaining
            np.array([0.65, 0.55, 0.45]),     # Second iteration: 3 remaining
            np.array([0.62, 0.58]),           # Third iteration: 2 remaining
        ]
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(side_effect=[0.3, 0.5, 0.8, 0.6, 0.4])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")
        mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimiser.search()

        # Should return the best configuration
        assert "score" in result
        assert result["score"] == 0.8
        assert len(optimiser.evaluations) == 5

    def test_search_all_iterations_failed(self, mock_search_space, mocker):
        """Test search when all iterations fail."""
        settings = GAMOptSettings(max_evals=3, n_random_nodes=3)

        objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        with pytest.raises(OptimisationError) as exc_info:
            optimiser.search()

        assert "Number of evaluations has reached limit" in str(exc_info.value)
        assert "All iterations have failed" in str(exc_info.value)

    def test_search_with_some_failed_iterations(self, mock_search_space, mocker):
        """Test search with a mix of successful and failed iterations."""
        settings = GAMOptSettings(max_evals=5, n_random_nodes=2, evals_per_trial=1)

        # Mock LinearGAM
        mock_gam = MagicMock()
        mock_gam.predict.return_value = np.array([0.6, 0.7, 0.5])
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(side_effect=[
            0.3,
            FailedIterationError("Failed"),
            0.5,
            0.8,
            FailedIterationError("Failed"),
            0.6,
        ])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")
        mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimiser.search()

        # Should return best successful evaluation
        assert result["score"] == 0.8

    def test_run_iteration_evaluates_best_predictions(self, mock_search_space, mocker):
        """Test that _run_iteration evaluates the best predicted combinations."""
        settings = GAMOptSettings(max_evals=6, n_random_nodes=2, evals_per_trial=2)

        # Mock LinearGAM to return predictions
        mock_gam = MagicMock()
        # Predictions for remaining combinations (should be 4 remaining)
        mock_gam.predict.return_value = np.array([0.4, 0.9, 0.3, 0.7])
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(side_effect=[0.3, 0.5, 0.8, 0.6])  # 2 initial + 2 from iteration
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")
        mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimiser.evaluate_initial_random_nodes()
        optimiser._run_iteration()

        # Should have evaluated evals_per_trial=2 more combinations
        assert len(optimiser.evaluations) == 4

    def test_run_iteration_filters_out_nan_scores_for_training(self, mock_search_space, mocker):
        """Test that _run_iteration filters out NaN scores when training GAM."""
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3, evals_per_trial=1)

        mock_gam = MagicMock()
        mock_gam.predict.return_value = np.array([0.6, 0.7, 0.5])
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        # Mix of successful and failed evaluations
        objective_func = MagicMock(side_effect=[
            0.3,
            FailedIterationError("Failed"),
            0.5,
            0.8,
        ])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")
        mocker.patch(
            "ai4rag.core.hpo.gam_opt.handle_missing_values_in_combinations_being_explored",
            side_effect=lambda df: df,
        )

        optimiser = GAMOptimiser(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        # Manually set evaluations with some None scores
        optimiser.evaluations = [
            {"param1": "a", "param2": 1, "score": 0.3},
            {"param1": "b", "param2": 2, "score": None},
            {"param1": "c", "param2": 3, "score": 0.5},
        ]
        optimiser._evaluated_combinations = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
            {"param1": "c", "param2": 3},
        ]

        optimiser._run_iteration()

        # GAM should be trained only on non-None scores
        # The fit call should receive only 2 samples (those with non-None scores)
        call_args = mock_gam.fit.call_args
        assert call_args[0][0].shape[0] == 2  # X_train should have 2 samples
        assert call_args[0][1].shape[0] == 2  # y_train should have 2 samples
