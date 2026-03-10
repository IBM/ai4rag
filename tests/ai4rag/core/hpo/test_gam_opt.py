# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

import numpy as np
import pytest

from ai4rag.core.hpo.base_optimizer import FailedIterationError, OptimizationError
from ai4rag.core.hpo.gam_opt import GAMOptimizer, GAMOptSettings
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

    def test_gam_opt_settings_inherits_from_optimizer_settings(self):
        """Test that GAMOptSettings inherits from OptimizerSettings."""
        from ai4rag.core.hpo.base_optimizer import OptimizerSettings

        settings = GAMOptSettings(max_evals=10)

        assert isinstance(settings, OptimizerSettings)


class TestGAMOptimizer:
    """Test the GAMOptimizer class."""

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
    def optimizer_settings(self):
        """Create GAMOptSettings."""
        return GAMOptSettings(max_evals=6, n_random_nodes=3)

    def test_gam_optimizer_initialization(self, mock_search_space, optimizer_settings):
        """Test that GAMOptimizer initializes correctly."""
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        assert optimizer.objective_function == objective_func
        assert optimizer._search_space == mock_search_space
        assert optimizer.settings == optimizer_settings
        assert optimizer.evaluations == []
        assert optimizer._evaluated_combinations == []
        assert optimizer._encoders_with_columns == []

    def test_max_iterations_getter(self, mock_search_space, optimizer_settings):
        """Test the max_iterations property getter."""
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        assert optimizer.max_iterations == 6

    def test_max_iterations_setter_within_limit(self, mock_search_space):
        """Test setting max_iterations when it's within the search space limit."""
        settings = GAMOptSettings(max_evals=4)
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.max_iterations = 3
        assert optimizer.max_iterations == 3

    def test_max_iterations_setter_exceeds_limit(self, mock_search_space):
        """Test setting max_iterations when it exceeds the search space combinations."""
        settings = GAMOptSettings(max_evals=10)
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        # Should be capped at max_combinations (6)
        assert optimizer.max_iterations == 6

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

        remaining = GAMOptimizer._get_remaining_evaluations(all_combinations, evaluated)

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

        remaining = GAMOptimizer._get_remaining_evaluations(all_combinations, evaluated)

        assert len(remaining) == 0

    def test_get_remaining_evaluations_none_evaluated(self):
        """Test _get_remaining_evaluations when no combinations are evaluated."""
        all_combinations = [
            {"param1": "a", "param2": 1},
            {"param1": "b", "param2": 2},
        ]

        evaluated = []

        remaining = GAMOptimizer._get_remaining_evaluations(all_combinations, evaluated)

        assert len(remaining) == 2

    def test_objective_function_wrapper_success(self, mock_search_space, optimizer_settings):
        """Test the _objective_function wrapper with successful execution."""
        objective_func = MagicMock(return_value=0.42)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        params = {"param1": "test", "param2": 10}
        result = optimizer._objective_function(params)

        assert result == 0.42
        objective_func.assert_called_once_with(params)

    def test_objective_function_wrapper_catches_failed_iteration_error(self, mock_search_space, optimizer_settings):
        """Test that _objective_function catches FailedIterationError and returns None."""
        objective_func = MagicMock(side_effect=FailedIterationError("Iteration failed"))

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        params = {"param1": "test", "param2": 10}
        result = optimizer._objective_function(params)

        assert result is None
        objective_func.assert_called_once_with(params)

    def test_get_iterations_limit(self, mock_search_space, optimizer_settings):
        """Test the _get_iterations_limit method."""
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        # With max_iterations=6, n_random_nodes=3, evals_per_trial=1
        # iterations_limit = ceil((6 - 0) / 1) = 6
        iterations_limit = optimizer._get_iterations_limit()
        assert iterations_limit == 6

        # After evaluating 3 random nodes
        optimizer.evaluations = [{"score": 0.5}] * 3
        iterations_limit = optimizer._get_iterations_limit()
        # ceil((6 - 3) / 1) = 3
        assert iterations_limit == 3

    def test_get_iterations_limit_with_multiple_evals_per_trial(self, mock_search_space):
        """Test _get_iterations_limit with evals_per_trial > 1."""
        settings = GAMOptSettings(max_evals=10, evals_per_trial=2)
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.evaluations = [{"score": 0.5}] * 4
        iterations_limit = optimizer._get_iterations_limit()
        # ceil((6 - 4) / 2) = ceil(1) = 1
        assert iterations_limit == 1

    def test_evaluate_initial_random_nodes(self, mock_search_space, optimizer_settings, mocker):
        """Test the evaluate_initial_random_nodes method."""
        objective_func = MagicMock(side_effect=[0.3, 0.7, 0.5])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        optimizer.evaluate_initial_random_nodes()

        # Should evaluate n_random_nodes=3 combinations
        assert len(optimizer.evaluations) == 3
        assert len(optimizer._evaluated_combinations) == 3
        assert objective_func.call_count == 3

        # Check that scores are stored
        for evaluation in optimizer.evaluations:
            assert "score" in evaluation

    def test_evaluate_initial_random_nodes_with_failures(self, mock_search_space, optimizer_settings, mocker):
        """Test evaluate_initial_random_nodes skips failed iterations (score=None) when counting successes."""
        # First fails (returns None), second succeeds, third fails (returns None), fourth succeeds, fifth succeeds
        objective_func = MagicMock(
            side_effect=[
                FailedIterationError("Failed"),
                0.7,
                FailedIterationError("Failed"),
                0.5,
                0.3,
            ]
        )
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        optimizer.evaluate_initial_random_nodes()

        # Failures (score=None) are NOT counted as successful, so we need 3 successes.
        # Sequence: fail(None), 0.7, fail(None), 0.5, 0.3 → 5 total evaluations, 3 successful
        assert len(optimizer.evaluations) == 5
        successful_evals = [e for e in optimizer.evaluations if e["score"] is not None]
        assert len(successful_evals) == 3
        failed_evals = [e for e in optimizer.evaluations if e["score"] is None]
        assert len(failed_evals) == 2

    def test_evaluate_initial_random_nodes_stops_at_max_iterations(self, mock_search_space, mocker):
        """Test that evaluate_initial_random_nodes stops at max_iterations."""
        settings = GAMOptSettings(max_evals=4, n_random_nodes=10)
        # All fail
        objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.evaluate_initial_random_nodes()

        # Should stop at max_iterations=4, not n_random_nodes=10
        assert len(optimizer.evaluations) == 4

    def test_prepare_encoder(self, mock_search_space, optimizer_settings, mocker):
        """Test the _prepare_encoder method."""
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        # Initially no encoders
        assert len(optimizer._encoders_with_columns) == 0

        optimizer._prepare_encoder()

        # Should have created encoders for each column
        assert len(optimizer._encoders_with_columns) == 2  # param1 and param2
        column_names = [col for col, enc in optimizer._encoders_with_columns]
        assert "param1" in column_names
        assert "param2" in column_names

    def test_prepare_encoder_called_only_once(self, mock_search_space, optimizer_settings, mocker):
        """Test that _prepare_encoder only prepares encoders once."""
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        optimizer._prepare_encoder()
        first_encoders = optimizer._encoders_with_columns.copy()

        # Call again
        optimizer._prepare_encoder()

        # Should not recreate encoders
        assert optimizer._encoders_with_columns == first_encoders

    def test_run_iteration(self, mock_search_space, mocker):
        """Test the _run_iteration method."""
        mock_gam_class = mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM")
        settings = GAMOptSettings(max_evals=6, n_random_nodes=2, evals_per_trial=1)

        # Setup mock GAM
        mock_gam_instance = MagicMock()
        mock_gam_instance.predict.return_value = np.array([0.6, 0.8, 0.4, 0.7])
        mock_gam_class.return_value = mock_gam_instance

        objective_func = MagicMock(side_effect=[0.3, 0.5, 0.9])  # 2 initial + 1 from iteration

        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        # First evaluate initial nodes
        optimizer.evaluate_initial_random_nodes()

        # Then run one iteration
        optimizer._run_iteration()

        # Should have evaluated one more combination
        assert len(optimizer.evaluations) == 3
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
            np.array([0.65, 0.55, 0.45]),  # Second iteration: 3 remaining
            np.array([0.62, 0.58]),  # Third iteration: 2 remaining
        ]
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(side_effect=[0.3, 0.5, 0.8, 0.6, 0.4])
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimizer.search()

        # Should return the best configuration
        assert "score" in result
        assert result["score"] == 0.8
        assert len(optimizer.evaluations) == 5

    def test_search_all_iterations_failed(self, mock_search_space, mocker):
        """Test search when all iterations fail."""
        settings = GAMOptSettings(max_evals=3, n_random_nodes=3)

        objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        with pytest.raises(OptimizationError) as exc_info:
            optimizer.search()

        assert "Number of evaluations has reached limit" in str(exc_info.value)
        assert "All iterations have failed" in str(exc_info.value)

    def test_search_with_some_failed_iterations(self, mock_search_space, mocker):
        """Test search with a mix of successful and failed iterations."""
        settings = GAMOptSettings(max_evals=5, n_random_nodes=2, evals_per_trial=1)

        # Mock LinearGAM
        # After initial random evals (0.3 success, fail None, need 1 more success),
        # we get 3 evals during random phase. Then 3 remaining for GAM iterations.
        # Note: None scores are filtered out before GAM training, so GAM sees only 2 samples.
        mock_gam = MagicMock()
        mock_gam.predict.side_effect = [
            np.array([0.6, 0.7, 0.5]),  # First iteration: 3 remaining
            np.array([0.65, 0.55]),  # Second iteration: 2 remaining
        ]
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(
            side_effect=[
                0.3,  # Initial random 1 (success)
                FailedIterationError("Failed"),  # Initial random 2 (returns None, not counted)
                0.5,  # Initial random 3 (success, reaches n_random_nodes=2)
                0.8,  # GAM iteration 1
                0.6,  # GAM iteration 2
            ]
        )
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        result = optimizer.search()

        # Should return the best successful evaluation (score is not None)
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

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.evaluate_initial_random_nodes()
        optimizer._run_iteration()

        # Should have evaluated evals_per_trial=2 more combinations
        assert len(optimizer.evaluations) == 4

    def test_run_iteration_filters_out_none_scores_for_training(self, mock_search_space, mocker):
        """Test that _run_iteration filters out None scores when training GAM."""
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3, evals_per_trial=1)

        mock_gam = MagicMock()
        mock_gam.predict.return_value = np.array([0.6, 0.7])
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        # Need 3 successful evals (score is not None). Failure doesn't count.
        # Sequence: 0.3 (ok), fail(None), 0.5 (ok), 0.8 (ok) → 4 total, 3 successful
        objective_func = MagicMock(
            side_effect=[
                0.3,
                FailedIterationError("Failed"),  # This will return score=None
                0.5,
                0.8,
                0.6,  # extra for _run_iteration
            ]
        )
        mocker.patch("ai4rag.core.hpo.gam_opt.random.shuffle")

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.evaluate_initial_random_nodes()

        # Should have 4 evaluations: [0.3, None, 0.5, 0.8] (3 successful + 1 failure)
        assert len(optimizer.evaluations) == 4
        assert optimizer.evaluations[0]["score"] == 0.3
        assert optimizer.evaluations[1]["score"] is None  # Failed iteration
        assert optimizer.evaluations[2]["score"] == 0.5
        assert optimizer.evaluations[3]["score"] == 0.8

        optimizer._run_iteration()

        # GAM should be trained only on non-None scores
        # The fit call should receive only 3 samples (those with non-None scores)
        call_args = mock_gam.fit.call_args
        assert call_args[0][0].shape[0] == 3  # X_train should have 3 samples
        assert call_args[0][1].shape[0] == 3  # y_train should have 3 samples
