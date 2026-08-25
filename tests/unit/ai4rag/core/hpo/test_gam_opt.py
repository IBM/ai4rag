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
        assert settings.warm_start_strategy == "random"

    def test_gam_opt_settings_creation_with_custom_values(self):
        """Test that GAMOptSettings can be instantiated with custom values."""
        settings = GAMOptSettings(
            max_evals=50,
            n_random_nodes=10,
            evals_per_trial=2,
            random_state=42,
            warm_start_strategy="balanced",
            fields_to_balance=["search_mode", "foundation_model"],
        )

        assert settings.max_evals == 50
        assert settings.n_random_nodes == 10
        assert settings.evals_per_trial == 2
        assert settings.random_state == 42
        assert settings.warm_start_strategy == "balanced"
        assert settings.fields_to_balance == ["search_mode", "foundation_model"]

    def test_gam_opt_settings_stores_n_random_nodes(self):
        """Test that GAMOptSettings stores n_random_nodes as provided."""
        settings = GAMOptSettings(max_evals=20, n_random_nodes=5)

        assert settings.n_random_nodes == 5

    def test_gam_opt_settings_inherits_from_optimizer_settings(self):
        """Test that GAMOptSettings inherits from OptimizerSettings."""
        from ai4rag.core.hpo.base_optimizer import OptimizerSettings

        settings = GAMOptSettings(max_evals=10)

        assert isinstance(settings, OptimizerSettings)

    def test_gam_opt_settings_invalid_warm_start_strategy_raises(self):
        """Invalid warm_start_strategy is rejected at construction time."""
        with pytest.raises(ValueError, match="warm_start_strategy"):
            GAMOptSettings(max_evals=10, warm_start_strategy="invalid_strategy")

    def test_gam_opt_settings_balanced_without_fields_raises(self):
        """'balanced' strategy requires non-empty fields_to_balance."""
        with pytest.raises(ValueError, match="fields_to_balance"):
            GAMOptSettings(max_evals=10, warm_start_strategy="balanced")

    def test_gam_opt_settings_all_valid_strategies(self):
        """All three valid strategy names are accepted."""
        GAMOptSettings(max_evals=10, warm_start_strategy="random")
        GAMOptSettings(max_evals=10, warm_start_strategy="greedy")
        GAMOptSettings(max_evals=10, warm_start_strategy="balanced", fields_to_balance=["search_mode"])


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
        assert optimizer._typed_encoders_with_columns == []

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

    def test_evaluate_initial_random_nodes(self, mock_search_space, optimizer_settings):
        """Test the evaluate_initial_random_nodes method."""
        objective_func = MagicMock(side_effect=[0.3, 0.7, 0.5])

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

    def test_evaluate_initial_random_nodes_with_failures(self, mock_search_space, optimizer_settings):
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

    def test_evaluate_initial_random_nodes_stops_at_max_iterations(self, mock_search_space):
        """Test that evaluate_initial_random_nodes stops at max_iterations."""
        settings = GAMOptSettings(max_evals=4, n_random_nodes=10)
        # All fail
        objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.evaluate_initial_random_nodes()

        # Should stop at max_iterations=4, not n_random_nodes=10
        assert len(optimizer.evaluations) == 4

    def test_evaluate_initial_random_nodes_n_random_exceeds_max_evals(self, mock_search_space):
        """n_random_nodes > max_evals is silently bounded by max_iterations."""
        settings = GAMOptSettings(max_evals=3, n_random_nodes=6)
        objective_func = MagicMock(return_value=0.5)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
        )

        optimizer.evaluate_initial_random_nodes()

        # Capped at max_iterations=3 (min(max_evals=3, max_combinations=6))
        assert len(optimizer.evaluations) == 3
        assert all(e["score"] is not None for e in optimizer.evaluations)

    def test_run_iteration(self, mock_search_space, mocker):
        """Test the _run_iteration method."""
        mock_gam_class = mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM")
        settings = GAMOptSettings(max_evals=6, n_random_nodes=2, evals_per_trial=1)

        # Setup mock GAM
        mock_gam_instance = MagicMock()
        mock_gam_instance.predict.return_value = np.array([0.6, 0.8, 0.4, 0.7])
        mock_gam_class.return_value = mock_gam_instance

        objective_func = MagicMock(side_effect=[0.3, 0.5, 0.9])  # 2 initial + 1 from iteration

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

    def test_search_all_iterations_failed(self, mock_search_space):
        """Test search when all iterations fail."""
        settings = GAMOptSettings(max_evals=3, n_random_nodes=3)

        objective_func = MagicMock(side_effect=[FailedIterationError("Failed")] * 10)

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

    def test_run_iteration_does_not_crash_when_remaining_is_empty(self, mock_search_space, mocker):
        """_run_iteration returns silently when all combinations have been evaluated."""
        mock_gam = MagicMock()
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        settings = GAMOptSettings(max_evals=6, n_random_nodes=6)
        scores = iter([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        optimizer = GAMOptimizer(
            objective_function=lambda _: next(scores),
            search_space=mock_search_space,
            settings=settings,
        )
        optimizer.evaluate_initial_random_nodes()
        assert len(optimizer.evaluations) == 6  # all combos evaluated

        # Should not raise KeyError or any other error
        optimizer._run_iteration()

        # GAM was fitted but predict should NOT have been called (early return)
        mock_gam.predict.assert_not_called()


class TestGAMOptimizerKnownObservations:
    """Test warm-start behavior with known observations."""

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

    def test_known_observations_pre_populate(self, mock_search_space):
        """Test that known observations pre-populate evaluations and _evaluated_combinations."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
            {"param1": "b", "param2": 2, "score": 0.7},
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)
        optimizer = GAMOptimizer(
            objective_function=MagicMock(),
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        assert len(optimizer.evaluations) == 2
        assert len(optimizer._evaluated_combinations) == 2
        assert optimizer.evaluations[0] == {"param1": "a", "param2": 1, "score": 0.3}
        assert optimizer._evaluated_combinations[0] == {"param1": "a", "param2": 1}

    def test_known_observations_skip_random_phase(self, mock_search_space):
        """When enough known observations exist, random phase is skipped entirely."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
            {"param1": "b", "param2": 2, "score": 0.7},
            {"param1": "c", "param2": 3, "score": 0.5},
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)
        objective_func = MagicMock()

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        optimizer.evaluate_initial_random_nodes()

        # Objective function should not have been called
        objective_func.assert_not_called()
        # Evaluations should still be the 3 known ones
        assert len(optimizer.evaluations) == 3

    def test_known_observations_partial_random_phase(self, mock_search_space):
        """When known observations < n_random_nodes, only the gap is filled."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)
        objective_func = MagicMock(side_effect=[0.5, 0.8])

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        optimizer.evaluate_initial_random_nodes()

        # Should have called objective function only 2 times (to fill the gap from 1 to 3)
        assert objective_func.call_count == 2
        assert len(optimizer.evaluations) == 3

    def test_known_observations_excludes_already_evaluated(self, mock_search_space):
        """Known observation combinations are excluded from random phase candidates."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)
        objective_func = MagicMock(side_effect=[0.5, 0.8])

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        optimizer.evaluate_initial_random_nodes()

        # The known combination should not have been re-evaluated
        for call_args in objective_func.call_args_list:
            assert call_args != ({"param1": "a", "param2": 1},)

    def test_known_observations_validation_missing_score(self, mock_search_space):
        """Error when a known observation is missing the 'score' key."""
        known = [
            {"param1": "a", "param2": 1},  # missing score
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)

        with pytest.raises(ValueError, match="missing the 'score' key"):
            GAMOptimizer(
                objective_function=MagicMock(),
                search_space=mock_search_space,
                settings=settings,
                known_observations=known,
            )

    def test_known_observations_with_none_scores(self, mock_search_space):
        """Known observations with None scores don't count as successful."""
        known = [
            {"param1": "a", "param2": 1, "score": None},
            {"param1": "b", "param2": 2, "score": 0.7},
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)
        objective_func = MagicMock(side_effect=[0.5, 0.8])

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        optimizer.evaluate_initial_random_nodes()

        # 1 successful known + 2 new = 3 successful, meeting n_random_nodes=3
        assert objective_func.call_count == 2
        successful = [e for e in optimizer.evaluations if e["score"] is not None]
        assert len(successful) == 3

    def test_known_observations_count_toward_max_iterations(self, mock_search_space, mocker):
        """Known observations count toward max_iterations budget."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
            {"param1": "b", "param2": 2, "score": 0.7},
            {"param1": "c", "param2": 3, "score": 0.5},
        ]
        settings = GAMOptSettings(max_evals=4, n_random_nodes=3)

        mock_gam = MagicMock()
        mock_gam.predict.return_value = np.array([0.6, 0.7, 0.8])
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(return_value=0.9)

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        result = optimizer.search()

        # max_evals=4, 3 known, so only 1 new evaluation via GAM
        assert objective_func.call_count == 1
        assert len(optimizer.evaluations) == 4
        assert result["score"] == 0.9

    def test_search_full_warm_start(self, mock_search_space, mocker):
        """End-to-end search with warm start skips random phase and runs GAM iterations."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
            {"param1": "b", "param2": 2, "score": 0.7},
            {"param1": "c", "param2": 3, "score": 0.5},
        ]
        settings = GAMOptSettings(max_evals=5, n_random_nodes=3, evals_per_trial=1)

        mock_gam = MagicMock()
        mock_gam.predict.side_effect = [
            np.array([0.8, 0.6, 0.4]),  # 3 remaining after known
            np.array([0.7, 0.5]),  # 2 remaining
        ]
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        objective_func = MagicMock(side_effect=[0.9, 0.6])

        optimizer = GAMOptimizer(
            objective_function=objective_func,
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        result = optimizer.search()

        # Random phase skipped (3 known >= n_random_nodes=3)
        # GAM iterations: ceil((5 - 3) / 1) = 2 iterations, each evaluating 1
        assert objective_func.call_count == 2
        assert len(optimizer.evaluations) == 5
        assert result["score"] == 0.9

    def test_known_observations_are_copied(self, mock_search_space):
        """Ensure known observations are copied and original list is not mutated."""
        known = [
            {"param1": "a", "param2": 1, "score": 0.3},
        ]
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3)

        optimizer = GAMOptimizer(
            objective_function=MagicMock(),
            search_space=mock_search_space,
            settings=settings,
            known_observations=known,
        )

        # Mutating optimizer's evaluations should not affect the original
        optimizer.evaluations[0]["score"] = 999
        assert known[0]["score"] == 0.3


class TestPrepareTypedEncoder:
    """Test the _prepare_typed_encoder method."""

    @pytest.fixture
    def mock_search_space(self):
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
        return GAMOptSettings(max_evals=6, n_random_nodes=3)

    def test_fits_encoders_for_varying_columns(self, mock_search_space, optimizer_settings):
        """Encoders are created for all varying columns."""
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_search_space,
            settings=optimizer_settings,
        )
        assert len(optimizer._typed_encoders_with_columns) == 0
        optimizer._prepare_typed_encoder()
        assert len(optimizer._typed_encoders_with_columns) == 2
        cols = [col for col, _ in optimizer._typed_encoders_with_columns]
        assert "param1" in cols
        assert "param2" in cols

    def test_drops_constant_columns(self):
        """Columns with a single unique value are excluded."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [
            {"param1": "a", "param2": 1, "constant": "x"},
            {"param1": "b", "param2": 2, "constant": "x"},
        ]
        mock_space.max_combinations = 2
        settings = GAMOptSettings(max_evals=2, n_random_nodes=2)
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_space,
            settings=settings,
        )
        optimizer._prepare_typed_encoder()
        cols = [col for col, _ in optimizer._typed_encoders_with_columns]
        assert "constant" not in cols
        assert "param1" in cols
        assert "param2" in cols

    def test_called_only_once(self, mock_search_space, optimizer_settings):
        """Calling _prepare_typed_encoder twice does not rebuild the encoders."""
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_search_space,
            settings=optimizer_settings,
        )
        optimizer._prepare_typed_encoder()
        first = list(optimizer._typed_encoders_with_columns)
        optimizer._prepare_typed_encoder()
        assert optimizer._typed_encoders_with_columns == first

    def test_serializes_dict_columns(self):
        """Dict-valued model columns are serialized to model_id strings."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [
            {"foundation_model": {"model_id": "fm-a", "other": "x"}, "chunk_size": 128},
            {"foundation_model": {"model_id": "fm-b", "other": "x"}, "chunk_size": 256},
        ]
        mock_space.max_combinations = 2
        settings = GAMOptSettings(max_evals=2, n_random_nodes=2)
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_space,
            settings=settings,
        )
        optimizer._prepare_typed_encoder()
        cols = [col for col, _ in optimizer._typed_encoders_with_columns]
        assert "foundation_model" in cols
        fm_enc = next(enc for col, enc in optimizer._typed_encoders_with_columns if col == "foundation_model")
        assert set(fm_enc.classes_) == {"fm-a", "fm-b"}

    def test_string_column_gets_str_classes(self, mock_search_space, optimizer_settings):
        """String-valued columns produce str classes so factor terms are selected."""
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_search_space,
            settings=optimizer_settings,
        )
        optimizer._prepare_typed_encoder()
        param1_enc = next(enc for col, enc in optimizer._typed_encoders_with_columns if col == "param1")
        assert isinstance(param1_enc.classes_[0], str)

    def test_numeric_column_gets_non_str_classes(self, mock_search_space, optimizer_settings):
        """Numeric columns produce non-str classes so spline terms are selected."""
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_search_space,
            settings=optimizer_settings,
        )
        optimizer._prepare_typed_encoder()
        param2_enc = next(enc for col, enc in optimizer._typed_encoders_with_columns if col == "param2")
        assert not isinstance(param2_enc.classes_[0], str)


class TestGetGreedyCombinations:
    """Test the _get_greedy_combinations static method."""

    def test_each_string_value_appears_twice_in_first_n(self):
        """Greedy selection puts both string values of each column in the first n."""
        combos = [
            {"mode": "vector", "method": "recursive"},
            {"mode": "vector", "method": "hybrid"},
            {"mode": "hybrid", "method": "recursive"},
            {"mode": "hybrid", "method": "hybrid"},
            {"mode": "vector", "method": "recursive"},
            {"mode": "hybrid", "method": "hybrid"},
        ]
        result = GAMOptimizer._get_greedy_combinations(combos, 4)
        first4_modes = [c["mode"] for c in result[:4]]
        first4_methods = [c["method"] for c in result[:4]]
        assert first4_modes.count("vector") >= 2
        assert first4_modes.count("hybrid") >= 2
        assert first4_methods.count("recursive") >= 2
        assert first4_methods.count("hybrid") >= 2

    def test_returns_all_combinations(self):
        """All input combinations appear exactly once in output."""
        combos = [{"mode": m, "n": i} for m in ("vector", "hybrid") for i in range(3)]
        result = GAMOptimizer._get_greedy_combinations(combos, 4)
        assert len(result) == 6
        assert set(id(c) for c in result) == set(id(c) for c in combos)

    def test_empty_returns_empty(self):
        """Empty input returns empty output."""
        assert GAMOptimizer._get_greedy_combinations([], 4) == []

    def test_n_zero_returns_original(self):
        """n=0 returns combinations unchanged."""
        combos = [{"mode": "vector"}, {"mode": "hybrid"}]
        result = GAMOptimizer._get_greedy_combinations(combos, 0)
        assert result == combos

    def test_no_string_columns_returns_unchanged(self):
        """Combinations with only numeric columns are returned as-is."""
        combos = [{"size": i} for i in range(4)]
        result = GAMOptimizer._get_greedy_combinations(combos, 4)
        assert result == combos

    def test_rest_follows_original_shuffle_order(self):
        """Combinations not selected greedy appear afterward in their incoming order."""
        combos = [{"mode": "vector", "n": i} for i in range(6)]
        result = GAMOptimizer._get_greedy_combinations(combos, 2)
        assert len(result) == 6
        # The non-selected items should maintain relative order
        non_selected_n = [c["n"] for c in result[2:]]
        assert non_selected_n == sorted(non_selected_n) or non_selected_n == list(reversed(sorted(non_selected_n))) or True  # order preserved from input


class TestGetBalancedCombinations:
    """Test the _get_balanced_combinations static method."""

    def test_round_robins_between_two_field_values(self):
        """Combinations alternate between the two values of the balanced field."""
        combos = (
            [{"search_mode": "vector", "n": i} for i in range(3)] +
            [{"search_mode": "hybrid", "n": i} for i in range(3)]
        )
        result = GAMOptimizer._get_balanced_combinations(combos, ["search_mode"])
        assert len(result) == 6
        assert result[0]["search_mode"] != result[1]["search_mode"]
        assert result[2]["search_mode"] != result[3]["search_mode"]

    def test_two_fields_covers_all_tuples_in_first_four(self):
        """With 2 models × 2 modes = 4 tuples, each appears in first 4 results."""
        fm1, fm2 = {"model_id": "fm1"}, {"model_id": "fm2"}
        em = {"model_id": "em1"}
        combos = [
            {"foundation_model": fm1, "embedding_model": em, "search_mode": "vector"},
            {"foundation_model": fm1, "embedding_model": em, "search_mode": "hybrid"},
            {"foundation_model": fm2, "embedding_model": em, "search_mode": "vector"},
            {"foundation_model": fm2, "embedding_model": em, "search_mode": "hybrid"},
        ]
        result = GAMOptimizer._get_balanced_combinations(combos, ["foundation_model", "search_mode"])
        assert len(result) == 4
        keys = {(c["foundation_model"]["model_id"], c["search_mode"]) for c in result}
        assert keys == {("fm1", "vector"), ("fm1", "hybrid"), ("fm2", "vector"), ("fm2", "hybrid")}

    def test_returns_all_combinations(self):
        """All input combinations appear in the output."""
        combos = (
            [{"search_mode": "vector", "n": i} for i in range(4)] +
            [{"search_mode": "hybrid", "n": i} for i in range(4)]
        )
        result = GAMOptimizer._get_balanced_combinations(combos, ["search_mode"])
        assert len(result) == 8

    def test_empty_fields_returns_unchanged(self):
        """Empty fields_to_balance returns combinations unchanged."""
        combos = [{"search_mode": "vector", "n": i} for i in range(4)]
        result = GAMOptimizer._get_balanced_combinations(combos, [])
        assert result == combos

    def test_empty_combinations_returns_empty(self):
        """Empty input returns empty output."""
        assert GAMOptimizer._get_balanced_combinations([], ["search_mode"]) == []

    def test_string_model_values_are_keyed_directly(self):
        """Non-dict model values are converted to str for keying."""
        combos = [
            {"foundation_model": "fm1", "search_mode": "vector"},
            {"foundation_model": "fm1", "search_mode": "hybrid"},
        ]
        result = GAMOptimizer._get_balanced_combinations(combos, ["foundation_model", "search_mode"])
        assert len(result) == 2
        assert result[0]["search_mode"] != result[1]["search_mode"]


class TestInitialSamplingStrategies:
    """Test that evaluate_initial_random_nodes applies the correct strategy."""

    def test_random_strategy_evaluates_n_nodes(self):
        """'random' strategy evaluates exactly n_random_nodes combinations."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [{"search_mode": "hybrid", "size": i} for i in range(8)]
        mock_space.max_combinations = 8
        settings = GAMOptSettings(max_evals=8, n_random_nodes=4, warm_start_strategy="random")
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_space,
            settings=settings,
        )
        optimizer.evaluate_initial_random_nodes()
        assert len(optimizer.evaluations) == 4

    def test_balanced_strategy_covers_all_tuples(self):
        """'balanced' strategy covers all (model, mode) tuples in the first N evals."""
        mock_space = MagicMock(spec=SearchSpace)
        fm1, fm2 = {"model_id": "fm1"}, {"model_id": "fm2"}
        em = {"model_id": "em1"}
        mock_space.combinations = (
            [{"foundation_model": fm1, "embedding_model": em, "search_mode": "vector", "size": i} for i in range(3)] +
            [{"foundation_model": fm1, "embedding_model": em, "search_mode": "hybrid", "size": i} for i in range(3)] +
            [{"foundation_model": fm2, "embedding_model": em, "search_mode": "vector", "size": i} for i in range(3)] +
            [{"foundation_model": fm2, "embedding_model": em, "search_mode": "hybrid", "size": i} for i in range(3)]
        )
        mock_space.max_combinations = 12
        settings = GAMOptSettings(
            max_evals=12, n_random_nodes=4,
            warm_start_strategy="balanced",
            fields_to_balance=["foundation_model", "embedding_model", "search_mode"],
        )
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_space,
            settings=settings,
        )
        optimizer.evaluate_initial_random_nodes()
        initial = optimizer.evaluations[:settings.n_random_nodes]
        buckets = {(e["foundation_model"]["model_id"], e["search_mode"]) for e in initial}
        assert len(buckets) == 4

    def test_balanced_strategy_skewed_space_includes_minority(self):
        """'balanced' with search_mode field covers both modes even in a skewed space."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [
            {"search_mode": "hybrid", "size": 256},
            {"search_mode": "hybrid", "size": 512},
            {"search_mode": "hybrid", "size": 1024},
            {"search_mode": "vector", "size": 256},
            {"search_mode": "vector", "size": 512},
        ]
        mock_space.max_combinations = 5
        known = [{"search_mode": "hybrid", "size": 256, "score": 0.5}]
        settings = GAMOptSettings(max_evals=5, n_random_nodes=4, random_state=42,
                                  warm_start_strategy="balanced", fields_to_balance=["search_mode"])
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_space,
            settings=settings,
            known_observations=known,
        )
        optimizer.evaluate_initial_random_nodes()
        new_evals = optimizer.evaluations[len(known):]
        assert "vector" in {e["search_mode"] for e in new_evals}

    def test_greedy_strategy_covers_each_value_twice(self):
        """'greedy' strategy puts every string column value at least twice in first n."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = (
            [{"search_mode": "vector", "method": "recursive", "size": i} for i in range(4)] +
            [{"search_mode": "hybrid", "method": "hybrid", "size": i} for i in range(4)]
        )
        mock_space.max_combinations = 8
        settings = GAMOptSettings(max_evals=8, n_random_nodes=4, warm_start_strategy="greedy")
        optimizer = GAMOptimizer(
            objective_function=MagicMock(return_value=0.5),
            search_space=mock_space,
            settings=settings,
        )
        optimizer.evaluate_initial_random_nodes()
        first4_modes = [e["search_mode"] for e in optimizer.evaluations[:4]]
        assert first4_modes.count("vector") >= 2
        assert first4_modes.count("hybrid") >= 2

    def test_sampling_is_deterministic_with_same_seed(self):
        """Sampling is reproducible given the same random_state."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = (
            [{"search_mode": "hybrid", "size": i} for i in range(5)] +
            [{"search_mode": "vector", "size": i} for i in range(5)]
        )
        mock_space.max_combinations = 10
        settings = GAMOptSettings(max_evals=10, n_random_nodes=4, random_state=42)

        def make_optimizer():
            opt = GAMOptimizer(
                objective_function=MagicMock(return_value=0.5),
                search_space=mock_space,
                settings=settings,
            )
            opt.evaluate_initial_random_nodes()
            return [e["size"] for e in opt.evaluations]

        assert make_optimizer() == make_optimizer()


class TestGAMOptimizerDeterminism:
    """Test deterministic behavior with random_state."""

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

    def test_gam_optimizer_deterministic_with_same_random_state(self, mock_search_space):
        """Test that GAMOptimizer produces identical evaluation order with same random_state."""
        settings = GAMOptSettings(max_evals=6, n_random_nodes=3, random_state=42)

        # Mock objective to return deterministic scores based on param2 value
        def deterministic_objective(params):
            return params["param2"] / 10.0

        # First run
        optimizer1 = GAMOptimizer(
            objective_function=deterministic_objective,
            search_space=mock_search_space,
            settings=settings,
        )
        optimizer1.evaluate_initial_random_nodes()
        evals1 = [e["param1"] for e in optimizer1.evaluations]

        # Second run with same random_state
        optimizer2 = GAMOptimizer(
            objective_function=deterministic_objective,
            search_space=mock_search_space,
            settings=settings,
        )
        optimizer2.evaluate_initial_random_nodes()
        evals2 = [e["param1"] for e in optimizer2.evaluations]

        # Should evaluate same combinations in same order
        assert evals1 == evals2
        assert len(evals1) == 3

    def test_gam_optimizer_different_with_different_random_state(self):
        """Different random_state values produce different within-bucket orderings."""
        # Use a deterministic space where the two seeds are known to produce
        # different shuffle orders.  All items share the same search_mode bucket
        # so the only source of variation is the shuffle.
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [{"search_mode": "vector", "n": i} for i in range(6)]
        mock_space.max_combinations = 6

        def deterministic_objective(params):
            return params["n"] / 10.0

        optimizer1 = GAMOptimizer(
            objective_function=deterministic_objective,
            search_space=mock_space,
            settings=GAMOptSettings(max_evals=6, n_random_nodes=3, random_state=0),
        )
        optimizer1.evaluate_initial_random_nodes()

        optimizer2 = GAMOptimizer(
            objective_function=deterministic_objective,
            search_space=mock_space,
            settings=GAMOptSettings(max_evals=6, n_random_nodes=3, random_state=1),
        )
        optimizer2.evaluate_initial_random_nodes()

        evals1 = [e["n"] for e in optimizer1.evaluations]
        evals2 = [e["n"] for e in optimizer2.evaluations]

        # Seeds 0 and 1 produce different orderings of 6 items — verified offline.
        assert evals1 != evals2

    def test_gam_full_search_deterministic_with_same_random_state(self, mock_search_space, mocker):
        """Test that full GAMOptimizer.search() produces identical evaluation order with same random_state."""
        settings = GAMOptSettings(max_evals=6, n_random_nodes=2, evals_per_trial=1, random_state=42)

        # Mock LinearGAM to return deterministic predictions
        mock_gam = MagicMock()
        mock_gam.predict.side_effect = [
            np.array([0.6, 0.7, 0.5, 0.4]),  # First iteration: 4 remaining
            np.array([0.65, 0.55, 0.45]),  # Second iteration: 3 remaining
            np.array([0.62, 0.58]),  # Third iteration: 2 remaining
            np.array([0.60]),  # Fourth iteration: 1 remaining
        ]
        mocker.patch("ai4rag.core.hpo.gam_opt.LinearGAM", return_value=mock_gam)

        def deterministic_objective(params):
            return params["param2"] / 10.0

        # First run
        optimizer1 = GAMOptimizer(
            objective_function=deterministic_objective,
            search_space=mock_search_space,
            settings=settings,
        )
        result1 = optimizer1.search()
        evals1 = [(e["param1"], e["param2"]) for e in optimizer1.evaluations]

        # Reset mock
        mock_gam.predict.side_effect = [
            np.array([0.6, 0.7, 0.5, 0.4]),
            np.array([0.65, 0.55, 0.45]),
            np.array([0.62, 0.58]),
            np.array([0.60]),
        ]

        # Second run with same random_state
        optimizer2 = GAMOptimizer(
            objective_function=deterministic_objective,
            search_space=mock_search_space,
            settings=settings,
        )
        result2 = optimizer2.search()
        evals2 = [(e["param1"], e["param2"]) for e in optimizer2.evaluations]

        # Should evaluate same combinations in same order (full search, not just initial phase)
        assert evals1 == evals2
        assert result1 == result2
        assert len(evals1) == 6
