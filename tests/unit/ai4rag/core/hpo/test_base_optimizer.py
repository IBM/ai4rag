# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from ai4rag.core.hpo.base_optimizer import (
    BaseOptimizer,
    FailedIterationError,
    OptimizationError,
    OptimizerSettings,
)
from ai4rag.search_space.src.search_space import SearchSpace


class TestOptimizationError:
    """Test the OptimizationError exception."""

    def test_optimization_error_can_be_raised(self):
        """Test that OptimizationError can be raised and caught."""
        with pytest.raises(OptimizationError) as exc_info:
            raise OptimizationError("Test error message")

        assert str(exc_info.value) == "Test error message"

    def test_optimization_error_is_exception(self):
        """Test that OptimizationError is an Exception subclass."""
        assert issubclass(OptimizationError, Exception)


class TestFailedIterationError:
    """Test the FailedIterationError exception."""

    def test_failed_iteration_error_can_be_raised(self):
        """Test that FailedIterationError can be raised and caught."""
        with pytest.raises(FailedIterationError) as exc_info:
            raise FailedIterationError("Test iteration failed")

        assert str(exc_info.value) == "Test iteration failed"

    def test_failed_iteration_error_is_exception(self):
        """Test that FailedIterationError is an Exception subclass."""
        assert issubclass(FailedIterationError, Exception)


class TestOptimizerSettings:
    """Test the OptimizerSettings dataclass."""

    def test_optimizer_settings_creation(self):
        """Test that OptimizerSettings can be instantiated."""
        settings = OptimizerSettings(max_evals=10)

        assert settings.max_evals == 10

    def test_optimizer_settings_to_dict(self):
        """Test the to_dict method of OptimizerSettings."""
        settings = OptimizerSettings(max_evals=20)
        settings_dict = settings.to_dict()

        assert settings_dict == {"max_evals": 20}
        assert isinstance(settings_dict, dict)


class TestBaseOptimizer:
    """Test the BaseOptimizer abstract class."""

    @pytest.fixture
    def mock_objective_function(self):
        """Create a mock objective function."""
        return MagicMock(return_value=0.5)

    @pytest.fixture
    def mock_search_space(self):
        """Create a mock search space."""
        mock_space = MagicMock(spec=SearchSpace)
        mock_space.combinations = [
            {"param1": "value1", "param2": "value2"},
            {"param1": "value3", "param2": "value4"},
        ]
        return mock_space

    @pytest.fixture
    def optimizer_settings(self):
        """Create optimizer settings."""
        return OptimizerSettings(max_evals=10)

    def test_base_optimizer_cannot_be_instantiated(
        self, mock_objective_function, mock_search_space, optimizer_settings
    ):
        """Test that BaseOptimizer cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseOptimizer(
                objective_function=mock_objective_function,
                search_space=mock_search_space,
                settings=optimizer_settings,
            )

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_base_optimizer_initialization_via_subclass(
        self, mock_objective_function, mock_search_space, optimizer_settings
    ):
        """Test that BaseOptimizer attributes are set correctly via a concrete subclass."""

        # Create a concrete implementation for testing
        class ConcreteOptimizer(BaseOptimizer):
            def search(self):
                return {"result": "test"}

        optimizer = ConcreteOptimizer(
            objective_function=mock_objective_function,
            search_space=mock_search_space,
            settings=optimizer_settings,
        )

        assert optimizer.objective_function == mock_objective_function
        assert optimizer._search_space == mock_search_space
        assert optimizer.settings == optimizer_settings

    def test_base_optimizer_search_method_is_abstract(self):
        """Test that the search method is abstract and must be implemented."""

        # Create a subclass without implementing search
        class IncompleteOptimizer(BaseOptimizer):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteOptimizer(
                objective_function=MagicMock(),
                search_space=MagicMock(spec=SearchSpace),
                settings=OptimizerSettings(max_evals=5),
            )

        assert "Can't instantiate abstract class" in str(exc_info.value)
