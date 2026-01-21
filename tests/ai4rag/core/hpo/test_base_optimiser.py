# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock

from ai4rag.core.hpo.base_optimiser import (
    BaseOptimiser,
    OptimiserSettings,
    OptimisationError,
    FailedIterationError,
)
from ai4rag.search_space.src.search_space import SearchSpace


class TestOptimisationError:
    """Test the OptimisationError exception."""

    def test_optimisation_error_can_be_raised(self):
        """Test that OptimisationError can be raised and caught."""
        with pytest.raises(OptimisationError) as exc_info:
            raise OptimisationError("Test error message")

        assert str(exc_info.value) == "Test error message"

    def test_optimisation_error_is_exception(self):
        """Test that OptimisationError is an Exception subclass."""
        assert issubclass(OptimisationError, Exception)


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


class TestOptimiserSettings:
    """Test the OptimiserSettings dataclass."""

    def test_optimiser_settings_creation(self):
        """Test that OptimiserSettings can be instantiated."""
        settings = OptimiserSettings(max_evals=10)

        assert settings.max_evals == 10

    def test_optimiser_settings_to_dict(self):
        """Test the to_dict method of OptimiserSettings."""
        settings = OptimiserSettings(max_evals=20)
        settings_dict = settings.to_dict()

        assert settings_dict == {"max_evals": 20}
        assert isinstance(settings_dict, dict)


class TestBaseOptimiser:
    """Test the BaseOptimiser abstract class."""

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
    def optimiser_settings(self):
        """Create optimiser settings."""
        return OptimiserSettings(max_evals=10)

    def test_base_optimiser_cannot_be_instantiated(
        self, mock_objective_function, mock_search_space, optimiser_settings
    ):
        """Test that BaseOptimiser cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseOptimiser(
                objective_function=mock_objective_function,
                search_space=mock_search_space,
                settings=optimiser_settings,
            )

        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_base_optimiser_initialization_via_subclass(
        self, mock_objective_function, mock_search_space, optimiser_settings
    ):
        """Test that BaseOptimiser attributes are set correctly via a concrete subclass."""

        # Create a concrete implementation for testing
        class ConcreteOptimiser(BaseOptimiser):
            def search(self):
                return {"result": "test"}

        optimiser = ConcreteOptimiser(
            objective_function=mock_objective_function,
            search_space=mock_search_space,
            settings=optimiser_settings,
        )

        assert optimiser.objective_function == mock_objective_function
        assert optimiser._search_space == mock_search_space
        assert optimiser.settings == optimiser_settings

    def test_base_optimiser_search_method_is_abstract(self):
        """Test that the search method is abstract and must be implemented."""

        # Create a subclass without implementing search
        class IncompleteOptimiser(BaseOptimiser):
            pass

        with pytest.raises(TypeError) as exc_info:
            IncompleteOptimiser(
                objective_function=MagicMock(),
                search_space=MagicMock(spec=SearchSpace),
                settings=OptimiserSettings(max_evals=5),
            )

        assert "Can't instantiate abstract class" in str(exc_info.value)
