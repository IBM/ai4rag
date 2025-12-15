from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, Any

from ai4rag.search_space.src.search_space import SearchSpace

__all__ = ["BaseOptimiser", "OptimiserSettings", "OptimisationError", "FailedIterationError"]


class OptimisationError(Exception):
    """Custom class representing exception occurring in the Optimiser."""


class FailedIterationError(Exception):
    """Error used to signalize failed iteration in the experiment."""


@dataclass
class OptimiserSettings:
    """
    Representation of the general Optimiser Settings.

    Parameters
    ----------
    max_evals : int
        Maximum number of evaluations performed during optimisation process.

    Methods
    -------
    to_dict()
        Cast all the dataclass into the dictionary
    """

    max_evals: int

    def to_dict(self) -> dict:
        """
        Cast settings to dictionary

        Returns
        -------
        dict
            Dictionary representation of the settings class.
        """
        return asdict(self)


class BaseOptimiser(ABC):
    """
    Abstract class defining interface of Optimiser used in AI4RAGExperiment

    Parameters
    ----------
    objective_function : Callable[[dict], float]
        Target function that will be used in every evaluation. Output of
        this function should be 'float', as this is the value that algorithms
        try to minimize. Function should take dict filled with key: value pairs
        that are argument: corresponding value.

    search_space : SearchSpace
        List of parameters that algorithm will optimize.

    settings : OptimiserSettings
        Instance holding all the settings needed for the user

    Methods
    -------
    search()
        Perform hyperparameter optimisation to find what point in search space
        gives the minimal value for the objective function.
    """

    def __init__(
        self,
        objective_function: Callable[[dict], float],
        search_space: SearchSpace,
        settings: OptimiserSettings,
    ):
        self.objective_function = objective_function
        self._search_space = search_space
        self.settings = settings

    @abstractmethod
    def search(self) -> dict[str, Any]:
        """Start process of exploring parameters space to find the best combination and the target value."""
