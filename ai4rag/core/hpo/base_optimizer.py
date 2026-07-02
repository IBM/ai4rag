# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable

from ai4rag.search_space.src.search_space import SearchSpace

__all__ = ["BaseOptimizer", "OptimizerSettings", "OptimizationError", "FailedIterationError"]


class OptimizationError(Exception):
    """Custom class representing exception occurring in the Optimizer."""


class FailedIterationError(Exception):
    """Error used to signalize failed iteration in the experiment."""


@dataclass
class OptimizerSettings:
    """
    Representation of the general Optimizer Settings.

    Parameters
    ----------
    max_evals : int
        Maximum number of evaluations performed during optimization process.
    random_state : int, default=64
        Seed for random number generator controlling exploration order.
        Use the same seed across runs to get deterministic evaluation
        sequences (given deterministic objective function scores).

    Methods
    -------
    to_dict()
        Cast all the dataclass into the dictionary
    """

    max_evals: int
    random_state: int = 64

    def to_dict(self) -> dict:
        """
        Cast settings to dictionary

        Returns
        -------
        dict
            Dictionary representation of the settings class.
        """
        return asdict(self)


class BaseOptimizer(ABC):
    """
    Abstract class defining interface of Optimizer used in AI4RAGExperiment

    Parameters
    ----------
    objective_function : Callable[[dict], float]
        Target function that will be used in every evaluation. Output of
        this function should be 'float', as this is the value that algorithms
        try to minimize. Function should take dict filled with 'key: value' pairs
        that are 'argument: corresponding value'.

    search_space : SearchSpace
        List of parameters that algorithm will optimize.

    settings : OptimizerSettings
        Instance holding all the settings needed for the user

    Methods
    -------
    search()
        Perform hyperparameter optimization to find what point in search space
        gives the minimal value for the objective function.
    """

    def __init__(
        self,
        objective_function: Callable[[dict], float],
        search_space: SearchSpace,
        settings: OptimizerSettings,
    ):
        self.objective_function = objective_function
        self._search_space = search_space
        self.settings = settings

    @abstractmethod
    def search(self) -> dict[str, Any]:
        """Start process of exploring parameters space to find the best combination and the target value."""
