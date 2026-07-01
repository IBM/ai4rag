# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import random
from dataclasses import dataclass
from typing import Any, Callable

from ai4rag import logger
from ai4rag.core.hpo.base_optimizer import BaseOptimizer, FailedIterationError, OptimizationError, OptimizerSettings
from ai4rag.search_space.src.search_space import SearchSpace

__all__ = ["RandomOptimizer", "RandomOptSettings", "FailedIterationError"]


@dataclass
class RandomOptSettings(OptimizerSettings):
    """Settings for random optimizer."""

    random_state: int = 64


class RandomOptimizer(BaseOptimizer):
    """Optimizer running random search on the given search space."""

    def __init__(
        self, objective_function: Callable[[dict], float], search_space: SearchSpace, settings: RandomOptSettings
    ):
        super().__init__(objective_function, search_space, settings)
        self._evaluated_combinations = []
        self._rng = random.Random(self.settings.random_state)

    def search(self) -> dict[str, Any]:
        """
        Actual function performing hyperparameter optimization for the selected
        objective function.

        Returns
        -------
        dict[str, Any]
            The best set of parameters with achieved score.

        Raises
        ------
        OptimizationError
            When there were no successful evaluations for given constraints.
        """
        combinations = list(self._search_space.combinations)
        self._rng.shuffle(combinations)

        for idx in range(self.settings.max_evals):
            score = self._objective_function(combinations[idx])
            self._evaluated_combinations.append(combinations[idx] | {"score": score})

        successful_evaluations = [ev for ev in self._evaluated_combinations if ev["score"] is not None]

        if not successful_evaluations:
            raise OptimizationError("Number of evaluations has reached limit. All iterations have failed.")

        best_config_with_score = sorted(successful_evaluations, key=lambda d: d["score"])[-1]

        return best_config_with_score

    def _objective_function(self, params: dict) -> float | None:
        """
        Wrapper around the objective function provided to the optimizer.

        Parameters
        ----------
        params : dict
            A dictionary containing parameters of pattern to be evaluated.

        Returns
        -------
        float | None
            Optimization score achieved for single node evaluation.
            If None - iteration has ended up with a failed status.
        """

        try:
            logger.info("Evaluating objective function with parameters: %s", params)
            loss = self.objective_function(params)

        except FailedIterationError:
            # None is here to avoid penalization of iterations failing due to unknown reasons
            loss = None

        return loss
