#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#
from dataclasses import dataclass
import random
from typing import Callable, Any

from ai4rag import logger
from ai4rag.core.hpo.base_optimiser import OptimiserSettings, BaseOptimiser, OptimisationError, FailedIterationError
from ai4rag.search_space.src.search_space import SearchSpace


__all__ = ["RandomOptimiser", "RandomOptSettings", "FailedIterationError"]

@dataclass
class RandomOptSettings(OptimiserSettings):
    """Settings for random optimiser."""


class RandomOptimiser(BaseOptimiser):
    """Optimiser running random search on the given search space."""
    def __init__(
        self,
        objective_function: Callable[[dict], float],
        search_space: SearchSpace,
        settings: RandomOptSettings
    ):
        super().__init__(objective_function, search_space, settings)
        self._evaluated_combinations = []

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
        OptimisationError
            When there were no successful evaluations for given constraints.
        """
        combinations = self._search_space.combinations
        random.shuffle(combinations)

        for idx in range(self.settings.max_evals):
            score = self._objective_function(combinations[idx])
            self._evaluated_combinations.append(
                combinations[idx] | {"score": score}
            )

        successful_evaluations = [ev for ev in self._evaluated_combinations if ev["score"] is not None]

        if not successful_evaluations:
            raise OptimisationError("Number of evaluations has reached limit. All iterations have failed.")

        best_config_with_score = sorted(successful_evaluations, key=lambda d: d["score"])[-1]

        return best_config_with_score

    def _objective_function(self, params: dict) -> float | None:
        """
        Wrapper around the objective function provided to the optimiser.

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
