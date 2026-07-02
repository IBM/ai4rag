# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import random
from copy import copy
from dataclasses import dataclass
from math import ceil
from typing import Any, Callable

import numpy as np
import pandas as pd
from pygam import LinearGAM
from sklearn.preprocessing import LabelEncoder

from ai4rag import logger
from ai4rag.core.hpo.base_optimizer import BaseOptimizer, FailedIterationError, OptimizationError, OptimizerSettings
from ai4rag.search_space.src.search_space import SearchSpace

__all__ = ["GAMOptSettings", "GAMOptimizer"]


@dataclass
class GAMOptSettings(OptimizerSettings):
    """
    Settings for the GAMOptimizer. For the detailed description
    of parameters for Generalized Additive Models, please see pygam
    documentation.

    Parameters
    ----------
    max_evals : int
        Maximum number of evaluations performed during optimization process.
    n_random_nodes : int, default=4
        Number of random configurations to evaluate before starting GAM iterations.
    evals_per_trial : int, default=1
        Number of configurations to evaluate per GAM iteration.
    random_state : int, default=64
        Inherited from OptimizerSettings. Controls shuffle order of initial
        random exploration phase. Does NOT control GAM model randomness
        (GAM training is deterministic).
    """

    n_random_nodes: int = 4
    evals_per_trial: int = 1


class GAMOptimizer(BaseOptimizer):
    """
    Optimizer based on Generalized Additive Models.
    Trained model is used to suggest next node in the search space
    for evaluation.

    Parameters
    ----------
    objective_function : Callable[[dict], float]
        Target function that will be used in every evaluation. Output of
        this function should be 'float', as this is the value for which algorithms
        try to optimize solution. Function should take dict filled with 'key: value' pairs
        that are 'argument: corresponding value'.

    search_space : SearchSpace
        Instance containing information about nodes in the solutions space that
        will be evaluated during the optimization.

    settings : GAMOptSettings
        Instance with settings required for configuring the optimization process.

    Attributes
    ----------
    evaluations : list[dict]
        Already evaluated hyperparameters combinations with corresponding score.

    max_iterations : int
        Validated maximum number of iterations during HPO.
    """

    def __init__(
        self,
        objective_function: Callable[[dict], float],
        search_space: SearchSpace,
        settings: GAMOptSettings,
        known_observations: list[dict] | None = None,
    ):
        super().__init__(objective_function, search_space, settings)
        self.settings = settings
        self.evaluations = []
        self._evaluated_combinations = []
        self._encoders_with_columns: list[tuple[str, LabelEncoder]] = []

        if known_observations:
            self._load_known_observations(known_observations)

        self.max_iterations = self.settings.max_evals

    @property
    def max_iterations(self) -> int:
        """Get max possible number of iterations for the HPO."""
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val: int) -> None:
        """Set maximum number of iterations that should be performed during HPO."""
        max_comb = self._search_space.max_combinations
        if val > max_comb:
            logger.info(
                (
                    "'max_number_of_rag_patterns' exceeded number of possible combinations: %s. "
                    "Setting 'max_number_of_rag_patterns' to: %s"
                ),
                max_comb,
                max_comb,
            )
            self._max_iterations = max_comb
        else:
            self._max_iterations = val

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
        self.evaluate_initial_random_nodes()

        iterations_limit = self._get_iterations_limit()

        for _ in range(iterations_limit):
            self._run_iteration()

        successful_evaluations = [evaluation for evaluation in self.evaluations if evaluation["score"] is not None]
        if not successful_evaluations:
            raise OptimizationError("Number of evaluations has reached limit. All iterations have failed.")

        # Sort in ascending order and take the last element (highest score).
        # This assumes we're maximizing the score.
        best_config_with_score = sorted(successful_evaluations, key=lambda d: d["score"])[-1]

        return best_config_with_score

    def _get_iterations_limit(self) -> int:
        """
        Calculate maximum number of iterations that can be proceeded based on the
        already evaluated random nodes and settings for the optimizer.
        """
        iterations_limit = ceil((self.max_iterations - len(self.evaluations)) / self.settings.evals_per_trial)
        return iterations_limit

    def _load_known_observations(self, known_observations: list[dict]) -> None:
        """
        Load known observations to warm-start the optimizer.

        Parameters
        ----------
        known_observations : list[dict]
            List of previously evaluated parameter combinations with scores.
            Each dict must contain the same keys as search space combinations
            plus a "score" key.

        Raises
        ------
        ValueError
            When any observation is missing the "score" key.
        """
        for idx, obs in enumerate(known_observations):
            if "score" not in obs:
                raise ValueError(f"Known observation at index {idx} is missing the 'score' key.")

            params = {k: v for k, v in obs.items() if k != "score"}
            self._evaluated_combinations.append(params)
            self.evaluations.append(obs.copy())

        logger.info("Loaded %d known observations into the optimizer.", len(known_observations))

    def evaluate_initial_random_nodes(self) -> None:
        """
        Perform evaluation of randomly chosen n nodes from the solutions space.
        Evaluations are performed until desired number of successful evaluations
        is reached or maximum number of evaluations is reached.

        When the optimizer has been warm-started with known observations,
        already-successful evaluations count toward the n_random_nodes target
        and already-evaluated combinations are excluded from candidates.
        """
        successful_evaluations = sum(1 for e in self.evaluations if e["score"] is not None)

        if successful_evaluations >= self.settings.n_random_nodes:
            logger.info(
                "Skipping random evaluation phase: %d known successful evaluations >= n_random_nodes (%d).",
                successful_evaluations,
                self.settings.n_random_nodes,
            )
            return

        if len(self.evaluations) >= self.max_iterations:
            return

        combinations_local = [c for c in copy(self._search_space.combinations) if c not in self._evaluated_combinations]
        random.Random(self.settings.random_state).shuffle(combinations_local)

        gen = (x for x in combinations_local)

        while successful_evaluations < self.settings.n_random_nodes:
            params = next(gen)
            score = self._objective_function(params=params)
            if score is not None:
                successful_evaluations += 1
            self._evaluated_combinations.append(params)
            params_with_score = params | {"score": score}
            self.evaluations.append(params_with_score)

            if len(self.evaluations) == self.max_iterations:
                break

    # pylint: disable=too-many-locals
    def _run_iteration(self) -> None:
        """
        Run single optimization iteration that consists of training GAM model
        to predict score for remaining nodes in the solutions space and choose
        the best n ones for further evaluation.
        """
        self._prepare_encoder()
        df = pd.DataFrame(data=self.evaluations)  # --> These are already known observations with scores.
        df = df[df["score"].notna()].copy()
        data = df.drop(columns=["score"])
        target = df["score"]

        x_train_enc = []
        for column, encoder in self._encoders_with_columns:
            x_train_enc.append(encoder.transform(data[column]))
        x_train_enc = np.column_stack(x_train_enc)

        gam = LinearGAM()
        gam.fit(x_train_enc, target)

        remaining_evaluations = self._get_remaining_evaluations(
            self._search_space.combinations, self._evaluated_combinations
        )

        remaining_evaluations_df = pd.DataFrame(remaining_evaluations)

        # Optimize encoding: build array directly
        encoded_data_to_predict = np.column_stack(
            [encoder.transform(remaining_evaluations_df[column]) for column, encoder in self._encoders_with_columns]
        )

        predictions = gam.predict(encoded_data_to_predict)

        for idx, val in enumerate(remaining_evaluations):
            val["score"] = predictions[idx]

        # Sort in descending order to get highest predictions first
        best_predictions = sorted(remaining_evaluations, key=lambda d: d["score"], reverse=True)

        n_best_predictions = best_predictions[: self.settings.evals_per_trial]

        for params in n_best_predictions:
            params.pop("score", None)
            score = self._objective_function(params)
            self._evaluated_combinations.append(params)
            self.evaluations.append(params | {"score": score})

    def _prepare_encoder(self) -> None:
        """
        Prepare encoder for the further processing based on all available combinations.
        """
        if not self._encoders_with_columns:
            logger.debug("Preparing encoder for %s...", self.__class__.__name__)
            df = pd.DataFrame(data=self._search_space.combinations)
            for column in df.columns:
                self._encoders_with_columns.append((column, LabelEncoder().fit(df[column])))
            logger.debug("Encoder for %s has been prepared.", self.__class__.__name__)

    @staticmethod
    def _get_remaining_evaluations(all_combinations: list[dict], evaluations: list[dict]) -> list[dict]:
        """
        Get all evaluations that has not been yet proceeded.

        Parameters
        ----------
        all_combinations : list[dict]
            All possible combinations of parameters.

        evaluations : list[dict]
            Combinations that have already been evaluated.

        Returns
        -------
        list[dict]
            Remaining combinations that have not yet been evaluated.
        """
        remaining = []

        for ev in all_combinations:
            if ev not in evaluations:
                remaining.append(ev.copy())

        return remaining

    # pylint: disable=duplicate-code
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
