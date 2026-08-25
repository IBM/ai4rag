# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import random
from collections import defaultdict, deque
from copy import copy
from dataclasses import dataclass
from math import ceil
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from pygam import LinearGAM
from pygam import f as gam_f
from pygam import s as gam_s
from sklearn.preprocessing import LabelEncoder

from ai4rag import logger
from ai4rag.core.hpo.base_optimizer import BaseOptimizer, FailedIterationError, OptimizationError, OptimizerSettings
from ai4rag.search_space.src.search_space import SearchSpace

__all__ = ["GAMOptSettings", "GAMOptimizer"]


def _serialize_dict_col(series: pd.Series) -> pd.Series:
    """Serialize dict-valued cells to their model_id string (or str(x) fallback)."""
    if series.apply(lambda x: isinstance(x, dict)).any():
        return series.apply(lambda x: x.get("model_id", str(x)) if isinstance(x, dict) else x)
    return series


def _round_robin(combinations: list[dict], key_fn: Callable[[dict], Any]) -> list[dict]:
    """Re-order combinations by round-robin across buckets determined by key_fn."""
    buckets: dict[Any, deque] = defaultdict(deque)
    for c in combinations:
        buckets[key_fn(c)].append(c)
    bucket_list = list(buckets.values())
    balanced: list[dict] = []
    i = 0
    while bucket_list:
        idx = i % len(bucket_list)
        bucket = bucket_list[idx]
        if bucket:
            balanced.append(bucket.popleft())
            i += 1
        else:
            bucket_list.pop(idx)
    return balanced


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
        Selection is balanced across search_mode values (mode_balanced strategy) or
        across (foundation_model, embedding_model, search_mode) triples
        (model_mode_balanced strategy), ensuring the GAM receives representative
        signal on the most impactful categorical dimensions before training begins.
    evals_per_trial : int, default=1
        Number of configurations to evaluate per GAM iteration.
    warm_start_strategy : {"mode_balanced", "model_mode_balanced"}, default="mode_balanced"
        Controls how the initial n_random_nodes observations are ordered.
        "mode_balanced"        — round-robin across search_mode values.
        "model_mode_balanced"  — round-robin across (foundation_model,
                                 embedding_model, search_mode) triples.
    random_state : int, default=64
        Inherited from OptimizerSettings. Controls shuffle order of initial
        random exploration phase. Does NOT control GAM model randomness
        (GAM training is deterministic).
    """

    n_random_nodes: int = 4
    evals_per_trial: int = 1
    warm_start_strategy: Literal["mode_balanced", "model_mode_balanced"] = "mode_balanced"

    def __post_init__(self) -> None:
        valid = {"mode_balanced", "model_mode_balanced"}
        if self.warm_start_strategy not in valid:
            raise ValueError(
                f"warm_start_strategy must be one of {sorted(valid)}; "
                f"got {self.warm_start_strategy!r}."
            )


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
        self._typed_encoders_with_columns: list[tuple[str, LabelEncoder]] = []

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

        The selection is balanced: combinations are ordered by round-robin across
        search_mode values (mode_balanced) or across (foundation_model,
        embedding_model, search_mode) triples (model_mode_balanced), ensuring
        the GAM receives representative signal on the most impactful categorical
        dimensions before training begins.
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

        if self.settings.warm_start_strategy == "model_mode_balanced":
            combinations_local = self._get_model_mode_balanced_combinations(combinations_local)
        else:  # "mode_balanced"
            combinations_local = self._get_mode_balanced_combinations(combinations_local)

        modes_in_space = {c.get("search_mode") for c in combinations_local}
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

        modes_covered = {e.get("search_mode") for e in self.evaluations if e.get("score") is not None}
        uncovered = modes_in_space - modes_covered
        if uncovered:
            logger.warning(
                "n_random_nodes=%d was too small to cover all search_mode values. "
                "Uncovered modes: %s. Consider increasing n_random_nodes.",
                self.settings.n_random_nodes,
                sorted(str(m) for m in uncovered),
            )

    @staticmethod
    def _get_mode_balanced_combinations(combinations: list[dict]) -> list[dict]:
        """Order combinations by round-robin across search_mode values."""
        return _round_robin(combinations, lambda c: c.get("search_mode", None))

    @staticmethod
    def _get_model_mode_balanced_combinations(combinations: list[dict]) -> list[dict]:
        """Order combinations by round-robin across (foundation_model, embedding_model, search_mode) triples."""
        def _model_id(v: object) -> str:
            return v.get("model_id", str(v)) if isinstance(v, dict) else str(v)

        def _key(c: dict) -> tuple:
            return (
                _model_id(c.get("foundation_model", None)),
                _model_id(c.get("embedding_model", None)),
                c.get("search_mode", None),
            )

        return _round_robin(combinations, _key)

    def _prepare_typed_encoder(self) -> None:
        """
        Fit label encoders on the full search space for all varying columns.

        Dict-valued columns (model objects) are serialized to their model_id
        strings. Constant columns (single unique value) are dropped — they
        carry no signal for the GAM.
        """
        if self._typed_encoders_with_columns:
            return
        logger.debug("Preparing typed encoder for %s...", self.__class__.__name__)
        df = pd.DataFrame(data=self._search_space.combinations)
        for col in df.columns:
            df[col] = _serialize_dict_col(df[col])
        varying_cols = [c for c in df.columns if df[c].nunique() > 1]
        for col in varying_cols:
            self._typed_encoders_with_columns.append(
                (col, LabelEncoder().fit(df[col]))
            )
        logger.debug("Typed encoder for %s has been prepared.", self.__class__.__name__)

    # pylint: disable=too-many-locals
    def _run_iteration(self) -> None:
        """
        Run single optimization iteration using a factor-typed LinearGAM.

        String-typed columns receive f() (factor) terms; numeric columns receive
        s() (spline) terms. Constant columns are excluded. Dict-valued model
        columns are serialized to model_id strings before encoding.
        """
        self._prepare_typed_encoder()
        encoders = self._typed_encoders_with_columns

        if not encoders:
            return

        df = pd.DataFrame(data=self.evaluations)
        df = df[df["score"].notna()].copy()
        data = df.drop(columns=["score"])
        for col in data.columns:
            data[col] = _serialize_dict_col(data[col])
        target = df["score"]

        x_train_enc = np.column_stack(
            [enc.transform(data[col]) for col, enc in encoders]
        )

        terms = None
        for i, (_, enc) in enumerate(encoders):
            term = gam_f(i) if isinstance(enc.classes_[0], str) else gam_s(i)
            terms = term if terms is None else terms + term

        gam = LinearGAM(terms)
        gam.fit(x_train_enc, target)

        remaining_evaluations = self._get_remaining_evaluations(
            self._search_space.combinations, self._evaluated_combinations
        )

        if not remaining_evaluations:
            return

        remaining_df = pd.DataFrame(remaining_evaluations)
        for col in remaining_df.columns:
            remaining_df[col] = _serialize_dict_col(remaining_df[col])

        encoded = np.column_stack(
            [enc.transform(remaining_df[col]) for col, enc in encoders]
        )
        predictions = gam.predict(encoded)

        for idx, val in enumerate(remaining_evaluations):
            val["score"] = predictions[idx]

        best_predictions = sorted(remaining_evaluations, key=lambda d: d["score"], reverse=True)

        for params in best_predictions[: self.settings.evals_per_trial]:
            params.pop("score", None)
            score = self._objective_function(params)
            self._evaluated_combinations.append(params)
            self.evaluations.append(params | {"score": score})

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
