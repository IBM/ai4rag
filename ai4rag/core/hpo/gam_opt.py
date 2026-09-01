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
    """Serialize model-valued cells to their model_id string.

    Handles both dict-valued models ({"model_id": "..."}, production) and
    model object instances with a model_id attribute (tests / direct API use).
    """

    def _needs_serialization(x: object) -> bool:
        return isinstance(x, dict) or hasattr(x, "model_id")

    def _to_model_id(x: object) -> object:
        if isinstance(x, dict):
            return x.get("model_id", str(x))
        if hasattr(x, "model_id"):
            return x.model_id
        return x

    if series.apply(_needs_serialization).any():
        return series.apply(_to_model_id)
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


def _str_val(v: object) -> str:
    """Normalize any cell value to a string key (handles model objects and plain strings)."""
    if v is None:
        return "__none__"
    if isinstance(v, dict):
        return v.get("model_id", str(v))
    if hasattr(v, "model_id"):
        return v.model_id
    return str(v)


def _get_discrete_column_values(combinations: list[dict]) -> dict[str, set[str]]:
    """Return {col: set_of_str_values} for all discrete columns in the combinations.

    All parameter types are included — strings, model objects, numeric values — so
    that coverage tracking works for columns like chunk_size and chunk_overlap as
    well as string columns like chunking_method or search_mode.
    """
    if not combinations:
        return {}
    result: dict[str, set[str]] = {}
    for col in combinations[0]:
        sample = next((c.get(col) for c in combinations if c.get(col) is not None), None)
        if sample is None:
            continue
        result[col] = {_str_val(c.get(col)) for c in combinations}
    return result


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
    warm_start_strategy : {"random", "greedy", "balanced"}, default="random"
        Controls how the initial n_random_nodes observations are selected/ordered.
        "random"   — shuffle the candidate list and take the first n as-is.
        "greedy"   — greedily pick n combinations so every string column value
                     appears at least twice (raises if n_random_nodes is too small).
        "balanced" — round-robin across the tuple of fields_to_balance values;
                     non-balanced discrete column values each appear at least once.
                     Requires fields_to_balance to be set.
    fields_to_balance : list[str] | None, default=None
        Field names to balance by round-robin when warm_start_strategy="balanced".
        Each unique value combination of these fields is guaranteed to appear at
        least once in the first n_random_nodes evaluations.
    random_state : int, default=64
        Inherited from OptimizerSettings. Controls shuffle order of initial
        random exploration phase. Does NOT control GAM model randomness
        (GAM training is deterministic).
    """

    n_random_nodes: int = 4
    evals_per_trial: int = 1
    warm_start_strategy: Literal["random", "greedy", "balanced"] = "random"
    fields_to_balance: list[str] | None = None

    def __post_init__(self) -> None:
        valid = {"random", "greedy", "balanced"}
        if self.warm_start_strategy not in valid:
            raise ValueError(
                f"warm_start_strategy must be one of {sorted(valid)}; " f"got {self.warm_start_strategy!r}."
            )
        if self.warm_start_strategy == "balanced" and not self.fields_to_balance:
            raise ValueError("fields_to_balance must be a non-empty list when warm_start_strategy='balanced'.")


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

        self._validate_n_random_nodes()

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

    def _validate_n_random_nodes(self) -> None:
        """Raise ValueError when n_random_nodes is below the required minimum for the strategy.

        - random:   No minimum enforced — combinations are taken in shuffle order.
        - greedy:   n_random_nodes >= 2 * max_unique_values_per_column, so every
                    discrete column value (string or numeric) appears at least twice.
        - balanced: n_random_nodes >= max(n_balanced_tuples, max_non_balanced_unique),
                    where n_balanced_tuples is the number of unique value-tuples for
                    fields_to_balance and max_non_balanced_unique is the max number of
                    unique values among the remaining discrete columns.
        """
        combinations = self._search_space.combinations
        if not combinations:
            return

        strategy = self.settings.warm_start_strategy

        if strategy == "random":
            return

        str_cols = _get_discrete_column_values(combinations)

        # Already-successful known_observations count toward the coverage budget.
        successful_known = sum(1 for e in self.evaluations if e.get("score") is not None)
        # effective_budget: the larger of n_random_nodes and what known_observations
        # already provide — if known obs alone meet the minimum, no raise is needed.
        effective_budget = max(self.settings.n_random_nodes, successful_known)

        if strategy == "greedy":
            if not str_cols:
                return
            max_unique = max(len(vals) for vals in str_cols.values())
            min_required = max(4, 2 * max_unique)
            if effective_budget < min_required:
                raise ValueError(
                    f"n_random_nodes={self.settings.n_random_nodes} is too small for "
                    f"warm_start_strategy='greedy': each discrete column value must appear "
                    f"at least twice (max unique values per column: {max_unique}). "
                    f"Set n_random_nodes >= {min_required}."
                )

        elif strategy == "balanced":
            fields_to_balance = self.settings.fields_to_balance or []
            balanced_tuples = {tuple(_str_val(c.get(f)) for f in fields_to_balance) for c in combinations}
            n_balanced = len(balanced_tuples)
            non_balanced = {col: vals for col, vals in str_cols.items() if col not in fields_to_balance}
            max_non_balanced = max((len(vals) for vals in non_balanced.values()), default=0)
            min_required = max(4, n_balanced, max_non_balanced)
            if effective_budget < min_required:
                raise ValueError(
                    f"n_random_nodes={self.settings.n_random_nodes} is too small for "
                    f"warm_start_strategy='balanced' with fields_to_balance={fields_to_balance!r}. "
                    f"n_balanced_tuples={n_balanced}, max_non_balanced_unique={max_non_balanced}. "
                    f"Set n_random_nodes >= {min_required}."
                )

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

        The selection order depends on warm_start_strategy:
        "random"   — shuffled order (no reordering).
        "greedy"   — greedy selection maximizing string-column coverage (each value >= 2 times).
        "balanced" — round-robin across fields_to_balance value tuples.
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

        if self.settings.warm_start_strategy == "greedy":
            str_cols = _get_discrete_column_values(combinations_local)
            initial_coverage: dict[str, dict[str, int]] = {
                col: {val: 0 for val in vals} for col, vals in str_cols.items()
            }
            for obs in self.evaluations:
                if obs.get("score") is None:
                    continue
                for col in initial_coverage:
                    val = _str_val(obs.get(col))
                    if val in initial_coverage[col]:
                        initial_coverage[col][val] = min(initial_coverage[col][val] + 1, 2)
            remaining_budget = self.settings.n_random_nodes - successful_evaluations
            combinations_local = self._get_greedy_combinations(
                combinations_local, remaining_budget, initial_coverage=initial_coverage
            )
        elif self.settings.warm_start_strategy == "balanced":
            combinations_local = self._get_balanced_combinations(
                combinations_local, self.settings.fields_to_balance or []
            )
        # "random": use shuffled list as-is

        discrete_cols_in_space = _get_discrete_column_values(combinations_local)
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

        uncovered_by_col: dict[str, list[str]] = {}
        for col, vals_in_space in discrete_cols_in_space.items():
            covered = {_str_val(e.get(col)) for e in self.evaluations if e.get("score") is not None}
            uncovered = vals_in_space - covered
            if uncovered:
                uncovered_by_col[col] = sorted(uncovered)
        if uncovered_by_col:
            logger.warning(
                "n_random_nodes=%d was too small to cover all discrete column values. "
                "Uncovered values by column: %s. Consider increasing n_random_nodes.",
                self.settings.n_random_nodes,
                uncovered_by_col,
            )

    @staticmethod
    def _get_greedy_combinations(
        combinations: list[dict],
        n: int,
        initial_coverage: dict[str, dict[str, int]] | None = None,
    ) -> list[dict]:
        """Greedily select n combinations ensuring every discrete column value appears >= 2 times.

        At each step the candidate with the highest coverage gain (number of discrete column
        values — string or numeric — whose current count is still below 2) is selected.
        Ties are broken by the shuffle order coming in. The n selected combinations are
        returned first, followed by the remaining combinations in their original (shuffled) order.

        initial_coverage seeds the per-value counts so that values already covered by
        known_observations are not redundantly targeted.
        """
        if not combinations or n <= 0:
            return combinations

        str_cols = _get_discrete_column_values(combinations)
        if not str_cols:
            return combinations

        coverage: dict[str, dict[str, int]] = {col: {val: 0 for val in vals} for col, vals in str_cols.items()}
        if initial_coverage:
            for col, val_counts in initial_coverage.items():
                if col in coverage:
                    for val, count in val_counts.items():
                        if val in coverage[col]:
                            coverage[col][val] = min(count, 2)

        def _gain(c: dict) -> int:
            return sum(1 for col, val_counts in coverage.items() if val_counts.get(_str_val(c.get(col)), 0) < 2)

        remaining_indices = list(range(len(combinations)))
        selected_indices: list[int] = []

        for _ in range(min(n, len(combinations))):
            if not remaining_indices:
                break
            best_pos = max(range(len(remaining_indices)), key=lambda p: _gain(combinations[remaining_indices[p]]))
            best_idx = remaining_indices.pop(best_pos)
            selected_indices.append(best_idx)
            for col in str_cols:
                val = _str_val(combinations[best_idx].get(col))
                if val in coverage[col]:
                    coverage[col][val] = min(coverage[col][val] + 1, 2)

        return [combinations[i] for i in selected_indices] + [combinations[i] for i in remaining_indices]

    @staticmethod
    def _get_balanced_combinations(combinations: list[dict], fields_to_balance: list[str]) -> list[dict]:
        """Order by outer round-robin across fields_to_balance tuples, with an inner
        round-robin across remaining string columns within each balanced bucket.

        The outer round-robin guarantees every fields_to_balance value-tuple appears
        at least once when n_random_nodes >= n_balanced_tuples. The inner round-robin
        ensures non-balanced discrete columns (e.g. chunking_method, chunk_size) also
        vary across the initial evaluations rather than repeating the same value for each bucket.
        """
        if not combinations or not fields_to_balance:
            return combinations

        def _outer_key(c: dict) -> tuple:
            return tuple(_str_val(c.get(f)) for f in fields_to_balance)

        str_cols = _get_discrete_column_values(combinations)
        other_str_fields = [col for col in str_cols if col not in fields_to_balance]

        outer_buckets: dict[tuple, list[dict]] = defaultdict(list)
        for c in combinations:
            outer_buckets[_outer_key(c)].append(c)

        if other_str_fields:
            # Apply round-robin per non-balanced field in ascending cardinality order
            # so the highest-cardinality field (e.g. chunk_size) dominates the cycling
            # pattern.  Using a full-tuple key creates unique keys for every combination
            # (single-item buckets), making _round_robin a no-op — so we must apply it
            # one field at a time.
            fields_by_cardinality = sorted(other_str_fields, key=lambda f: len(str_cols[f]))

            per_bucket_balanced: dict[tuple, list[dict]] = {}
            for key, combs in outer_buckets.items():
                result: list[dict] = list(combs)
                for field in fields_by_cardinality:
                    result = _round_robin(result, lambda c, f=field: _str_val(c.get(f)))
                per_bucket_balanced[key] = result
        else:
            per_bucket_balanced = dict(outer_buckets)

        ordered = [c for combs in per_bucket_balanced.values() for c in combs]
        return _round_robin(ordered, _outer_key)

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
            self._typed_encoders_with_columns.append((col, LabelEncoder().fit(df[col])))
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
        # known_observations may omit columns that vary in the search space; fill
        # with the encoder's first class so transform() does not KeyError.
        for col, enc in encoders:
            if col not in data.columns:
                data[col] = enc.classes_[0]
        target = df["score"]

        x_train_enc = np.column_stack([enc.transform(data[col]) for col, enc in encoders])

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

        encoded = np.column_stack([enc.transform(remaining_df[col]) for col, enc in encoders])
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
