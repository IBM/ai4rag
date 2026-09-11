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
    max_evals : int | None, default=None
        Maximum number of objective-function evaluations performed during
        optimization, including warm-start and GAM evaluations. When omitted,
        every available search-space combination is evaluated.
    max_iterations : int | None, default=None
        Maximum number of evaluated RAG patterns retained and published when the
        search completes. It controls the warm-start/GAM output allocation, not
        the hard evaluation budget; use ``max_evals`` to bound objective-function
        calls. When omitted, it is set to the effective ``max_evals`` value. It
        cannot exceed an explicitly configured ``max_evals``.
    n_random_nodes : int, default=4
        Number of random configurations to evaluate before starting GAM iterations.
    evals_per_trial : int, default=1
        Number of configurations to evaluate per GAM iteration.
    warm_start_strategy : {"random", "greedy", "balanced"}, default="random"
        Controls how the initial n_random_nodes observations are selected/ordered.
        "random"   — shuffle the candidate list and take the first n as-is.
        "greedy"   — greedily pick combinations so every discrete column value
                     appears at least twice. If n_random_nodes is below the
                     computed minimum (min_required), the warm start is
                     auto-adjusted upward to meet coverage. One output slot is
                     allocated per four effective warm-start evaluations; GAM
                     fills the remaining ``max_iterations`` slots.
        "balanced" — round-robin across the tuple of fields_to_balance values;
                     non-balanced discrete column values each appear at least once.
                     Requires fields_to_balance to be set. Same auto-adjustment,
                     and output allocation rules as "greedy".
    fields_to_balance : list[str] | None, default=None
        Field names to balance by round-robin when warm_start_strategy="balanced".
        Each unique value combination of these fields is guaranteed to appear at
        least once in the first n_random_nodes evaluations.
    random_state : int, default=64
        Inherited from OptimizerSettings. Controls shuffle order of initial
        random exploration phase. Does NOT control GAM model randomness
        (GAM training is deterministic).
    """

    max_evals: int | None = None
    max_iterations: int | None = None
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
        if self.max_iterations is not None:
            if self.max_iterations < 1:
                raise ValueError("max_iterations must be at least 1 when provided.")
            if self.max_evals is not None and self.max_iterations > self.max_evals:
                raise ValueError("max_iterations cannot exceed max_evals.")


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
        Effective maximum number of evaluated patterns retained as search results
        and published to the event handler. This is bounded by ``max_evals`` and
        the number of available search-space combinations.
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
        self.warm_start_evaluation_count: int = 0
        self.current_phase = "idle"

        if known_observations:
            self._load_known_observations(known_observations)

        self._validate_fields_to_balance()
        self._validate_n_random_nodes()

        self.max_evals = (
            self._search_space.max_combinations
            if self.settings.max_evals is None
            else min(self.settings.max_evals, self._search_space.max_combinations)
        )
        self.max_iterations = self.settings.max_iterations

    @property
    def max_iterations(self) -> int:
        """Get the effective maximum number of retained result patterns."""
        return self._max_iterations

    @max_iterations.setter
    def max_iterations(self, val: int | None) -> None:
        """Set maximum number of result patterns retained after HPO."""
        max_comb = self._search_space.max_combinations
        if val is None:
            self._max_iterations = self.max_evals
            return
        max_results = min(self.max_evals, max_comb)
        if val > max_results:
            logger.info(
                ("'max_iterations' exceeded the available evaluation budget: %s. " "Setting 'max_iterations' to: %s"),
                max_results,
                max_results,
            )
            self._max_iterations = max_results
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
        self.current_phase = "warm_start"
        self.evaluate_initial_random_nodes()

        strategy = self.settings.warm_start_strategy
        self.current_phase = "gam"
        if strategy in ("greedy", "balanced"):
            effective_warm_start = self._compute_warm_start_effective_target()
            output_limit = self.max_iterations
            gam_output_count = max(0, output_limit - (effective_warm_start // 4))
            gam_iterations = ceil(gam_output_count / self.settings.evals_per_trial)
            remaining_evaluation_capacity = max(0, self.max_evals - len(self.evaluations))
            capacity_iterations = ceil(remaining_evaluation_capacity / self.settings.evals_per_trial)
            for _ in range(min(gam_iterations, capacity_iterations)):
                self._run_iteration()
        else:
            iterations_limit = self._get_iterations_limit()
            for _ in range(iterations_limit):
                self._run_iteration()

        self._trim_evaluations_to_top(self.max_iterations)

        self.current_phase = "complete"

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
        iterations_limit = ceil((self.max_evals - len(self.evaluations)) / self.settings.evals_per_trial)
        return max(0, iterations_limit)

    def _validate_n_random_nodes(self) -> None:
        """Log a warning when n_random_nodes is below the required minimum for the strategy.

        - random:   No minimum enforced — combinations are taken in shuffle order.
        - greedy:   If n_random_nodes < 2 * max_unique_values_per_column, warm start
                    is auto-adjusted to the minimum required (no error raised).
        - balanced: If n_random_nodes < max(n_balanced_tuples, max_non_balanced_unique),
                    warm start is auto-adjusted to the minimum required (no error raised).
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
                logger.info(
                    "n_random_nodes=%d is below the minimum required %d for "
                    "warm_start_strategy='greedy' (max unique values per column: %d). "
                    "Warm start will be auto-adjusted to %d nodes.",
                    self.settings.n_random_nodes,
                    min_required,
                    max_unique,
                    min_required,
                )

        elif strategy == "balanced":
            fields_to_balance = self.settings.fields_to_balance or []
            balanced_tuples = {tuple(_str_val(c.get(f)) for f in fields_to_balance) for c in combinations}
            n_balanced = len(balanced_tuples)
            non_balanced = {col: vals for col, vals in str_cols.items() if col not in fields_to_balance}
            max_non_balanced = max((len(vals) for vals in non_balanced.values()), default=0)
            min_required = max(4, n_balanced, max_non_balanced)
            if effective_budget < min_required:
                logger.info(
                    "n_random_nodes=%d is below the minimum required %d for "
                    "warm_start_strategy='balanced' with fields_to_balance=%r "
                    "(n_balanced_tuples=%d, max_non_balanced_unique=%d). "
                    "Warm start will be auto-adjusted to %d nodes.",
                    self.settings.n_random_nodes,
                    min_required,
                    fields_to_balance,
                    n_balanced,
                    max_non_balanced,
                    min_required,
                )

    def _validate_fields_to_balance(self) -> None:
        """Reject balanced warm-start fields that are absent from the search space."""
        if self.settings.warm_start_strategy != "balanced":
            return

        combinations = self._search_space.combinations
        if not combinations:
            return

        available_fields = set(combinations[0])
        unknown_fields = set(self.settings.fields_to_balance or []) - available_fields
        if unknown_fields:
            raise ValueError(
                "fields_to_balance contains field(s) absent from the search space: "
                f"{sorted(unknown_fields)}. Available fields: {sorted(available_fields)}."
            )

    def _compute_warm_start_effective_target(self) -> int:
        """Return the effective number of successful warm-start nodes to evaluate.

        For "greedy" and "balanced" strategies, this is
        max(n_random_nodes, min_required) where min_required guarantees adequate
        discrete-value coverage. For "random", returns n_random_nodes unchanged.
        """
        n = self.settings.n_random_nodes
        strategy = self.settings.warm_start_strategy
        if strategy == "random":
            return n
        combinations = self._search_space.combinations
        if not combinations:
            return n
        str_cols = _get_discrete_column_values(combinations)
        if strategy == "greedy":
            if not str_cols:
                return max(4, n)
            max_unique = max(len(vals) for vals in str_cols.values())
            return max(n, 4, 2 * max_unique)
        # balanced
        fields = self.settings.fields_to_balance or []
        balanced_tuples = {tuple(_str_val(c.get(f)) for f in fields) for c in combinations}
        non_balanced = {col: vals for col, vals in str_cols.items() if col not in fields}
        max_non_balanced = max((len(vals) for vals in non_balanced.values()), default=0)
        return max(n, 4, len(balanced_tuples), max_non_balanced)

    def compute_warm_start_effective_target(self) -> int:
        """Return the effective number of successful warm-start nodes to evaluate.

        Returns
        -------
        int
            The effective warm-start target for the configured strategy.
        """
        return self._compute_warm_start_effective_target()

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
        Perform evaluation of randomly chosen nodes from the solutions space.
        All strategies stop at the configured maximum evaluation count. Greedy
        and balanced starts may therefore finish before their coverage target
        when the evaluation budget is smaller than the coverage requirement.

        When the optimizer has been warm-started with known observations,
        already-successful evaluations count toward the n_random_nodes target
        and already-evaluated combinations are excluded from candidates.

        The selection order depends on warm_start_strategy:
        "random"   — shuffled order (no reordering).
        "greedy"   — greedy selection maximizing string-column coverage (each value >= 2 times).
        "balanced" — round-robin across fields_to_balance value tuples.
        """
        successful_evaluations = sum(1 for e in self.evaluations if e["score"] is not None)
        effective_target = self._compute_warm_start_effective_target()

        if successful_evaluations >= effective_target:
            logger.info(
                "Skipping random evaluation phase: %d known successful evaluations >= warm_start_target (%d).",
                successful_evaluations,
                effective_target,
            )
            return

        if len(self.evaluations) >= self.max_evals:
            return

        combinations_local = self._prepare_warm_start_combinations(effective_target, successful_evaluations)
        discrete_cols_in_space = _get_discrete_column_values(combinations_local)
        self._evaluate_warm_start_combinations(combinations_local, effective_target, successful_evaluations)

        self._log_uncovered_values(discrete_cols_in_space, self.evaluations, effective_target)

        if self.settings.warm_start_strategy == "balanced":
            self._cover_non_balanced_values(discrete_cols_in_space)

        self.warm_start_evaluation_count = len(self.evaluations)

    def _prepare_warm_start_combinations(self, effective_target: int, successful_evaluations: int) -> list[dict]:
        """Prepare candidate combinations according to the warm-start strategy."""
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
            remaining_budget = effective_target - successful_evaluations
            return self._get_greedy_combinations(
                combinations_local, remaining_budget, initial_coverage=initial_coverage
            )

        if self.settings.warm_start_strategy == "balanced":
            return self._get_balanced_combinations(combinations_local, self.settings.fields_to_balance or [])

        # "random": use shuffled list as-is
        return combinations_local

    def _evaluate_warm_start_combinations(
        self, combinations: list[dict], effective_target: int, successful_evaluations: int
    ) -> None:
        """Evaluate warm-start candidates until the target or candidate list is exhausted."""
        gen = (x for x in combinations)

        while successful_evaluations < effective_target:
            params = next(gen, None)
            if params is None:
                break
            score = self._objective_function(params=params)
            if score is not None:
                successful_evaluations += 1
            self._evaluated_combinations.append(params)
            self.evaluations.append(params | {"score": score})

            if len(self.evaluations) >= self.max_evals:
                break

    @staticmethod
    def _log_uncovered_values(
        discrete_cols_in_space: dict[str, set],
        evaluations: list[dict],
        n_random_nodes: int,
    ) -> None:
        uncovered_by_col: dict[str, list[str]] = {}
        for col, vals_in_space in discrete_cols_in_space.items():
            covered = {_str_val(e.get(col)) for e in evaluations if e.get("score") is not None}
            uncovered = vals_in_space - covered
            if uncovered:
                uncovered_by_col[col] = sorted(uncovered)
        if uncovered_by_col:
            logger.warning(
                "n_random_nodes=%d was too small to cover all discrete column values. "
                "Uncovered values by column: %s. Consider increasing n_random_nodes.",
                n_random_nodes,
                uncovered_by_col,
            )

    def _cover_non_balanced_values(self, discrete_cols_in_space: dict[str, set[str]]) -> None:
        """Evaluate additional nodes so every non-balanced column value appears at least once.

        Called after the balanced warm start loop to guarantee GAM training data is not
        missing any non-balanced column value (which would cause pygam domain errors at
        prediction time when the GAM has only seen one class for a categorical feature).
        """
        fields_to_balance = set(self.settings.fields_to_balance or [])
        non_balanced = {col: vals for col, vals in discrete_cols_in_space.items() if col not in fields_to_balance}
        if not non_balanced:
            return

        seen: dict[str, set[str]] = {col: set() for col in non_balanced}
        for e in self.evaluations:
            if e.get("score") is None:
                continue
            for col in non_balanced:
                seen[col].add(_str_val(e.get(col)))

        remaining = [c for c in self._search_space.combinations if c not in self._evaluated_combinations]

        for col, vals in non_balanced.items():
            for val in vals:
                if val in seen[col] or not remaining or len(self.evaluations) >= self.max_evals:
                    continue
                candidate = next((c for c in remaining if _str_val(c.get(col)) == val), None)
                if candidate is None:
                    continue
                logger.info(
                    "Balanced warm start: evaluating extra node to cover uncovered value '%s'='%s'.",
                    col,
                    val,
                )
                score = self._objective_function(params=candidate)
                self._evaluated_combinations.append(candidate)
                self.evaluations.append(candidate | {"score": score})
                remaining.remove(candidate)
                if score is not None:
                    for other_col in non_balanced:
                        seen[other_col].add(_str_val(candidate.get(other_col)))

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
        Run single optimization iteration using typed LinearGAM terms.

        String-typed columns receive f() (factor) terms; numeric columns receive
        s() (spline) terms. Random warm starts and sparse categorical training
        data use s() so a category absent from the sample remains in-domain
        during prediction. Constant columns are excluded. Dict-valued model
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
            observed_values = set(x_train_enc[:, i])
            all_values = set(range(len(enc.classes_)))
            use_spline = (
                self.settings.warm_start_strategy == "random"
                or not isinstance(enc.classes_[0], str)
                or observed_values != all_values
            )
            term = gam_s(i) if use_spline else gam_f(i)
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

        remaining_evaluation_capacity = max(0, self.max_evals - len(self.evaluations))
        for params in best_predictions[: min(self.settings.evals_per_trial, remaining_evaluation_capacity)]:
            params.pop("score", None)
            score = self._objective_function(params)
            self._evaluated_combinations.append(params)
            self.evaluations.append(params | {"score": score})

    def _trim_evaluations_to_top(self, n: int) -> None:
        """Trim self.evaluations to the top n successful entries by score.

        Failed evaluations (score is None) are discarded. Successful evaluations
        are sorted descending by score and only the top n are retained.
        """
        successful = sorted(
            [e for e in self.evaluations if e.get("score") is not None],
            key=lambda d: d["score"],
            reverse=True,
        )
        self.evaluations = successful[:n]

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
