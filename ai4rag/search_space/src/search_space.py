# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import itertools
from typing import Any, Callable, TypeAlias

from ai4rag.search_space.src.default_search_space import (
    get_default_ai4rag_search_space_parameters,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import AI4RAGParamNames, RetrievalConstraints

__all__ = ["SearchSpace", "AI4RAGSearchSpace"]


RuleFunction: TypeAlias = Callable[[dict[str, Any]], bool]


def _rule_chunk_size_bigger_than_chunk_overlap(combination: dict) -> bool:
    """Define whether combination passes selected criterion.

    Parameters
    ----------
    combination : dict
        Single node in the solutions space represented as a dict.

    Returns
    -------
    bool
        Whether combination passes selected criterion.
    """
    chunk_size = combination.get(AI4RAGParamNames.CHUNK_SIZE)
    chunk_overlap = combination.get(AI4RAGParamNames.CHUNK_OVERLAP)

    if chunk_size is None or chunk_overlap is None:
        raise SearchSpaceValueError("Chunk size and chunk overlap are required.")

    return chunk_size > 2 * chunk_overlap


def _rule_adjust_window_to_retrieval_method(combination: dict) -> bool:
    """Define whether combination passes selected criterion."""

    window_size = combination.get(AI4RAGParamNames.WINDOW_SIZE)
    retrieval_method = combination.get(AI4RAGParamNames.RETRIEVAL_METHOD)

    if retrieval_method is None or window_size is None:
        raise SearchSpaceValueError("window_size and retrieval_method are required.")

    if window_size == 0 and retrieval_method == "window":
        return False
    if window_size > 0 and retrieval_method == "simple":
        return False

    return True


def _rule_chunk_size_within_embedding_context_length(combination: dict) -> bool:
    """Check that estimated token count of a chunk fits the embedding model's context length.

    The chunk_size is in characters.  We convert to an estimated token count
    using a conservative ratio of 4.5 characters per token (i.e. we
    *overestimate* the number of tokens so that borderline cases are pruned
    rather than silently failing at runtime).

    Note: ``chunk_overlap`` is not included in the estimation because
    ``RecursiveCharacterTextSplitter`` already keeps chunks within the
    ``chunk_size`` budget — overlap is not added on top.

    Parameters
    ----------
    combination : dict
        Single node in the solutions space represented as a dict.

    Returns
    -------
    bool
        Whether the estimated token count fits within the embedding model's context length.
    """
    chunk_size = combination.get(AI4RAGParamNames.CHUNK_SIZE)
    embedding_model = combination.get(AI4RAGParamNames.EMBEDDING_MODEL)

    if chunk_size is None or embedding_model is None:
        return True

    context_length = getattr(getattr(embedding_model, "params", None), "context_length", None)
    if context_length is None:
        params = getattr(embedding_model, "params", None)
        if isinstance(params, dict):
            context_length = params.get("context_length")

    if context_length is None:
        return True

    chars_per_token = 3.6
    estimated_tokens = chunk_size / chars_per_token

    return estimated_tokens <= context_length


def _rule_search_mode_ranker_consistency(combination: dict) -> bool:
    """Ranker parameters must only be set when search_mode is 'hybrid'.

    When search_mode is 'vector', all ranker params must be sentinels
    (empty string for strategy, 0 for ranker_k, 1 for ranker_alpha).
    When search_mode is 'hybrid', ranker_strategy must be a non-empty string.

    Parameters
    ----------
    combination : dict
        Single node in the solutions space represented as a dict.

    Returns
    -------
    bool
        Whether combination passes selected criterion.
    """
    search_mode = combination.get(AI4RAGParamNames.SEARCH_MODE)
    ranker_strategy = combination.get(AI4RAGParamNames.RANKER_STRATEGY)
    ranker_k = combination.get(AI4RAGParamNames.RANKER_K)
    ranker_alpha = combination.get(AI4RAGParamNames.RANKER_ALPHA)

    if search_mode == "vector":
        if ranker_strategy or ranker_k or ranker_alpha not in (1, None):
            return False
        return True

    if search_mode == "hybrid":
        if ranker_strategy not in RetrievalConstraints.RANKER_STRATEGIES:
            return False
        return True

    return True


def _rule_ranker_k_for_rrf_only(combination: dict) -> bool:
    """ranker_k is only applicable when ranker_strategy is 'rrf'.

    For non-rrf strategies, ranker_k must be 0 (sentinel meaning unused).
    For rrf strategy, ranker_k must not be 0.

    Parameters
    ----------
    combination : dict
        Single node in the solutions space represented as a dict.

    Returns
    -------
    bool
        Whether combination passes selected criterion.
    """
    ranker_strategy = combination.get(AI4RAGParamNames.RANKER_STRATEGY)
    ranker_k = combination.get(AI4RAGParamNames.RANKER_K)

    if ranker_strategy is None or ranker_k is None:
        return True

    if ranker_strategy != "rrf" and ranker_k != 0:
        return False

    if ranker_strategy == "rrf" and ranker_k == 0:
        return False

    return True


def _rule_ranker_alpha_for_weighted_only(combination: dict) -> bool:
    """ranker_alpha is only applicable when ranker_strategy is 'weighted'.

    For non-weighted strategies, ranker_alpha must be 1 (sentinel meaning vector-only).
    For weighted strategy, ranker_alpha must not be 1.

    Parameters
    ----------
    combination : dict
        Single node in the solutions space represented as a dict.

    Returns
    -------
    bool
        Whether combination passes selected criterion.
    """
    ranker_strategy = combination.get(AI4RAGParamNames.RANKER_STRATEGY)
    ranker_alpha = combination.get(AI4RAGParamNames.RANKER_ALPHA)

    if ranker_strategy is None or ranker_alpha is None:
        return True

    if ranker_strategy != "weighted" and ranker_alpha != 1:
        return False

    if ranker_strategy == "weighted" and ranker_alpha == 1:
        return False

    return True


class SearchSpace:
    """
    Class that represents a search space used hyperparameter optimization.

    Parameters
    ----------
    params : list[Parameter]
        List of Parameters, each of which is a parameter to optimize in hyperparameter optimization process.
    """

    def __init__(self, params: list[Parameter] = None, rules: list[RuleFunction] | None = None):
        self.params = params or []
        self._search_space = {param.name: param for param in self._params}
        self._rules = rules

    def __getitem__(self, item: str) -> Parameter:
        return self._search_space[item]

    def __setitem__(self, key: str, item: Parameter) -> None:
        for idx, param in enumerate(self.params):
            if item.name == param.name:
                self.params[idx] = item
        self._search_space[item.name] = item

    def as_list(self) -> list[Parameter]:
        """
        Get the list of parameter composing the search space.

        Returns
        -------
        list[Parameter]
            List of parameters composing the search space.
        """
        return list(self.params)

    def as_dict(self) -> dict[str, Any]:
        """Return dict representation of the search space.

        Returns
        -------
        dict[str, Any]
            Dict representation of the search space."""
        return {param.name: param.all_values() for param in self._params}

    @property
    def params(self) -> list[Parameter]:
        """Get params."""
        return self._params

    @params.setter
    def params(self, params: list[Parameter]) -> None:
        """Set params."""
        if len(params) != len({param.name for param in params}):
            raise SearchSpaceValueError("Parameters must have unique names.")

        self._params = params

    @staticmethod
    def _apply_rules(combinations: list[dict], rules: list[RuleFunction]) -> list[dict]:
        """
        Apply set of rules on the given combinations.
        Remove all solutions (nodes in the space) that do not meet criteria defined in rules.

        Parameters
        ----------
        combinations : list[dict]
            Possible combinations of parameters (nodes in the space of solutions).

        rules : list[RuleFunction]
            List of rules to apply on the combinations.

        Returns
        -------
        list[dict]
            Filtered combinations of parameters after applying rules.
        """
        indexes_to_remove = []

        for idx, combination in enumerate(combinations):
            for rule in rules:
                if not rule(combination):
                    indexes_to_remove.append(idx)
                    continue

        combinations = [combination for idx, combination in enumerate(combinations) if idx not in indexes_to_remove]

        return combinations

    @property
    def combinations(self) -> list[dict]:
        """Get all possible parameters combinations."""

        space_params = {param.name: param.all_values() for param in self.params}
        combinations = [dict(zip(space_params.keys(), values)) for values in itertools.product(*space_params.values())]

        if self._rules:
            combinations = self._apply_rules(combinations, self._rules)

        return combinations

    @property
    def max_combinations(self) -> int:
        """
        Calculate how many possible combinations could be evaluated based
        on the search space.

        Returns
        -------
        int
            Number of nodes in the hyperspace.
        """
        return len(self.combinations)


class AI4RAGSearchSpace(SearchSpace):
    """
    Class that represents the search space used for the RAG hyperparameters optimization.

    Parameters
    ----------
    params : list[Parameter]
        List of Parameter, each of which is a parameter to optimize in the ai4rag process.

    rules : list[RuleFunction]
        List of functions - called "rules" - that will be applied on each combination in the search space.

    vector_store_type : str, default="ls_milvus"
        Type of vector store. When "chroma", hybrid search parameters are excluded
        from the default search space since ChromaDB does not support hybrid search.
    """

    _base_rules = (
        _rule_chunk_size_bigger_than_chunk_overlap,
        _rule_adjust_window_to_retrieval_method,
        _rule_chunk_size_within_embedding_context_length,
    )

    _hybrid_rules = (
        _rule_search_mode_ranker_consistency,
        _rule_ranker_k_for_rrf_only,
        _rule_ranker_alpha_for_weighted_only,
    )

    def __init__(
        self,
        params: list[Parameter] | None = None,
        rules: list[RuleFunction] | None = None,
        vector_store_type: str = "ls_milvus",
    ):
        default_search_space_parameters = get_default_ai4rag_search_space_parameters(vector_store_type)
        params = params or []
        self._validate_user_params(params)

        params = self._overwrite_default_search_space_with_user_provided_parameters(
            params, default_search_space_parameters
        )

        builtin_rules = self._base_rules + self._hybrid_rules if vector_store_type != "chroma" else self._base_rules
        _summed_rules = builtin_rules + rules if rules else builtin_rules
        super().__init__(params, _summed_rules)

    @staticmethod
    def _validate_user_params(params: list[Parameter]) -> None:
        """Validate parameters provided by the user, that will be later
        used for overriding the defaults.

        Parameters
        ----------
        params : list[Parameter]
            Parameters provided by the user.

        Raises
        ------
        SearchSpaceValueError
            Raised when some parameters are not recognized or required ones are missing.
        """

        required_params = (AI4RAGParamNames.FOUNDATION_MODEL, AI4RAGParamNames.EMBEDDING_MODEL)
        user_params = [param.name for param in params]
        missing_params = set(required_params) - set(user_params)

        if missing_params:
            raise SearchSpaceValueError(f"Missing required parameters in the search space: {missing_params}.")

        not_supported_params = [param for param in user_params if param not in AI4RAGParamNames]

        if not_supported_params:
            raise SearchSpaceValueError(
                f"Not supported parameters were given to the search space: {not_supported_params}."
            )

    @staticmethod
    def _overwrite_default_search_space_with_user_provided_parameters(
        params: list[Parameter],
        default_search_space_params: list[Parameter],
    ) -> list[Parameter]:
        """
        User-provided data has higher precedence than the defaults that's why we're overwriting the defaults here.

        Parameters
        ----------
        params : list[Parameter]
            List of parameters to build up this search space.

        default_search_space_params : list[Parameter]
            Default parameters to be considered.

        Returns
        -------
            Default search space overwritten and expanded (whenever necessary) with user-provided parameters.

        Raises
        ------
        SearchSpaceValueError
            When user provided unsupported parameter (not existing in the default search space).

        """
        user_params = {param.name: param for param in params}
        default_params = {param.name: param for param in default_search_space_params}

        selected_params_dict = default_params | user_params
        selected_params = list(selected_params_dict.values())

        return selected_params
