# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import itertools
from typing import Any, Callable

from ai4rag.search_space.src.default_search_space import (
    get_default_ai4rag_search_space_parameters,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = ["SearchSpace", "AI4RAGSearchSpace"]


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

    return chunk_size > chunk_overlap


def _rule_adjust_window_to_retrieval_method(combination: dict) -> bool:
    """Define whether combination passes selected criterion."""

    window_size = combination.get(AI4RAGParamNames.WINDOW_SIZE)
    retrieval_method = combination.get(AI4RAGParamNames.RETRIEVAL_METHOD)

    if retrieval_method is None or window_size is None:
        raise SearchSpaceValueError("window_size and retrieval_method are required.")

    if window_size == 0 and retrieval_method == "window":
        return False
    elif window_size > 0 and retrieval_method == "simple":
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

    def __init__(self, params: list[Parameter] = None, rules: list[Callable] | None = None):
        self.params = params or []
        self._search_space = {param.name: param for param in self._params}
        self._rules = rules

    def __getitem__(self, item: str) -> Parameter:
        return self._search_space[item]

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
        return self._params

    @params.setter
    def params(self, params: list[Parameter]) -> None:
        if len(params) != len(set([param.name for param in params])):
            raise SearchSpaceValueError("Parameters must have unique names.")

        self._params = params

    @staticmethod
    def _apply_rules(combinations: list[dict], rules: list[Callable]) -> list[dict]:
        """
        Apply set of rules on the given combinations.
        Remove all solutions (nodes in the space) that do not meet criteria defined in rules.

        Parameters
        ----------
        combinations : list[dict]
            Possible combinations of parameters (nodes in the space of solutions).

        rules : list[Callable]
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

    rules : list[Callable]
        List of functions - called "rules" - that will be applied on each combination in the search space.
    """

    _rules = (
        _rule_chunk_size_bigger_than_chunk_overlap,
        _rule_adjust_window_to_retrieval_method,
    )

    def __init__(self, params: list[Parameter], rules: list[Callable] | None = None):
        default_search_space_parameters = get_default_ai4rag_search_space_parameters()

        self._validate_user_params(params)

        params = self._overwrite_default_search_space_with_user_provided_parameters(
            params, default_search_space_parameters
        )

        _summed_rules = self._rules + rules if rules else self._rules
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

        selected_params = []

        for k, v in default_params.items():
            if k in user_params.keys():
                selected_params.append(user_params[k])
            else:
                selected_params.append(v)

        return selected_params
