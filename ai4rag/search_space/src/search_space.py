# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import itertools
from functools import cached_property

from pydantic import BaseModel, field_validator

from ai4rag.search_space.src.default_search_space import (
    create_cartesian_product_of_possible_configurations,
    get_default_search_space,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils import remove_duplicates
from ai4rag.utils.constants import AI4RAGParamNames


__all__ = ["SearchSpace", "AI4RAGSearchSpace"]


class SearchSpace:
    """
    Class that represents a search space used hyperparameter optimization.

    Parameters
    ----------
    params : list[Parameter]
        List of Parameters, each of which is a parameter to optimize in hyperparameter optimization process.
    """

    def __init__(self, params: list[Parameter] = None) -> None:
        self.search_space = [] if params is None else params

    def __getitem__(self, item: str) -> Parameter:
        return self.search_space[item]

    @property
    def search_space(self) -> dict[str, Parameter]:
        """Get search space."""
        return self._search_space

    @search_space.setter
    def search_space(self, params: list[Parameter]) -> None:
        """
        Set _search_space value.
        """
        params = self._apply_constraints(params)
        self._search_space = {param.name: param for param in params}

    def as_list(self) -> list[Parameter]:
        """
        Get the list of parameter composing the search space.

        Returns
        -------
        list[Parameter]
            List of parameters composing the search space.
        """
        return list(self._search_space.values())

    @staticmethod
    def _apply_constraints(params: list[Parameter]) -> list[Parameter]:
        """
        Modifies parameter list taking into account search space's constraints.
            - no duplicated params should exist
            - no different params with the same name should exist

        Parameters
        ----------
        params: list[Parameter]
            list of Parameter objects that this search space holds.

        Returns
        -------
        list[Parameter]
            processed list of Parameter objects.

        Raises
        ------
        SearchSpaceValueError
            if there are different parameters with the same name.

        """
        deduplicated_params = set(params)
        if len(set(p.name for p in deduplicated_params)) != len(deduplicated_params):
            raise SearchSpaceValueError("Search space parameters are invalid.")
        return list(deduplicated_params)

    @cached_property
    def combinations(self) -> list[dict]:
        """Get all possible parameters combinations."""
        params = self.as_list()

        if any(param.param_type == "R" for param in params):
            raise TypeError("Cannot calculate max possible combinations for 'R' type parameters.")

        space_params = {param.name: param.all_values() for param in params}
        combinations = [dict(zip(space_params.keys(), values)) for values in itertools.product(*space_params.values())]

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


_rules_keys_mapping = {
    "&&": "and",
    "||": "or",
    "==": "eq",
    "!=": "!=",
    ">": ">",
    ">=": ">=",
    "<": "<",
    "<=": "<="
}


class Rule(BaseModel):
    """Class representing single rule constraining the search space."""
    rule_str: str

    @field_validator("rule_str", mode="before")
    def validate_single_rule(self, rule_str: str) -> str:
        return rule_str

    def to_python_str_expression(self) -> str:
        p = self.rule_str.split("IF ")[1]
        lr = p.split("==>")[0]
        pr = p.split("==>")[1]



class RuleSet(BaseModel):
    rules: list[Rule]


class AI4RAGSearchSpace(SearchSpace):
    """
    Class that represents the search space used for the RAG hyperparameters optimization.

    Parameters
    ----------
    params : list[Parameter]
        List of Parameter, each of which is a parameter to optimize in the ai4rag process.
    """

    def __init__(self, params: list[Parameter] = None, **kwargs) -> None:
        params = [] if params is None else params

        default_search_space = get_default_search_space()

        params = self._overwrite_default_search_space_with_user_provided_parameters(
            params, kwargs.get("default_search_space", default_search_space)
        )
        super().__init__(params)

    @staticmethod
    def _overwrite_default_search_space_with_user_provided_parameters(
        params: list[Parameter], default_search_space: dict
    ) -> list[Parameter]:
        """
        User-provided data has higher precedence than the defaults that's why we're overwriting the defaults here.
        Retrieval and chunking's settings might undergo further expansion with
        the default configurations that the user did not specify.

        Parameters
        ----------
        params
            List of parameters to build up this search space.

        Returns
        -------
            Default search space overwritten and expanded (whenever necessary) with user-provided parameters.

        Raises
        ------
        SearchSpaceValueError
            When user provided unsupported parameter (not existing in the default search space).

        """
        for p in params:
            if p.name == "retrieval":
                expanded_settings = [
                    _
                    for user_setting in p.values
                    for _ in create_cartesian_product_of_possible_configurations(user_setting)
                ]
                expanded_settings = remove_duplicates(expanded_settings)
                p = Parameter(p.name, param_type="C", values=expanded_settings if expanded_settings else p.values)

            if p.name == "chunking":
                expanded_settings = [
                    _
                    for user_setting in p.values
                    for _ in create_cartesian_product_of_possible_configurations(user_setting)
                ]
                expanded_settings = remove_duplicates(expanded_settings)
                p = Parameter(p.name, param_type="C", values=expanded_settings if expanded_settings else p.values)

            if p.name in (
                    AI4RAGParamNames.AGENT_TYPE,
                    AI4RAGParamNames.INFERENCE_MODEL_ID,
                    AI4RAGParamNames.EMBEDDING_MODEL,
                    AI4RAGParamNames.RETRIEVAL,
                    AI4RAGParamNames.CHUNKING,
            ):
                default_search_space[p.name] = p
            else:
                # TO-DO change the error message here to more verbose sth like: not in allowed parameters sets
                raise SearchSpaceValueError("Search space parameters are invalid.")

        return list(default_search_space.values())

    @cached_property
    def combinations(self) -> list[dict]:
        """
        The flow of AutoRAG experiment requires that each combination is a flat mapping
        (e.g. due to application of OneHot encoding at some stage).

        Returns
        -------
            Flattened list of all parameter combinations for this search space.
        """
        combinations = list(filter(self.remove_react_combinations_for_byom_and_granite, super().combinations))
        return AI4RAGSearchSpace.flatten_combinations(combinations)

    @staticmethod
    def flatten_combinations(data: list[dict]) -> list[dict]:
        """
        Flattens each dictionary in the input list of possible parameter combinations for this search space.

        Parameters
        ----------
        data: list[dict]
            List of possible parameter combinations for this search space.

        Returns
        -------
            List of flattened possible parameter combinations for this search space.

        Notes
        -----
        Current implementation is temporary because it's very naive. The better, shorter
        and more general version will be provided shortly.
        """
        flattened_data = []
        for comb in data:
            if "score" in comb:
                # score is present when we're flattening evaluated data. Don't want to miss it.
                tmp_dct = {"score": comb["score"]}
            else:
                tmp_dct = {}
            if comb.get("chunking"):
                chunking = comb["chunking"]
                tmp_dct["chunking_method"] = chunking["method"]
                tmp_dct.update(chunking)
                del tmp_dct["method"]
                del comb["chunking"]
            if comb.get("retrieval"):
                retrieval = comb["retrieval"]
                tmp_dct["retrieval_method"] = retrieval["method"]
                tmp_dct["retrieval_window_size"] = retrieval["window_size"]
                tmp_dct["number_of_retrieved_chunks"] = retrieval["number_of_chunks"]
                del comb["retrieval"]
            tmp_dct |= comb
            flattened_data.append(tmp_dct)
        return flattened_data

    def __repr__(self) -> str:
        return f"{self.search_space}"
