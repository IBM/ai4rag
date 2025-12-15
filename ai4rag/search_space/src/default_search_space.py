# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from itertools import product
from typing import Optional

from ibm_watsonx_ai import APIClient

from ai4rag import logger
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = ["get_default_ai4rag_search_space"]

_default_recursive_chunking_configs = {"method": "recursive", "chunk_size": (1024, 2048), "chunk_overlap": (256, 512)}

_default_simple_retrieval_configs = {
    "method": "simple",
    "window_size": 0,
    "number_of_chunks": (3, 5),
}

_default_window_retrieval_configs = {
    "method": "window",
    "window_size": range(1, 5),
    "number_of_chunks": (3, 5),
}


def create_cartesian_product_of_possible_configurations(configs: dict, defaults: Optional[dict] = None) -> list[dict]:
    """
    By default, this function creates a cartesian product of all configurations provided via `configs` parameter.

    When passed optional `defaults` dictionary it updates the `configs` dict with missing configurations
    found in `defaults` and only then proceeds to creating the cartesian product.

    Parameters
    ----------
    configs : dict
        A mapping of configuration names to all values they might, take.

    defaults : dict | None
        Optional set of configurations to use for specifying default configurations missing from `configs`.

    Returns
    -------
    list[dict]
        A cartesian product of provided configurations. Expanded with missing defaults if `defaults` provided.

    Examples
    --------
    >>> configs = {
    ...    "size": 10
    ...    "overlap": (20, 30)
    ... }
    >>> create_cartesian_product_of_possible_configurations(configs)
    [{"size": 10, "overlap": 20}, {"size": 10, "overlap": 30}]

    Passing `defaults` to expand the resulting settings
    >>> defaults = {
    ...     "method": "recursive"
    ... }
    >>> create_cartesian_product_of_possible_configurations(configs, defaults)
    [{"size": 10, "overlap": 20, "method": "recursive"}, {"size": 10, "overlap": 30, "method": "recursive"}]

    """
    user_configs = dict(configs.items())
    # pylint: disable=unnecessary-lambda-assignment
    # settings should always consist of all possible configurations (except for `hybrid_ranker` as it can be turned off)
    if not defaults:
        if "recursive" in (method := configs.get("method", "")):
            defaults = _default_recursive_chunking_configs
        elif "simple" in method:
            defaults = _default_simple_retrieval_configs
        elif "window" in method:
            defaults = _default_window_retrieval_configs
        else:
            # `method` is a required option in both `retrieval` and `chunking` parameters
            # so most probably we should be able to guess either config.
            logger.warning(
                "Default configurations were not provided. Could not tell if creating cartesian "
                "product for `chunking` or `retrieval` configurations either. "
                "Resulting settings will include only configurations provided as `configs` positional param."
            )
            defaults = {}

        defaults = {k: v for k, v in defaults.items() if k != "hybrid_ranker"}

    for k in defaults.keys() - user_configs.keys():
        user_configs[k] = defaults[k]  # update with missing defaults

    # for cartesian product to be successful each key has to be a sequence
    ensure_is_sequence = lambda elem: (elem,) if not isinstance(elem, (list, tuple, range)) else elem
    configs_values_as_sequences = map(ensure_is_sequence, user_configs.values())

    return [dict(zip(user_configs.keys(), cart_prod)) for cart_prod in product(*configs_values_as_sequences)]


def get_default_ai4rag_search_space(
    api_client: APIClient,
    retries: int = 3,
    delay: int = 2,
    backoff: int = 3,
) -> dict[str, Parameter]:
    """
    Function to return default search space containing experiment parameters.

    Parameters
    ----------
    api_client : APIClient
        Instance of the client for communication with watsonx.ai services.

    retries : int, optional
        Number of times to retry request using cached function `_get_foundation_models_specs`
        before fallback to the call via APIClient.

    delay : int, optional
        Number of seconds to wait between retries.

    backoff : int, optional
        The factor by which delay is increased after each retry.


    Returns
    -------
    list[Parameter]
        Parameters that will be used for creating AI4RAGSearchSpace
    """

    default_search_space = {
        AI4RAGParamNames.AGENT_TYPE: Parameter(
            name=AI4RAGParamNames.AGENT_TYPE, param_type="C", values=["sequential"]
        ),
        AI4RAGParamNames.RETRIEVAL: Parameter(
            name=AI4RAGParamNames.RETRIEVAL,
            param_type="C",
            values=[
                *create_cartesian_product_of_possible_configurations(
                    _default_simple_retrieval_configs, defaults=_default_simple_retrieval_configs
                ),
                *create_cartesian_product_of_possible_configurations(
                    _default_window_retrieval_configs, defaults=_default_window_retrieval_configs
                ),
            ],
        ),
        AI4RAGParamNames.EMBEDDING_MODEL: Parameter(
            name=AI4RAGParamNames.EMBEDDING_MODEL,
            param_type="C",
            values=[
                model_spec["model_id"]
                for model_spec in get_ai4rag_models_specs(
                    api_client,
                    models_type="embedding",
                    retries=retries,
                    delay=delay,
                    backoff=backoff,
                )
            ],
        ),
        AI4RAGParamNames.CHUNKING: Parameter(
            name=AI4RAGParamNames.CHUNKING,
            param_type="C",
            values=[
                *create_cartesian_product_of_possible_configurations(
                    _default_semantic_chunking_configs, defaults=_default_semantic_chunking_configs
                ),
                *create_cartesian_product_of_possible_configurations(
                    _default_recursive_chunking_configs, defaults=_default_recursive_chunking_configs
                ),
            ],
        )
    }

    # In case where there are no IBM provided models
    if ibm_provided_models := [
        model_spec["model_id"]
        for model_spec in get_ai4rag_models_specs(
            api_client,
            models_type="foundation",
            retries=retries,
            delay=delay,
            backoff=backoff,
        )
    ]:
        default_search_space.update(
            {
                AI4RAGParamNames.INFERENCE_MODEL_ID: Parameter(
                    name=AI4RAGParamNames.INFERENCE_MODEL_ID,
                    param_type="C",
                    values=ibm_provided_models,
                )
            }
        )
    return default_search_space
