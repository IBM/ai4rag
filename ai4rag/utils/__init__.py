# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections import deque
from collections.abc import Hashable
from datetime import datetime
from math import floor
from textwrap import dedent
from typing import Sequence

import jinja2
import pandas as pd
from ibm_watsonx_ai.metanames import GenChatParamsMetaNames, GenTextParamsMetaNames


def get_hashable_repr(dct: dict):
    """
    Returns
    -------
    A hashable representation of the provided dictionary.
    """
    queue = deque((k, v, 0, None) for k, v in dct.items())
    dict_unpacked = []
    while queue:
        key, val, lvl, p_ref = queue.pop()
        if hasattr(val, "items"):  # we have a nested dict
            dict_unpacked.append((key, "+", lvl, p_ref))  # key is an aggregator at this level (that's why '+')
            if hash(key) != p_ref:  # but it could be an aggregator for a Sequence (and not other dict)
                lvl += 1
            queue.extendleft((k, v, lvl, hash(key)) for k, v in val.items())
        elif isinstance(val, Hashable):
            dict_unpacked.append((key, val, lvl, p_ref))
        elif isinstance(val, Sequence):
            # only sequences supported now
            dict_unpacked.append((key, "+", lvl, p_ref))

            queue.extendleft((key, vv, floor(lvl) + ind * 0.01, hash(key)) for ind, vv in enumerate(val, 1))

        else:
            raise ValueError(f"Some value in the provided dict is not supported. {type(val)} is not supported")

    return tuple(sorted(dict_unpacked, key=lambda it: (it[2], it[0])))


def remove_duplicates(items: list[dict]) -> list[dict]:
    """
    Deduplicates list of provided items. As for now only supported are dictionary members.
    They must be also supported by `get_hashable_repr` function.

    Parameters
    ----------
    items : list[dict]
        List of items to deduplicate. Currently only dictionaries are supported.

    Returns
    -------
    list[dict]
        A deduplicated list of input items.
    """
    duplicate_tracker = set()
    deduplicated_items = []
    for ind, elem in enumerate(map(get_hashable_repr, items)):
        if elem not in duplicate_tracker:
            duplicate_tracker.add(elem)
            deduplicated_items.append(items[ind])
    return deduplicated_items


def handle_missing_values_in_combinations_being_explored(df: pd.DataFrame):
    """
    With the support for hybrid search and semantic chunker the retrieval
    and chunking settings started to differ between themselves
    in terms of configurations (hybrid search can be ON for some and OFF for others,
    while chunking settings for semantic chunker doesn't contain chunk overlap).
    This results in the situation where combinations explored throughout ai4rag experiment might also differ
    by `hybrid_ranker` or `chunking` related configurations.
    This function unifies experiment data by making sure each combination is complete, i.e. has appropriate value
    for each configuration checked throughout whole experiment.

    Params
    ------
    df: pd.DataFrame
        Experiment data, i.e. combinations being explored throughout the experiment.

    Returns
    -------
        Experiment data with `NaN` values properly replaced by more meaningful data.

    Notes
    -----
    This is basically only needed for fitting and transforming data using sklearn.OneHotEncoding.

    """
    if "ranker_strategy" in df.columns:
        df["ranker_strategy"] = df["ranker_strategy"].map(lambda el: "" if pd.isna(el) else el)
        df["ranker_sparse_vectors"] = (
            df["ranker_sparse_vectors"]
            .map(lambda el: str(el) if isinstance(el, list) else el)
            .map(lambda el: "" if pd.isna(el) else el)
        )

    for hybrid_numerical_col in ("ranker_k", "ranker_alpha"):
        if hybrid_numerical_col in df.columns:
            df[hybrid_numerical_col] = df[hybrid_numerical_col].map(lambda el: -1 if pd.isna(el) else el)

    if "chunk_overlap" in df.columns:
        df["chunk_overlap"] = df["chunk_overlap"].map(lambda el: 0 if pd.isna(el) else el)

    return df


def datetime_str_to_epoch_time(timestamp: str | int) -> str | int:
    """
    If `timestamp` is a datetime strings try to parse it to an epoch time.
    Currenty only ISO8601 format strings are supported as this should be usually
    returned (if at all as a datetime string) by elasticsearch.

    Parameters
    ----------
    timestamp : str | int
        Either a datetime string or an unix timestamp.

    Returns
    -------
        Either an unchanged unix timestamp, datetime string parsed to a unix timestamp or -1 if parsing fails.
    """
    if not isinstance(timestamp, str):
        return timestamp
    try:
        iso_parseable = datetime.fromisoformat(timestamp)
    except ValueError:
        return -1
    return int(iso_parseable.timestamp())


def create_gen_params_dict(params: dict) -> dict:
    """
    Creates dictionary containing generation parameters.

    Parameters
    ----------
    params : dict
        Input parameters from which generation parameters dictionary is created.

    Returns
    -------
        Filtered items from input params which are in GenParams.
    """
    return {
        key: value
        for key, value in params.items()
        if key.upper() in GenTextParamsMetaNames().get()
        or key.upper() in GenChatParamsMetaNames().get()  # remove GenText in CPD 5.3.0
    }


def _dedent(value):
    return dedent(value.expandtabs(2))


def _prepare_template():
    """
    A closure following singleton and lazy initialisation principles. Encompasses dict holding jinja related objects.
    This will ensure that any templates loaded throughout the codebase share the same configuration environment
    while not wasting time for unnecessary re-initialisation of jinja objects.
    """

    jinja_objects = {"env": None, "loader": None}

    def _render_template(name: str, **kwargs) -> str:
        """
        Loads and renders the template specified by `name`.
        If jinja objects are not yet created then instantiates them first.

        Arguments
        ---------
        name : str
            Name of the template to be loaded.

        kwargs
            Passed to `Template.render()` function as-is.

        Returns
        -------
        str
            A templated string.
        """
        if not jinja_objects["loader"]:
            jinja_objects["loader"] = jinja2.PackageLoader("ai4rag.core.ai_service", package_path="function_templates")
        if not jinja_objects["env"]:
            jinja_objects["env"] = jinja2.Environment(
                loader=jinja_objects["loader"], trim_blocks=True, lstrip_blocks=True
            )
            jinja_objects["env"].filters["dedent"] = _dedent

        templ = jinja_objects["env"].get_template(name)

        return templ.render(**kwargs)

    return _render_template


prepare_template = _prepare_template()
