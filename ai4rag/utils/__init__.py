# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections import deque
from collections.abc import Hashable
from math import floor
from typing import Sequence

import pandas as pd


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
