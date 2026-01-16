# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pandas as pd


def handle_missing_values_in_combinations_being_explored(df: pd.DataFrame):
    """
    Each record in the dataframe created to fit the model may differ.
    For single configuration one feature may not exist when other features takes specific value
    (i.e. for different retrieval types settings may differ).

    This function exists to handle dataframes for model training despite such differentiation.

    Params
    ------
    df : pd.DataFrame
        Experiment data, i.e. combinations being explored throughout the experiment.

    Returns
    -------
    pd.DataFrame
        Experiment data with `NaN` values properly replaced by more meaningful data.

    Notes
    -----
    This is basically only needed for fitting and transforming data using sklearn encoders.
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

    return df
