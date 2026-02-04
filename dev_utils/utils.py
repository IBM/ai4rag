# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path

import pandas as pd


def read_benchmark_from_json(file_path: str | Path) -> pd.DataFrame:
    """
    A helper function to read benchmark data stored in a json file.

    Notes
    ------
    The json file is assumed to be of the form:
    {
        "data": [
            {
                "question": "q",
                "answers": ["a"],
                "document_ids": ["d1", "d2"]
            }
        ]
    }

    Parameters
    ----------
    file_path: str | Path
        Location of the benchmark file

    Returns
    -------
    Dataframe made of question, correct_answer and correct_answer_document_ids
    """
    with open(file_path, "r") as file:
        benchmark = json.load(file)
    df = pd.DataFrame.from_dict(data=benchmark)
    return df
