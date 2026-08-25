# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
import os
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI

from ai4rag.components.utils import create_maas_client
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.prepare.models import _list_maas_model_ids


def create_dev_maas_client() -> OpenAI:
    """Create a MaaS client from the ``MAAS_BASE_URL`` / ``MAAS_API_KEY`` env vars.

    A single client serves everything: it lists available models and serves
    chat/embeddings for all of them. ``MAAS_BASE_URL`` is the **complete**
    OpenAI-compatible endpoint and is used **verbatim** — exactly as the
    generated notebooks consume it (``OpenAI(base_url=MAAS_BASE_URL, ...)``) — so
    the same value works for local runs and the produced artifacts alike.

    Returns
    -------
    OpenAI
        A connected MaaS client.
    """
    return create_maas_client(base_url=os.environ["MAAS_BASE_URL"], api_key=os.environ["MAAS_API_KEY"])


def build_maas_model(
    client: OpenAI,
    model_id: str,
    model_type: Literal["llm", "embedding"],
    embedding_params: dict | None = None,
) -> OpenAIFoundationModel | OpenAIEmbeddingModel:
    """Build a MaaS-backed model on the shared serving client.

    A single client serves every model, so this helper only checks that
    *model_id* is available and instantiates the matching wrapper on *client*.
    Embedding parameters (dimension, context length) are auto-detected when
    *embedding_params* is omitted, since MaaS exposes no model metadata.

    Parameters
    ----------
    client : OpenAI
        MaaS client (see :func:`create_dev_maas_client`).
    model_id : str
        The full model id, exactly as returned by ``models.list()``.
    model_type : {"llm", "embedding"}
        Which wrapper to build.
    embedding_params : dict | None, default=None
        Optional explicit embedding parameters; auto-detected when omitted.

    Returns
    -------
    OpenAIFoundationModel | OpenAIEmbeddingModel
        The model bound to the shared client.
    """
    available_ids = _list_maas_model_ids(client)
    if model_id not in available_ids:
        raise ValueError(f"Model '{model_id}' is not available in MaaS. Available models: {sorted(available_ids)}.")

    if model_type == "embedding":
        return OpenAIEmbeddingModel(model_id=model_id, client=client, params=embedding_params)
    return OpenAIFoundationModel(model_id=model_id, client=client)


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
    with open(file_path, "r", encoding="utf-8") as file:
        benchmark = json.load(file)
    df = pd.DataFrame.from_dict(data=benchmark)
    return df
