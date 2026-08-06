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

from ai4rag.components.utils import create_maas_client, create_maas_model_client
from ai4rag.components.utils.maas_client import maas_model_base_url
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.prepare.maas_utils import _list_maas_models, _model_owned_by


def create_dev_maas_client() -> OpenAI:
    """Create a general MaaS client from the ``MAAS_BASE`` / ``MAAS_API_KEY`` env vars.

    The general (list) endpoint lives at ``{MAAS_BASE}/maas-api/v1`` and is used
    to discover available models; per-model endpoints are derived from it.

    Returns
    -------
    OpenAI
        A connected general MaaS client.
    """
    base_url = f"{os.environ['MAAS_BASE']}/maas-api/v1"
    return create_maas_client(base_url=base_url, api_key=os.environ["MAAS_API_KEY"])


def build_maas_model(
    client: OpenAI,
    model_id: str,
    model_type: Literal["llm", "embedding"],
    embedding_params: dict | None = None,
) -> OpenAIFoundationModel | OpenAIEmbeddingModel:
    """Build a MaaS-backed model behind its own per-model OpenAI client.

    MaaS serves each model at its own OpenAI-compatible endpoint. This helper
    lists models on the general *client*, resolves the requested model's
    per-model URL, and instantiates the matching wrapper. Embedding parameters
    (dimension, context length) are auto-detected when *embedding_params* is
    omitted, since MaaS exposes no model metadata.

    Parameters
    ----------
    client : OpenAI
        General MaaS client (see :func:`create_dev_maas_client`).
    model_id : str
        Short model id (the last segment of the fully-qualified MaaS id).
    model_type : {"llm", "embedding"}
        Which wrapper to build.
    embedding_params : dict | None, default=None
        Optional explicit embedding parameters; auto-detected when omitted.

    Returns
    -------
    OpenAIFoundationModel | OpenAIEmbeddingModel
        The model bound to its per-model client.
    """
    registry = _list_maas_models(client)
    if model_id not in registry:
        raise ValueError(f"Model '{model_id}' is not available in MaaS. Available models: {sorted(registry)}.")

    per_model_client = create_maas_model_client(
        base_url=maas_model_base_url(client.base_url, _model_owned_by(registry[model_id])),
        api_key=client.api_key,
    )

    if model_type == "embedding":
        return OpenAIEmbeddingModel(model_id=model_id, client=per_model_client, params=embedding_params)
    return OpenAIFoundationModel(model_id=model_id, client=per_model_client)


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
