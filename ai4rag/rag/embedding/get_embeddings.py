# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from ai4rag.rag.embedding.base_model import EmbeddingModel
from ai4rag.rag.embedding.mocked import MockedEmbeddingModel


def get_embeddings(
    provider: str, model_id: str, embedding_params: dict[str, Any] | None = None, **kwargs
) -> EmbeddingModel:
    """
    Create instance of EmbeddingModel.

    Parameters
    ----------
    provider : str
        The source of embedding model.

    model_id : str
        ID of the embedding model to use.

    embedding_params : dict[str, Any] | None, default=None
        Parameters for the embedding.

    Returns
    -------
    EmbeddingModel
        Instance of the embedding model wrapper.
    """

    params = embedding_params or {}

    match provider:
        case _:
            embeddings = MockedEmbeddingModel(model_id=model_id, params=params, **kwargs)

    return embeddings
