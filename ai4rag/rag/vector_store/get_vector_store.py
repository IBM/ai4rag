# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from .base_vector_store import BaseVectorStore
from .chroma import ChromaVectorStore

from llama_stack_client import LlamaStackClient

from ..embedding.base_model import EmbeddingModel


def get_vector_store(
    vs_type: str,
    embedding_model: EmbeddingModel,
    ls_client: LlamaStackClient | None = None,
) -> BaseVectorStore:
    """Get vector store of desired type with chosen settings.

    Parameters
    ----------
    vs_type : str
        Type of vector store.

    embedding_model : EmbeddingModel
        Embedding model used for the embeddings creation in the created vector store instance.

    ls_client : LlamaStackClient | None, default=None
        Instance of the llama stack client to communicate with registered vector databases.

    Returns
    -------
    BaseVectorStore
        Instance of the vector store.
    """

    match vs_type:
        case "chroma":
            vs = ChromaVectorStore(
                embedding_model=embedding_model,
            )

        case _:
            raise ValueError(f"Vector store of type {vs_type} is not supported.")

    return vs
