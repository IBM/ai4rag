# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from llama_stack_client import LlamaStackClient

from ..embedding.base_model import BaseEmbeddingModel
from .base_vector_store import BaseVectorStore
from .chroma import ChromaVectorStore
from .llama_stack import LSVectorStore


def get_vector_store(
    vs_type: str,
    embedding_model: BaseEmbeddingModel,
    reuse_collection_name: str | None = None,
    client: LlamaStackClient | None = None,
) -> BaseVectorStore:
    """Get vector store of desired type with chosen settings.

    Parameters
    ----------
    vs_type : str
        Type of vector store.

    embedding_model : BaseEmbeddingModel
        Embedding model used for the embeddings creation in the created vector store instance.

    reuse_collection_name : str | None, default=None
        Name of the collection that will be created in the vector database.

    client : LlamaStackClient | None, default=None
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
                reuse_collection_name=reuse_collection_name,
            )

        case str() as vs_type if vs_type.startswith("ls_") and len(vs_type) > 3:
            provider_id = vs_type[3:]
            vs = LSVectorStore(
                embedding_model=embedding_model,
                reuse_collection_name=reuse_collection_name,
                distance_metric="cosine",
                client=client,
                provider_id=provider_id,
            )

        case _:
            raise ValueError(f"Vector store of type {vs_type} is not supported.")

    return vs
