# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from ogx_client import OgxClient

from ..embedding.base_model import BaseEmbeddingModel
from .base_vector_store import BaseVectorStore
from .chroma import ChromaVectorStore
from .ogx import OGXVectorStore


def get_vector_store(
    vs_type: str,
    embedding_model: BaseEmbeddingModel,
    reuse_collection_name: str | None = None,
    client: OgxClient | None = None,
    ogx_vector_io_provider_id: str | None = None,
) -> BaseVectorStore:
    """Get vector store of desired type with chosen settings.

    Parameters
    ----------
    vs_type : str
        Type of vector store. Supported values: ``"ogx"`` and ``"chroma"``.

    embedding_model : BaseEmbeddingModel
        Embedding model used for the embeddings creation in the created vector store instance.

    reuse_collection_name : str | None, default=None
        Name of the collection that will be created in the vector database.

    client : OgxClient | None, default=None
        Instance of the OGX client to communicate with registered vector databases.

    ogx_vector_io_provider_id : str | None, default=None
        Provider ID for OGX vector store (e.g., ``"milvus"``, ``"qdrant"``).
        Required when ``vs_type="ogx"``.

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

        case "ogx":
            if not ogx_vector_io_provider_id:
                raise ValueError("ogx_vector_io_provider_id is required when vector_store_type is 'ogx'.")
            vs = OGXVectorStore(
                embedding_model=embedding_model,
                reuse_collection_name=reuse_collection_name,
                distance_metric="cosine",
                client=client,
                provider_id=ogx_vector_io_provider_id,
            )

        case _:
            raise ValueError(f"Vector store of type {vs_type} is not supported.")

    return vs
