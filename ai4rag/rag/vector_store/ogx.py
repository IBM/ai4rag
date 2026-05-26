# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from langchain_core.documents import Document
from ogx_client import OgxClient
from ogx_client.types.vector_store import VectorStore

from ai4rag import logger
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore


class OGXVectorStore(BaseVectorStore):
    """OGX client wrapper used for communication with vector store (single index/collection)."""

    _VALID_SEARCH_MODES = ("vector", "hybrid")
    _VALID_RANKER_STRATEGIES = ("rrf", "weighted", "normalized")

    def __init__(
        self,
        embedding_model: OGXEmbeddingModel,
        client: OgxClient,
        provider_id: str,
        reuse_collection_name: str | None = None,
        distance_metric: str | None = None,
    ):
        super().__init__(embedding_model, distance_metric, reuse_collection_name)
        self.client = client
        self._ogx_vs = self._initialize_ogx_vector_store(
            client=client,
            embedding_model=embedding_model,
            provider_id=provider_id,
            reuse_collection_name=reuse_collection_name,
        )
        self._collection_name = self._ogx_vs.id

    @staticmethod
    def _initialize_ogx_vector_store(
        client: OgxClient,
        embedding_model: OGXEmbeddingModel,
        provider_id: str,
        reuse_collection_name: str | None,
    ) -> VectorStore:
        """
        Create or retrieve vector store instance via OGX.

        Parameters
        ----------
        client : OgxClient
            OGX client instance for communication with OGX server.

        embedding_model : OGXEmbeddingModel
            Wrapped OGX based embedding model with proper parameters.

        provider_id : str
            Provider id within the OGX server.

        reuse_collection_name : str | None
            vector_store_id within OGX server (usually collection name) to reuse (if already existing)

        Returns
        -------
        ogx_client.types.vector_store.VectorStore
            Instance for communication with OGX vector store.
        """

        if reuse_collection_name:
            _vs = client.vector_stores.retrieve(reuse_collection_name)
            return _vs

        # Handle both dict and OGXEmbeddingParams for backward compatibility
        if isinstance(embedding_model.params, dict):
            embedding_dimension = embedding_model.params["embedding_dimension"]
        else:
            embedding_dimension = embedding_model.params.embedding_dimension

        _vs = client.vector_stores.create(
            extra_body={
                "provider_id": provider_id,
                "embedding_model": embedding_model.model_id,
                "embedding_dimension": embedding_dimension,
            }
        )

        return _vs

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @staticmethod
    def _validate_search_params(
        search_mode: str,
        ranker_strategy: str | None,
        ranker_k: int | None,
        ranker_alpha: float | None,
    ) -> None:
        """Validate hybrid search parameter consistency."""
        if search_mode not in OGXVectorStore._VALID_SEARCH_MODES:
            raise ValueError(
                f"Invalid search_mode '{search_mode}'. Must be one of {OGXVectorStore._VALID_SEARCH_MODES}."
            )

        has_strategy = ranker_strategy is not None and ranker_strategy != ""
        has_k = ranker_k is not None and ranker_k > 0
        has_alpha = ranker_alpha is not None and ranker_alpha != 1

        if search_mode != "hybrid":
            if has_strategy:
                raise ValueError(
                    f"ranker_strategy='{ranker_strategy}' is only valid when search_mode='hybrid', "
                    f"but search_mode='{search_mode}'."
                )
            if has_k:
                raise ValueError(
                    f"ranker_k={ranker_k} is only valid when search_mode='hybrid', " f"but search_mode='{search_mode}'."
                )
            if has_alpha:
                raise ValueError(
                    f"ranker_alpha={ranker_alpha} is only valid when search_mode='hybrid', "
                    f"but search_mode='{search_mode}'."
                )
        else:
            if not has_strategy:
                raise ValueError("ranker_strategy must be set when search_mode='hybrid'.")
            if ranker_strategy not in OGXVectorStore._VALID_RANKER_STRATEGIES:
                raise ValueError(
                    f"Invalid ranker_strategy='{ranker_strategy}'. "
                    f"Must be one of {OGXVectorStore._VALID_RANKER_STRATEGIES}."
                )
            if has_k and ranker_strategy != "rrf":
                raise ValueError(
                    f"ranker_k={ranker_k} is only valid when ranker_strategy='rrf', "
                    f"but ranker_strategy='{ranker_strategy}'."
                )
            if has_alpha and ranker_strategy != "weighted":
                raise ValueError(
                    f"ranker_alpha={ranker_alpha} is only valid when ranker_strategy='weighted', "
                    f"but ranker_strategy='{ranker_strategy}'."
                )

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        search_mode: str = "vector",
        ranker_strategy: str | None = None,
        ranker_k: int | None = None,
        ranker_alpha: float | None = None,
        **kwargs,
    ) -> list[Document] | list[tuple[Document, float]]:
        """
        Search for the chunks relevant to the query.

        Parameters
        ----------
        query : str
            Question / query for which the similarity search will be executed.

        k : int
            Number of chunks to be returned as a result of similarity search

        include_scores : bool, default=False
            If True, similarity scores will be returned in the response

        search_mode : str, default="vector"
            Search mode: "vector" or "hybrid".

        ranker_strategy : str | None, default=None
            Ranking strategy for hybrid search: "rrf", "weighted", or "normalized".
            Empty string means no ranker (used for non-hybrid modes).

        ranker_k : int | None, default=None
            Parameter k for the ranking function. 0 means not set.

        ranker_alpha : float, default=None
            Alpha parameter for weighted ranking strategy. 1 means not set (vector-only sentinel).

        Returns
        -------
        list[Document] | list[tuple[Document, float]]
            List of chunks as Document instances with or without scores, depending on the input.
        """
        self._validate_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha)
        params = {
            "max_chunks": k,
            "mode": search_mode,
        }

        if search_mode == "hybrid" and ranker_strategy:
            params["reranker_type"] = ranker_strategy
            reranker_params = {}
            if ranker_strategy == "rrf" and ranker_k is not None and ranker_k > 0:
                reranker_params["impact_factor"] = ranker_k
            if ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
                reranker_params["alpha"] = ranker_alpha
            params["reranker_params"] = reranker_params

        resp = self.client.vector_io.query(query=query, vector_store_id=self._ogx_vs.id, params=params)

        if include_scores:
            return [
                (
                    Document(
                        page_content=chunk.content,
                        metadata=chunk.chunk_metadata.to_dict(),
                    ),
                    score,
                )
                for chunk, score in zip(resp.chunks, resp.scores)
            ]

        return [Document(page_content=chunk.content, metadata=chunk.chunk_metadata.to_dict()) for chunk in resp.chunks]

    def add_documents(self, documents: list[Document], **kwargs) -> None:
        """
        Add documents to the collection.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents to add to the collection.
        """
        batch_size = kwargs.get("batch_size", 2048)

        # Handle both dict and OGXEmbeddingParams for backward compatibility
        if isinstance(self.embedding_model.params, dict):
            embedding_dimension = self.embedding_model.params["embedding_dimension"]
        else:
            embedding_dimension = self.embedding_model.params.embedding_dimension

        chunks = [
            {
                "content": doc.page_content,
                "chunk_metadata": {
                    "document_id": doc.metadata["document_id"],
                },
                "metadata": doc.metadata,
                "chunk_id": str(
                    hash(f"{doc.metadata.get('document_id')}_{doc.metadata.get('start_index')}_{doc.page_content}")
                ),
                "embedding_model": self.embedding_model.model_id,
                "embedding_dimension": embedding_dimension,
            }
            for doc in documents
        ]

        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])

        seen_ids = set()
        unique_chunks = []
        unique_embeddings = []

        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["chunk_id"]
            if chunk_id in seen_ids:
                logger.warning(
                    f"Skipping duplicate chunk_id: {chunk_id} from document: {chunk['chunk_metadata']['document_id']}"
                )
                continue
            seen_ids.add(chunk_id)
            unique_chunks.append(chunk)
            unique_embeddings.append(embedding)

        full_chunks = [chunk | {"embedding": embedding} for chunk, embedding in zip(unique_chunks, unique_embeddings)]

        for idx in range(0, len(full_chunks), batch_size):
            self.client.vector_io.insert(
                vector_store_id=self._ogx_vs.id,
                chunks=full_chunks[idx : idx + batch_size],
            )

    def clean_collection(self):
        """Remove content of the collection and remove vector store instance."""
        self.client.vector_stores.delete(self._ogx_vs.id)
