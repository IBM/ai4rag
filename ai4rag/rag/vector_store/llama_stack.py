# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from langchain_core.documents import Document
from llama_stack_client import LlamaStackClient
from llama_stack_client.types.vector_store import VectorStore

from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore


class LSVectorStore(BaseVectorStore):
    """LLamaStack client wrapper used for communication with vector store (single index/collection)."""

    _VALID_SEARCH_MODES = ("vector", "hybrid")
    _VALID_RANKER_STRATEGIES = ("rrf", "weighted", "normalized")

    def __init__(
        self,
        embedding_model: LSEmbeddingModel,
        client: LlamaStackClient,
        provider_id: str,
        reuse_collection_name: str | None = None,
        distance_metric: str | None = None,
    ):
        super().__init__(embedding_model, distance_metric, reuse_collection_name)
        self.client = client
        self._ls_vs = self._initialize_ls_vector_store(
            client=client,
            embedding_model=embedding_model,
            provider_id=provider_id,
            reuse_collection_name=reuse_collection_name,
        )
        self._collection_name = self._ls_vs.id

    @staticmethod
    def _initialize_ls_vector_store(
        client: LlamaStackClient, embedding_model: LSEmbeddingModel, provider_id: str, reuse_collection_name: str | None
    ) -> VectorStore:
        """
        Create or retrieve vector store instance via llama-stack.

        Parameters
        ----------
        client : LlamaStackClient
            Llama-stack client instance for communication with llama-stack.

        embedding_model : LSEmbeddingModel
            Wrapped llama-stack based embedding model with proper parameters.

        provider_id : str
            Provider id within the llama stack server.

        reuse_collection_name : str | None
            vector_store_id within llama-stack-server (usually collection name) to reuse (if already existing)

        Returns
        -------
        llama_stack_client.types.vector_store.VectorStore
            Instance for communication with llama-stack vector store.
        """

        if reuse_collection_name:
            _vs = client.vector_stores.retrieve(reuse_collection_name)
            return _vs

        # Handle both dict and LSEmbeddingParams for backward compatibility
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
        if search_mode not in LSVectorStore._VALID_SEARCH_MODES:
            raise ValueError(
                f"Invalid search_mode '{search_mode}'. Must be one of {LSVectorStore._VALID_SEARCH_MODES}."
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
            if ranker_strategy not in LSVectorStore._VALID_RANKER_STRATEGIES:
                raise ValueError(
                    f"Invalid ranker_strategy='{ranker_strategy}'. "
                    f"Must be one of {LSVectorStore._VALID_RANKER_STRATEGIES}."
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
            ranker = {"strategy": ranker_strategy, "params": {}}
            if ranker_k is not None and ranker_k > 0:
                ranker["params"]["k"] = ranker_k
            if ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
                ranker["params"]["alpha"] = ranker_alpha
            params["ranker"] = ranker

        resp = self.client.vector_io.query(query=query, vector_store_id=self._ls_vs.id, params=params)

        if include_scores:
            return [
                (Document(page_content=chunk.content, metadata=chunk.chunk_metadata.to_dict()), score)
                for chunk, score in zip(resp.chunks, resp.scores)
            ]

        return [Document(page_content=chunk.content, metadata=chunk.chunk_metadata.to_dict()) for chunk in resp.chunks]

    def add_documents(self, documents: list[Document]) -> None:
        """
        Add documents to the collection.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents to add to the collection.
        """

        # Handle both dict and LSEmbeddingParams for backward compatibility
        if isinstance(self.embedding_model.params, dict):
            embedding_dimension = self.embedding_model.params["embedding_dimension"]
        else:
            embedding_dimension = self.embedding_model.params.embedding_dimension

        chunks = [
            {
                "content": doc.page_content,
                "chunk_metadata": doc.metadata,
                "chunk_id": doc.metadata["document_id"],
                "embedding_model": self.embedding_model.model_id,
                "embedding_dimension": embedding_dimension,
            }
            for doc in documents
        ]
        embeddings = self.embedding_model.embed_documents([doc.page_content for doc in documents])
        full_chunks = [chunk | {"embedding": embedding} for chunk, embedding in zip(chunks, embeddings)]
        self.client.vector_io.insert(
            vector_store_id=self._ls_vs.id,
            chunks=full_chunks,
        )

    def clean_collection(self):
        """Remove content of the collection and remove vector store instance."""
        self.client.vector_stores.delete(self._ls_vs.id)
