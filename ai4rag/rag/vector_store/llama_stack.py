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
        self._collection_name = None
        self._ls_vs = self._initialize_ls_vector_store(
            client=client,
            embedding_model=embedding_model,
            provider_id=provider_id,
            reuse_collection_name=reuse_collection_name,
        )

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

        _vs = client.vector_stores.create(
            extra_body={
                "provider_id": provider_id,
                # "provider_vector_store_id": collection_name,  # --> not working in 0.4.x
                "embedding_model": embedding_model.model_id,
                "embedding_dimension": embedding_model.params["embedding_dimension"],
            }
        )

        return _vs

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def search(self, query: str, k: int, include_scores: bool = False) -> list[Document] | list[tuple[Document, float]]:
        """
        Search for the chunks relevant to the query.
        The method used will be simple similarity search.

        Parameters
        ----------
        query : str
            Question / query for which the similarity search will be executed.

        k : int
            Number of chunks to be returned as a result of similarity search

        include_scores : bool, default=False
            If True, similarity scores will be returned in the response

        Returns
        -------
        list[Document] | list[tuple[Document, float]]
            List of chunks as Document instances with or without scores, depending on the input.
        """
        params = {
            "max_chunks": k,
            "mode": "vector",  # keyword and hybrid supported as well
            # "ranker": {
            #     "strategy": "rrf",  # also weighted and normalized
            #     "params": {
            #         "k": 60,
            #         "weights": [0.5, 0.5]
            #     }
            # }
        }
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

        chunks = [
            {
                "content": doc.page_content,
                "chunk_metadata": doc.metadata,
                "chunk_id": doc.metadata["document_id"],
                "embedding_model": self.embedding_model.model_id,
                "embedding_dimension": self.embedding_model.params["embedding_dimension"],
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
        self.client.vector_stores.delete(self._ls_vs.id)
