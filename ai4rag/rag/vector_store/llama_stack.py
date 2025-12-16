from llama_stack_client import LlamaStackClient

from .base_vector_store import BaseVectorStore
from ai4rag.rag.embedding.llama_stack import EmbeddingModel


class LlamaStackVectorStore(BaseVectorStore):
    """LLamaStack client wrapper used for communication with vector store (single index/collection)."""

    def __init__(self, embedding_model: EmbeddingModel, collection_name: str, distance_metric: str, ls_client: LlamaStackClient):
        super().__init__(embedding_model, collection_name, distance_metric)
        self.ls_client = ls_client

    def search(self, query: str, k: int) -> list[dict]:
        """
        Search for the chunks relevant to the query.
        The method used will be simple similarity search.

        Parameters
        ----------
        query : str
            Question / query for which the similarity search will be executed.

        k : int
            Number of chunks to be returned as a result of similarity search

        Returns
        -------
        list[dict]
            List of chunks as dicts with content and metadata.
        """

        return [{"content": "Content of the document.", "metadata": {"document_id": "doc-id"}}]

    def add_documents(self, documents: list[dict]) -> None:
        """
        Add documents to the collection.

        Parameters
        ----------
        documents : Sequence[Document]
            Documents to add to the collection.
        """
        print("DOCUMENTS ADDED TO THE COLLECTION")
