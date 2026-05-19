# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Literal

from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore


class Retriever:
    """Class responsible for retrieving data from given vector store.

    Parameters
    ----------
    vector_store : BaseVectorStore
        Vector store / vector index to retrieve data from.

    method : Literal["simple"]
        Method describing how data should be retrieved.

    number_of_chunks : int
        Number of chunks to retrieve.

    search_mode : Literal["vector", "hybrid"], default="vector"
        Search mode passed to the vector store: "vector" or "hybrid".

    ranker_strategy : str | None, default=None
        Ranking strategy for hybrid search: "rrf", "weighted", or "normalized".

    ranker_k : int | None, default=None
        Parameter k for the ranking function.

    ranker_alpha : float | None, default=None
        Alpha parameter for weighted ranking strategy.
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        number_of_chunks: int,
        method: Literal["simple"] = "simple",
        search_mode: Literal["vector", "hybrid"] = "vector",
        ranker_strategy: str | None = None,
        ranker_k: int | None = None,
        ranker_alpha: float | None = None,
    ):
        self._vector_store = vector_store
        self.method = method
        self.number_of_chunks = number_of_chunks
        self.search_mode = search_mode
        self.ranker_strategy = ranker_strategy
        self.ranker_k = ranker_k
        self.ranker_alpha = ranker_alpha

    @property
    def collection_name(self):
        """Exposes the collection_name this retriever seraches through."""
        return self._vector_store.collection_name

    def retrieve(self, query: str, **kwargs) -> list[dict]:
        """Retrieve relevant documents from vector store.

        Parameters
        ----------
        query : str
            question for which documents should be retrieved.

        Returns
        -------
        list[Document]
            list of documents with their metadata corresponding to the query.
        """
        _number_of_chunks = kwargs.get("number_of_chunks", self.number_of_chunks)

        return self._vector_store.search(
            query,
            k=_number_of_chunks,
            search_mode=self.search_mode,
            ranker_strategy=self.ranker_strategy,
            ranker_k=self.ranker_k,
            ranker_alpha=self.ranker_alpha,
        )
