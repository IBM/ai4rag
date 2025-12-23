# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Literal

from langchain_core.documents import Document

from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore


class Retriever:
    def __init__(
        self,
        vector_store: BaseVectorStore,
        method: Literal["simple"],
        number_of_chunks: int,
    ):
        self._vector_store = vector_store
        self.method = method
        self.number_of_chunks = number_of_chunks

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

        return self._vector_store.search(query, k=_number_of_chunks)
