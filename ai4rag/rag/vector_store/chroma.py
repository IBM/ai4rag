#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024-2025.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .base_vector_store import BaseVectorStore
from .utils import merge_window_into_a_document

from ..embedding.base_model import BaseEmbeddingModel


class ChromaVectorStore(BaseVectorStore):
    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        collection_name: str,
        distance_metric: str,
        document_name_field: str = "document_id",
        chunk_sequence_number_field: str = "sequence_number",
    ) -> None:
        super().__init__(
            embedding_model=embedding_model, collection_name=collection_name, distance_metric=distance_metric
        )
        self._document_name_field = document_name_field
        self._chunk_sequence_number_field = chunk_sequence_number_field

    def get_client(self) -> Chroma:
        return super().get_client()

    def clear(self) -> None:
        client = self.get_client()
        all_docs_ids = client.get()["ids"]
        if len(all_docs_ids) > 0:
            self.delete(all_docs_ids)

    def count(self) -> int:
        client = self.get_client()
        return len(client.get()["ids"])

    def add_documents(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        max_batch_size = kwargs.get("max_batch_size")
        if max_batch_size is None:
            try:
                max_batch_size = (
                    self._langchain_vector_store._client.get_max_batch_size()
                )  # type: ignore[attr-defined]
            except AttributeError:
                max_batch_size = 10_000

        ids, docs = self._process_documents(content)
        if len(docs) > max_batch_size:
            batch_ids = []

            for batch_start in range(0, len(docs), max_batch_size):
                batch_ids.extend(
                    self._langchain_vector_store.add_documents(
                        docs[batch_start : batch_start + max_batch_size],
                        ids=ids[batch_start : batch_start + max_batch_size],
                        **kwargs,
                    )
                )
            return batch_ids
        else:
            return self._langchain_vector_store.add_documents(docs, ids=ids, **kwargs)

    def _get_window_documents(
        self, doc_id: str, seq_nums_window: list[int]
    ) -> list[Document]:
        """
        Receives a document ID and a list of chunks' sequence_numbers,
        and searches the vector store according to the metadata.

        :param doc_id: ID of document
        :type doc_id: str

        :param seq_nums_window: list of sequence numbers
        :type seq_nums_window: list[int]

        :return: list of documents from that document with these sequence_numbers
        :rtype: list[Document]
        """
        expr = {
            "$and": [
                {self._document_name_field: {"$eq": doc_id}},
                {self._chunk_sequence_number_field: {"$gte": seq_nums_window[0]}},
                {self._chunk_sequence_number_field: {"$lte": seq_nums_window[-1]}},
            ]
        }
        res = self._langchain_vector_store.get(where=expr)  # type: ignore[arg-type]
        texts, metadatas = res["documents"], res["metadatas"]
        window_documents = [
            Document(
                page_content=text,
                metadata={
                    self._chunk_sequence_number_field: metadata[
                        self._chunk_sequence_number_field
                    ],
                    self._document_name_field: doc_id,
                },
            )
            for text, metadata in zip(texts, metadatas)
        ]
        return window_documents
