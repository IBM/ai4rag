#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Any, cast
import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .base_vector_store import BaseVectorStore
from .utils import merge_window_into_a_document

from ..embedding.base_model import EmbeddingModel

from ai4rag import logger


class ChromaVectorStore(BaseVectorStore):
    """
    Class representing single index in the chroma vector database.

    Parameters
    ----------
    embedding_model : EmbeddingModel
        Instance used for embedding documents and user's queries.

    collection_name : str, default="default_collection"
        Name of the collection that will be created as a vector store.

    distance_metric : str, default="cosine"
        Metric that will be used to calculate similarity score between vectors.

    document_name_field : str, default="document_id"
        Default document ID field name.

    chunk_sequence_number_field : str, default="chunk_sequence_number"
        Default chunk sequence number field name.
    """

    _supported_distance_metrics = ("cosine", "l2")

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        collection_name: str = "default_collection",
        distance_metric: str = "cosine",
        document_name_field: str = "document_id",
        chunk_sequence_number_field: str = "sequence_number",
        **kwargs,
    ) -> None:
        super().__init__(
            embedding_model=embedding_model, collection_name=collection_name, distance_metric=distance_metric
        )
        self._document_name_field = document_name_field
        self._chunk_sequence_number_field = chunk_sequence_number_field
        self._vector_store = self._get_chroma_client(**kwargs)

    def _get_chroma_client(self, **kwargs) -> Chroma:
        """
        Create chroma client based on the given settings.

        ^kwargs are passed from the __init__ as parameters for Chroma client.

        Returns
        -------
        Chroma
            Client instance created based on the given settings.
        """

        chroma_client = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_model,
            collection_metadata={"hnsw:space": self.distance_metric},
            **kwargs,
        )

        return chroma_client

    @property
    def distance_metric(self) -> str:
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, value: str) -> None:
        """Set value of the distance metric.

        Raises
        ------
        ValueError
            If the distance metric is not supported.
        """
        if value not in self._supported_distance_metrics:
            raise ValueError(f"Invalid distance metric: {value}. Use one of: {self._supported_distance_metrics}.")
        self._distance_metric = value

    def clear(self) -> None:
        all_docs_ids = self._vector_store.get()["ids"]
        if len(all_docs_ids) > 0:
            self.delete(all_docs_ids)

    def count(self) -> int:
        """Count the number of shards in the vector store.

        Returns
        -------
        int
            Number of shards in the vector store.
        """
        return len(self._vector_store.get()["ids"])

    @staticmethod
    def _as_langchain_documents(content: list) -> list[Document]:
        """Creates a LangChain ``Document`` list from a list of potentially unstructured data.

        Parameters
        ----------
        content : list
            Unstructured data to be parsed.

        Returns
        -------
        list[Document]
            List of `Document` instances.
        """
        result = []
        for doc in content:
            if isinstance(doc, str):
                result.append(Document(page_content=doc))
            elif isinstance(doc, dict):
                content_str: str | None = doc.get("content", None)
                metadata = doc.get("metadata", {})

                if content_str:
                    if isinstance(metadata, dict):
                        result.append(Document(page_content=content_str, metadata=metadata))
                    else:
                        logger.warning(
                            f"Document: {doc} is incorrect. Metadata needs to be given with 'metadata' attribute and it needs to be a serializable dict. Skipping."
                        )
                        continue
                else:
                    logger.warning(f"Document: {doc} is incorrect. Field 'content' is required")
                    continue
            else:
                try:
                    result.append(Document(page_content=doc.page_content, metadata=doc.metadata))
                except AttributeError:
                    logger.warning(
                        f"Document: {doc} is not a dict, nor string, nor LangChain Document-like object. Skipping."
                    )

        return result

    def _process_documents(self, content: list) -> tuple[list[str], list[Document]]:
        """
        Processes arbitrary list of data to produce two lists:
        one with unique IDs, and one with LangChain documents.

        Handles duplicate documents.

        Parameters
        ----------
        content : list
            Arbitrary data.

        Returns
        -------
        tuple[list[str], list[Document]]
            Lists with IDs and docs
        """
        docs = self._as_langchain_documents(content)
        if docs:
            # Take only unique ID document. Get two lists, one with ids, one with documents
            # For some documents, not all chars can be encoded properly.
            # In such cases, replace invalid chars by question marks, i.e. setting errors="replace"
            return tuple(
                map(
                    list,
                    zip(*{hashlib.sha256(str(doc).encode(errors="replace")).hexdigest(): doc for doc in docs}.items()),
                )
            )
        else:
            return [], []

    def add_documents(self, content: list, **kwargs: Any) -> list[str]:
        """
        Embed and add documents to the vector store.

        Parameters
        ----------
        content : list
            Documents to be embedded and added to the vector store.

        Returns
        -------
        list[str]
            List of documents IDs.
        """
        max_batch_size = kwargs.get("max_batch_size")
        if max_batch_size is None:
            try:
                max_batch_size = self._vector_store._client.get_max_batch_size()  # type: ignore[attr-defined]
            except AttributeError:
                max_batch_size = 10_000

        ids, docs = self._process_documents(content)
        if len(docs) > max_batch_size:
            batch_ids = []

            for batch_start in range(0, len(docs), max_batch_size):
                batch_ids.extend(
                    self._vector_store.add_documents(
                        docs[batch_start : batch_start + max_batch_size],
                        ids=ids[batch_start : batch_start + max_batch_size],
                        **kwargs,
                    )
                )
            return batch_ids
        else:
            return self._vector_store.add_documents(docs, ids=ids, **kwargs)

    def _get_window_documents(self, doc_id: str, seq_nums_window: list[int]) -> list[Document]:
        """
        Receives a document ID and a list of chunks' sequence_numbers,
        and searches the vector store according to the metadata.

        Parameters
        ----------
        doc_id : str
            ID of document.

        seq_nums_window : list[int]
            Sequence numbers of chunks.

        Returns
        -------
        list[Document]
            Documents from that document with these sequence_numbers.
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
                    self._chunk_sequence_number_field: metadata[self._chunk_sequence_number_field],
                    self._document_name_field: doc_id,
                },
            )
            for text, metadata in zip(texts, metadatas)
        ]
        return window_documents

    def search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        **kwargs: Any,
    ) -> list[Document] | list[tuple[Document, float]]:
        """Searches for documents most similar to the query.

        The method is designed as a wrapper for respective LangChain VectorStores' similarity search methods.
        Therefore, additional search parameters passed in ``kwargs`` should be consistent with those methods,
        and can be found in the LangChain documentation.

        Parameters
        ----------
        query : str
            Query for which grounding documents will be searched for.

        k : int, default=5
            Number of documents to retrieve

        include_scores : bool, default=False
            Whether similarity scores of found documents should be returned.

        Returns
        -------
        list[Document] | list[tuple[Document, float]]
            Found documents with or without scores.
        """
        if include_scores:
            result = self._vector_store.similarity_search_with_score(query, k=k, **kwargs)
        else:
            result = self._vector_store.similarity_search(query, k=k, **kwargs)

        return result

    def window_search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list:
        """
        Searches for documents most similar to the query and extend a document (a chunk)
        to its adjacent chunks (if they exist) from the same origin document.

        The method is designed as a wrapper for respective LangChain VectorStores' similarity search methods.
        Therefore, additional search parameters passed in ``kwargs`` should be consistent with those methods,
        and can be found in the LangChain documentation.

        Parameters
        ----------
        query : str
            Query for which grounding documents will be searched for.

        k : int, default=5
            Number of documents to retrieve

        include_scores : bool, default=False
            Whether similarity scores of found documents should be returned.

        window_size : int, default=2
            Number of chunks from right and left side of the original chunk.

        Returns
        -------
        list
            Found documents with or without scores.
        """
        documents = self.search(query, k, include_scores, **kwargs)
        if window_size <= 0:
            return documents

        if not include_scores:
            documents = cast(list[Document], documents)
            return [self._window_extend_and_merge(document, window_size) for document in documents]
        else:
            documents_and_scores = cast(list[tuple[Document, float]], documents)
            documents = [t[0] for t in documents_and_scores]
            scores = [t[1] for t in documents_and_scores]
            extended_documents = [self._window_extend_and_merge(document, window_size) for document in documents]
            return list(zip(extended_documents, scores))

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """Delete by vector ID or other criteria. Sor more details see LangChain documentation
        https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore
        """
        self._vector_store.delete(ids, **kwargs)

    def _window_extend_and_merge(self, document: Document, window_size: int) -> Document:
        """
        Extends a document (a chunk) to its adjacent chunks (if they exist) from the same origin document.
        Then merges the adjacent chunks into one chunk while keeping their order,
        and merges intersecting text between them (if it exists).
        This requires chunks to have "document_id" and "sequence_number" in their metadata.

        Parameters
        ----------
        document : Document
            Chunk / document to be extended to its window and merged.

        window_size : int
            Number of adjacent chunks to retrieve before and after the center, according to the sequence_number.

        Returns
        -------
        Document
            Chunk / document after extending and merging.
        """
        if "document_id" not in document.metadata:
            raise ValueError('document must have "document_id" in its metadata')
        if "sequence_number" not in document.metadata:
            raise ValueError('document must have "sequence_number" in its metadata')
        doc_id = document.metadata["document_id"]
        seq_num = document.metadata["sequence_number"]
        seq_nums_window = [seq_num + i for i in range(-window_size, window_size + 1, 1)]

        window_documents = self._get_window_documents(doc_id, seq_nums_window)

        window_documents.sort(key=lambda x: x.metadata["sequence_number"])
        return merge_window_into_a_document(window_documents)
