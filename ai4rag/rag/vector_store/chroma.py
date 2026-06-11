#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2025-2026.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
import hashlib
from datetime import datetime
from typing import Any, cast

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk

from ..embedding.base_model import BaseEmbeddingModel
from .base_vector_store import BaseVectorStore
from .utils import merge_window_into_a_document


class ChromaVectorStore(BaseVectorStore):
    """
    Class representing single index in the chroma vector database.

    Internally converts between ``AI4RAGChunk`` (pipeline type) and
    langchain ``Document`` (required by ``langchain_chroma.Chroma``).

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Instance used for embedding documents and user's queries.

    reuse_collection_name : str, default=None
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
        embedding_model: BaseEmbeddingModel,
        reuse_collection_name: str | None = None,
        distance_metric: str = "cosine",
        document_name_field: str = "document_id",
        chunk_sequence_number_field: str = "sequence_number",
        **kwargs,
    ) -> None:
        super().__init__(
            embedding_model=embedding_model,
            distance_metric=distance_metric,
            reuse_collection_name=reuse_collection_name,
        )
        self._document_name_field = document_name_field
        self._chunk_sequence_number_field = chunk_sequence_number_field
        self._collection_name = reuse_collection_name or kwargs.pop(
            "collection_name", f"ai4rag_{datetime.now().strftime("%Y%m%d%H%M%S")}"
        )
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
        """Get used distance metric."""
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

    @property
    def collection_name(self) -> str:
        """Dynamically get collection name."""
        return self._collection_name

    def clear(self) -> None:
        """Clear the vector store."""
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
    def _to_langchain_documents(chunks: list[AI4RAGChunk]) -> list[Document]:
        """Convert ``AI4RAGChunk`` objects to langchain ``Document`` for Chroma storage.

        Parameters
        ----------
        chunks : list[AI4RAGChunk]
            Pipeline chunks to convert.

        Returns
        -------
        list[Document]
            Langchain documents suitable for ``langchain_chroma.Chroma``.
        """
        return [Document(page_content=chunk.text, metadata=chunk.metadata) for chunk in chunks]

    @staticmethod
    def _from_langchain_document(doc: Document) -> AI4RAGChunk:
        """Convert a single langchain ``Document`` back to ``AI4RAGChunk``."""
        return AI4RAGChunk(text=doc.page_content, metadata=doc.metadata)

    def _process_documents(self, chunks: list[AI4RAGChunk]) -> tuple[list[str], list[Document]]:
        """
        Convert chunks to langchain documents and deduplicate by content hash.

        Parameters
        ----------
        chunks : list[AI4RAGChunk]
            Pipeline chunks.

        Returns
        -------
        tuple[list[str], list[Document]]
            Lists with unique IDs and deduplicated langchain documents.
        """
        docs = self._to_langchain_documents(chunks)
        if docs:
            return tuple(
                map(
                    list,
                    zip(*{hashlib.sha256(str(doc).encode(errors="replace")).hexdigest(): doc for doc in docs}.items()),
                )
            )
        return [], []

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs: Any) -> list[str]:
        """
        Embed and add chunks to the vector store.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and added to the vector store.

        Returns
        -------
        list[str]
            List of document IDs.
        """
        max_batch_size = kwargs.get("max_batch_size", 2048)

        ids, docs = self._process_documents(documents)
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
        res = self._vector_store.get(where=expr)  # type: ignore[arg-type]
        texts, metadatas = res["documents"], res["metadatas"]
        window_documents = [
            Document(
                page_content=text,
                metadata=metadata,
            )
            for text, metadata in zip(texts, metadatas)
        ]
        return window_documents

    _HYBRID_KWARGS = frozenset({"search_mode", "ranker_strategy", "ranker_k", "ranker_alpha"})

    def search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        **kwargs: Any,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
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
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Found chunks with or without scores.
        """
        filtered_kwargs = {k_: v for k_, v in kwargs.items() if k_ not in self._HYBRID_KWARGS}
        if include_scores:
            lc_results = self._vector_store.similarity_search_with_score(query, k=k, **filtered_kwargs)
            return [(self._from_langchain_document(doc), score) for doc, score in lc_results]

        lc_results = self._vector_store.similarity_search(query, k=k, **filtered_kwargs)
        return [self._from_langchain_document(doc) for doc in lc_results]

    def window_search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
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
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Found chunks with or without scores.
        """
        results = self.search(query, k, include_scores, **kwargs)
        if window_size <= 0:
            return results

        if not include_scores:
            chunks = cast(list[AI4RAGChunk], results)
            return [self._window_extend_and_merge(chunk, window_size) for chunk in chunks]

        chunks_and_scores = cast(list[tuple[AI4RAGChunk, float]], results)
        chunks = [t[0] for t in chunks_and_scores]
        scores = [t[1] for t in chunks_and_scores]
        extended = [self._window_extend_and_merge(chunk, window_size) for chunk in chunks]
        return list(zip(extended, scores))

    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """Delete by vector ID or other criteria. For more details see LangChain documentation
        https://python.langchain.com/api_reference/core/vectorstores/langchain_core.vectorstores.base.VectorStore.html#langchain_core.vectorstores.base.VectorStore
        """
        self._vector_store.delete(ids, **kwargs)

    def _window_extend_and_merge(self, chunk: AI4RAGChunk, window_size: int) -> AI4RAGChunk:
        """
        Extends a chunk to its adjacent chunks (if they exist) from the same origin document.
        Then merges the adjacent chunks into one while keeping their order,
        and merges intersecting text between them (if it exists).

        Parameters
        ----------
        chunk : AI4RAGChunk
            Chunk to be extended to its window and merged.

        window_size : int
            Number of adjacent chunks to retrieve before and after the center.

        Returns
        -------
        AI4RAGChunk
            Chunk after extending and merging.
        """
        if "document_id" not in chunk.metadata:
            raise ValueError('chunk must have "document_id" in its metadata')
        if "sequence_number" not in chunk.metadata:
            raise ValueError('chunk must have "sequence_number" in its metadata')
        doc_id = chunk.metadata["document_id"]
        seq_num = chunk.metadata["sequence_number"]
        seq_nums_window = [seq_num + i for i in range(-window_size, window_size + 1, 1)]

        window_documents = self._get_window_documents(doc_id, seq_nums_window)
        window_documents.sort(key=lambda x: x.metadata["sequence_number"])

        merged_lc_doc = merge_window_into_a_document(window_documents)
        return self._from_langchain_document(merged_lc_doc)
