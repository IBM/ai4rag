# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI

from ai4rag.rag.chunking.chunk import AI4RAGChunk

from ..embedding.base_model import BaseEmbeddingModel
from .base_vector_store import BaseVectorStore
from .config import ChromaConfig
from .utils import merge_window_into_a_document

__all__ = ["ChromaVectorStore"]


class ChromaVectorStore(BaseVectorStore):
    """Vector store backed by ChromaDB via the native ``chromadb`` client.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    config : ChromaConfig | None, default=None
        Connection parameters selecting the Chroma running mode (ephemeral,
        persistent, or client/server). Defaults to an ephemeral in-memory
        instance.
    distance_metric : str, default="cosine"
        Metric used to calculate similarity between vectors. One of
        ``"cosine"`` or ``"l2"``.
    collection_name : str | None, default=None
        Existing collection to reuse; must start with the ``ai4rag`` prefix. When
        omitted, a new compliant name is generated (see
        :func:`ai4rag.rag.vector_store.utils.resolve_collection_name`).
    """

    _supported_distance_metrics = ("cosine", "l2")
    _BATCH_SIZE = 2048
    DOCUMENT_NAME_FIELD = "document_id"
    SEQUENCE_NUMBER_FIELD = "sequence_number"

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: ChromaConfig | None = None,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
    ) -> None:
        """Initialize the store and open (or create) the backing collection.

        Parameters
        ----------
        embedding_model : BaseEmbeddingModel
            Model used to embed documents and queries.
        config : ChromaConfig | None, default=None
            Connection parameters selecting the Chroma running mode (ephemeral,
            persistent, or client/server). Defaults to an ephemeral in-memory
            instance.
        distance_metric : str, default="cosine"
            Metric used to measure similarity between vectors. One of
            ``"cosine"`` or ``"l2"``.
        collection_name : str | None, default=None
            Existing collection to reuse; must start with the ``ai4rag`` prefix.
            When omitted, a new compliant name is generated.
        """
        # Resolve the config once so both the base class and the client builder
        # see the same instance; the default ephemeral config must not leak
        # ``None`` into ``_build_client``.
        config = config or ChromaConfig()
        super().__init__(embedding_model, config, distance_metric, collection_name)

        # Ephemeral mode (neither host nor persist_directory) is backed by a
        # single process-wide in-memory chromadb ``System`` shared across every
        # ``EphemeralClient`` (keyed by the constant "ephemeral" identifier).
        # This flag lets ``close()`` skip tearing that System down — see there
        # for why closing an ephemeral client is actively harmful.
        self._is_ephemeral = not config.host and not config.persist_directory
        self._client = self._build_client(config)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    @staticmethod
    def _build_client(config: ChromaConfig) -> ClientAPI:
        """Create a Chroma client for the mode implied by *config*.

        ``host`` selects a remote client/server connection and takes precedence;
        otherwise ``persist_directory`` selects an on-disk persistent client;
        with neither set an ephemeral in-memory client is used.

        Parameters
        ----------
        config : ChromaConfig
            Connection parameters selecting the Chroma running mode.

        Returns
        -------
        ClientAPI
            A configured Chroma client for the selected running mode.
        """
        if config.host:
            return chromadb.HttpClient(host=config.host, port=config.port)
        if config.persist_directory:
            return chromadb.PersistentClient(path=config.persist_directory)
        return chromadb.EphemeralClient()

    @property
    def distance_metric(self) -> str:
        """Distance metric currently used for similarity search.

        Returns
        -------
        str
            The active distance metric (``"cosine"`` or ``"l2"``).
        """
        return self._distance_metric

    @distance_metric.setter
    def distance_metric(self, value: str) -> None:
        """Set the distance metric used for similarity search.

        Parameters
        ----------
        value : str
            Distance metric to use. One of ``"cosine"`` or ``"l2"``.

        Raises
        ------
        ValueError
            If the distance metric is not supported.
        """
        if value not in self._supported_distance_metrics:
            raise ValueError(f"Invalid distance metric: {value}. Use one of: {self._supported_distance_metrics}.")
        self._distance_metric = value

    def _distance_to_similarity(self, distance: float) -> float:
        """Convert a Chroma distance into a "higher = more relevant" score.

        Keeps ``include_scores`` semantics consistent with the Milvus and
        PGVector stores. For ``cosine`` the returned value is the true cosine
        similarity (``1 - distance``); for ``l2`` a monotonically decreasing
        ``1 / (1 + distance)`` maps the unbounded distance into ``(0, 1]``.

        Parameters
        ----------
        distance : float
            Raw distance returned by Chroma for a matched vector.

        Returns
        -------
        float
            Similarity score where larger values indicate greater relevance.
        """
        if self._distance_metric == "cosine":
            return 1.0 - distance
        return 1.0 / (1.0 + distance)

    def clear(self) -> None:
        """Delete all entries while keeping the collection in place."""
        all_ids = self._collection.get()["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)

    def count(self) -> int:
        """Count the number of entries in the collection.

        Returns
        -------
        int
            Number of stored chunks.
        """
        return self._collection.count()

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs: Any) -> list[str]:
        """Embed, deduplicate, and upsert chunks into the collection.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and stored.
        **kwargs : Any
            Optional overrides. ``max_batch_size`` (int) sets the upsert batch
            size (default :attr:`_BATCH_SIZE`).

        Returns
        -------
        list[str]
            IDs of the stored chunks (deduplicated by ``chunk_id``).
        """
        if not documents:
            return []

        # Deduplicate by chunk_id, keeping first-seen order and last-seen content.
        unique_chunks: dict[str, AI4RAGChunk] = {}
        for chunk in documents:
            unique_chunks[chunk.chunk_id] = chunk

        ids = list(unique_chunks.keys())
        chunks = list(unique_chunks.values())
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_documents(texts)
        # Chroma rejects empty-dict metadata; represent "no metadata" as None.
        metadatas = [chunk.metadata if chunk.metadata else None for chunk in chunks]

        batch_size = kwargs.get("max_batch_size", self._BATCH_SIZE)
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=metadatas[start:end],  # type: ignore[arg-type]
            )
        return ids

    def search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        **kwargs: Any,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Search for chunks most similar to *query*.

        Chroma supports pure vector search only; hybrid-search keyword arguments
        (``search_mode``, ``ranker_*``) forwarded by the retriever are ignored.
        A metadata filter may be supplied via ``where`` (or ``filter``).

        Parameters
        ----------
        query : str
            Query for which grounding documents will be searched for.
        k : int, default=5
            Number of documents to retrieve.
        include_scores : bool, default=False
            Whether to return similarity scores. Scores follow the
            "higher = more relevant" convention shared across ai4rag stores.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Found chunks with or without scores.
        """
        where = kwargs.get("where") or kwargs.get("filter")
        embedding = self.embedding_model.embed_query(query)
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = result["documents"][0] if result["documents"] else []
        metadatas = result["metadatas"][0] if result["metadatas"] else []
        distances = result["distances"][0] if result["distances"] else []

        chunks = [
            AI4RAGChunk(text=text, metadata=dict(metadata) if metadata else {})
            for text, metadata in zip(documents, metadatas)
        ]
        if include_scores:
            return [(chunk, self._distance_to_similarity(distance)) for chunk, distance in zip(chunks, distances)]
        return chunks

    def window_search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Search for similar chunks and expand each with its neighbouring chunks.

        Each matched chunk is extended with up to ``window_size`` adjacent chunks
        on either side from the same source document, then merged into a single
        chunk (overlapping text is de-duplicated).

        Parameters
        ----------
        query : str
            Query for which grounding documents will be searched for.
        k : int, default=5
            Number of documents to retrieve.
        include_scores : bool, default=False
            Whether similarity scores of found documents should be returned.
        window_size : int, default=2
            Number of chunks from the right and left side of the original chunk.

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
        """Delete stored chunks by ID.

        Parameters
        ----------
        ids : list[str]
            IDs of the chunks to delete.
        """
        self._collection.delete(ids=ids, **kwargs)

    def clean_collection(self) -> None:
        """Drop the underlying Chroma collection."""
        self._client.delete_collection(self._collection_name)

    def close(self) -> None:
        """Release the underlying Chroma client's operating-system resources.

        No-op for an ephemeral client. Its data lives in a process-wide, in-memory
        ``System`` that chromadb shares across every ``EphemeralClient`` and
        reference-counts; ``client.close()`` decrements that count and, once it
        reaches zero, stops the ``System`` and discards all in-memory data —
        destroying collections a later store (e.g. a subsequent HPO trial reusing
        the same ``collection_name``) still depends on. An ephemeral client holds
        no OS resource to release, so skipping the close leaks nothing durable:
        the shared ``System`` is reclaimed when the interpreter exits.

        For a persistent client this releases the SQLite file lock (so the store
        can be reopened); for an HTTP client it releases the client-side sockets.
        In both cases the actual data survives on disk / on the server, so reuse
        across store instances is unaffected.
        """
        if self._is_ephemeral:
            return
        self._client.close()

    def _get_window_documents(self, doc_id: str, seq_nums_window: list[int]) -> list[AI4RAGChunk]:
        """Fetch chunks of a document within a contiguous sequence-number range.

        Parameters
        ----------
        doc_id : str
            ID of the source document.
        seq_nums_window : list[int]
            Ordered sequence numbers bounding the window (first and last used).

        Returns
        -------
        list[AI4RAGChunk]
            Chunks of ``doc_id`` whose sequence number falls within the window.
        """
        expr = {
            "$and": [
                {self.DOCUMENT_NAME_FIELD: {"$eq": doc_id}},
                {self.SEQUENCE_NUMBER_FIELD: {"$gte": seq_nums_window[0]}},
                {self.SEQUENCE_NUMBER_FIELD: {"$lte": seq_nums_window[-1]}},
            ]
        }
        result = self._collection.get(where=expr, include=["documents", "metadatas"])  # type: ignore[arg-type]
        texts, metadatas = result["documents"] or [], result["metadatas"] or []
        return [
            AI4RAGChunk(text=text, metadata=dict(metadata) if metadata else {})
            for text, metadata in zip(texts, metadatas)
        ]

    def _window_extend_and_merge(self, chunk: AI4RAGChunk, window_size: int) -> AI4RAGChunk:
        """Extend a chunk with its neighbours and merge them into one chunk.

        Retrieves the adjacent chunks (if any) from the same source document,
        orders them by sequence number, and merges them while de-duplicating any
        overlapping text.

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

        Raises
        ------
        ValueError
            If the chunk metadata lacks ``document_id`` or ``sequence_number``.
        """
        if self.DOCUMENT_NAME_FIELD not in chunk.metadata:
            raise ValueError(f'chunk must have "{self.DOCUMENT_NAME_FIELD}" in its metadata')
        if self.SEQUENCE_NUMBER_FIELD not in chunk.metadata:
            raise ValueError(f'chunk must have "{self.SEQUENCE_NUMBER_FIELD}" in its metadata')
        doc_id = chunk.metadata[self.DOCUMENT_NAME_FIELD]
        seq_num = chunk.metadata[self.SEQUENCE_NUMBER_FIELD]
        seq_nums_window = [seq_num + i for i in range(-window_size, window_size + 1, 1)]

        window_chunks = self._get_window_documents(doc_id, seq_nums_window)
        window_chunks.sort(key=lambda c: c.metadata[self.SEQUENCE_NUMBER_FIELD])

        return merge_window_into_a_document(window_chunks)
