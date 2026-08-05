# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for :class:`ChromaVectorStore` on the native ``chromadb`` client.

These tests exercise the store end-to-end against a real in-memory
``EphemeralClient`` rather than mocking the client. ``chromadb`` keeps a single
process-wide in-memory system that is shared across every ``EphemeralClient``
instance, so isolation is achieved by giving each store its own auto-generated
collection name and dropping the collection on teardown (see the ``store``
fixture).
"""

import hashlib
from unittest.mock import MagicMock

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.chroma import ChromaVectorStore
from ai4rag.rag.vector_store.config import ChromaConfig


class MockEmbeddingModel(BaseEmbeddingModel):
    """Deterministic 3-D embedding model for reproducible search ordering.

    The vector is derived from the SHA-256 digest of the text, so identical
    texts embed to identical vectors (cosine distance ``0`` → similarity ``1``)
    while distinct texts get distinct, non-zero vectors. This makes nearest-
    neighbour ordering deterministic without a real embedding backend.
    """

    def __init__(self) -> None:
        super().__init__(client=MagicMock(), model_id="mock-embed", params={"embedding_dimension": 3})

    @staticmethod
    def _vector(text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        # +1.0 keeps every component strictly positive so no vector is the zero
        # vector (for which cosine distance is undefined).
        return [1.0 + digest[0] / 255.0, 1.0 + digest[1] / 255.0, 1.0 + digest[2] / 255.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


@pytest.fixture
def embedding_model() -> MockEmbeddingModel:
    """Provide a fresh deterministic embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def store(embedding_model):
    """Provide an ephemeral store with an isolated, auto-cleaned collection."""
    vector_store = ChromaVectorStore(embedding_model=embedding_model)
    yield vector_store
    # Drop the collection so the shared in-memory system does not leak state
    # between tests.
    try:
        vector_store.clean_collection()
    except Exception:  # pragma: no cover - teardown best effort
        pass


def _doc_chunks() -> list[AI4RAGChunk]:
    """Build five single-word chunks of one document with disjoint vocabularies.

    Disjoint words keep the merged window text predictable (no incidental
    overlap de-duplication) while sequence numbers drive window expansion.
    """
    words = ["alpha", "bravo", "charlie", "delta", "echo"]
    return [
        AI4RAGChunk(text=word, metadata={"document_id": "docA", "sequence_number": i}) for i, word in enumerate(words)
    ]


class TestChromaVectorStoreInitialization:
    """Initialization and collection-name handling."""

    def test_init_with_defaults(self, store):
        assert store.collection_name.startswith("ai4rag_")
        assert store.distance_metric == "cosine"
        assert store.DOCUMENT_NAME_FIELD == "document_id"
        assert store.SEQUENCE_NUMBER_FIELD == "sequence_number"
        assert store.count() == 0

    def test_init_with_custom_parameters(self, embedding_model):
        vector_store = ChromaVectorStore(
            embedding_model=embedding_model,
            collection_name="ai4rag_custom_collection",
            distance_metric="l2",
        )
        try:
            assert vector_store.collection_name == "ai4rag_custom_collection"
            assert vector_store.distance_metric == "l2"
        finally:
            vector_store.clean_collection()

    def test_init_sanitizes_collection_name(self, embedding_model):
        vector_store = ChromaVectorStore(embedding_model=embedding_model, collection_name="ai4rag-collection.v1")
        try:
            assert vector_store.collection_name == "ai4rag_collection_v1"
        finally:
            vector_store.clean_collection()


class TestChromaVectorStoreClientSelection:
    """``_build_client`` selects the client implied by the config."""

    def test_ephemeral_client_by_default(self, mocker):
        ephemeral = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.EphemeralClient")
        persistent = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.PersistentClient")
        http = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.HttpClient")

        client = ChromaVectorStore._build_client(ChromaConfig())

        ephemeral.assert_called_once_with()
        persistent.assert_not_called()
        http.assert_not_called()
        assert client is ephemeral.return_value

    def test_persistent_client_when_persist_directory(self, mocker):
        persistent = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.PersistentClient")
        ephemeral = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.EphemeralClient")

        client = ChromaVectorStore._build_client(ChromaConfig(persist_directory="/data/chroma"))

        persistent.assert_called_once_with(path="/data/chroma")
        ephemeral.assert_not_called()
        assert client is persistent.return_value

    def test_http_client_when_host(self, mocker):
        http = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.HttpClient")
        persistent = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.PersistentClient")

        client = ChromaVectorStore._build_client(ChromaConfig(host="chroma.local", port=9000))

        http.assert_called_once_with(host="chroma.local", port=9000)
        persistent.assert_not_called()
        assert client is http.return_value

    def test_host_takes_precedence_over_persist_directory(self, mocker):
        http = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.HttpClient")
        persistent = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.PersistentClient")

        ChromaVectorStore._build_client(ChromaConfig(host="h", persist_directory="/data"))

        http.assert_called_once()
        persistent.assert_not_called()


class TestChromaVectorStoreDistanceMetric:
    """``distance_metric`` property and validation."""

    def test_distance_metric_getter(self, embedding_model):
        vector_store = ChromaVectorStore(embedding_model=embedding_model, distance_metric="l2")
        try:
            assert vector_store.distance_metric == "l2"
        finally:
            vector_store.clean_collection()

    def test_distance_metric_setter_valid(self, store):
        store.distance_metric = "l2"
        assert store.distance_metric == "l2"

    def test_distance_metric_setter_invalid(self, store):
        with pytest.raises(ValueError) as exc_info:
            store.distance_metric = "invalid_metric"
        assert "Invalid distance metric" in str(exc_info.value)


class TestChromaVectorStoreDistanceToSimilarity:
    """Score mapping keeps the 'higher = more relevant' convention."""

    def test_cosine_mapping(self, store):
        assert store.distance_metric == "cosine"
        assert store._distance_to_similarity(0.0) == pytest.approx(1.0)
        assert store._distance_to_similarity(0.2) == pytest.approx(0.8)
        assert store._distance_to_similarity(2.0) == pytest.approx(-1.0)

    def test_l2_mapping(self, embedding_model):
        vector_store = ChromaVectorStore(embedding_model=embedding_model, distance_metric="l2")
        try:
            assert vector_store._distance_to_similarity(0.0) == pytest.approx(1.0)
            assert vector_store._distance_to_similarity(3.0) == pytest.approx(0.25)
        finally:
            vector_store.clean_collection()


class TestChromaVectorStoreAddDocuments:
    """``add_documents`` embedding, dedup, and batching."""

    def test_add_documents_basic(self, store):
        chunks = [AI4RAGChunk(text="alpha"), AI4RAGChunk(text="bravo")]
        ids = store.add_documents(chunks)
        assert len(ids) == 2
        assert ids == [chunk.chunk_id for chunk in chunks]
        assert store.count() == 2

    def test_add_documents_with_metadata(self, store):
        store.add_documents([AI4RAGChunk(text="alpha", metadata={"source": "a"})])
        results = store.search("alpha", k=1)
        assert results[0].metadata == {"source": "a"}

    def test_add_documents_empty_metadata_roundtrips_as_empty_dict(self, store):
        # Chroma rejects empty-dict metadata; the store maps it to None on write
        # and it comes back as {} on read.
        store.add_documents([AI4RAGChunk(text="lonely")])
        results = store.search("lonely", k=1)
        assert results[0].text == "lonely"
        assert results[0].metadata == {}

    def test_add_documents_deduplicates_by_chunk_id(self, store):
        chunks = [AI4RAGChunk(text="dup"), AI4RAGChunk(text="dup"), AI4RAGChunk(text="unique")]
        ids = store.add_documents(chunks)
        assert len(ids) == 2
        assert store.count() == 2

    def test_add_documents_empty_list(self, store):
        assert store.add_documents([]) == []
        assert store.count() == 0

    def test_add_documents_batches_by_max_batch_size(self, store, mocker):
        spy = mocker.patch.object(store._collection, "upsert", wraps=store._collection.upsert)
        chunks = [AI4RAGChunk(text=f"word{i}") for i in range(5)]
        ids = store.add_documents(chunks, max_batch_size=2)
        # 5 chunks in batches of 2 -> 3 upsert calls.
        assert spy.call_count == 3
        assert len(ids) == 5
        assert store.count() == 5


class TestChromaVectorStoreSearch:
    """``search`` returns chunks and 'higher = better' scores."""

    def test_search_basic(self, store):
        store.add_documents([AI4RAGChunk(text="alpha"), AI4RAGChunk(text="bravo")])
        results = store.search("alpha", k=5)
        assert all(isinstance(chunk, AI4RAGChunk) for chunk in results)
        assert results[0].text == "alpha"

    def test_search_respects_k(self, store):
        store.add_documents([AI4RAGChunk(text=w) for w in ("alpha", "bravo", "charlie")])
        assert len(store.search("alpha", k=2)) == 2

    def test_search_with_scores_higher_is_more_relevant(self, store):
        store.add_documents([AI4RAGChunk(text="alpha"), AI4RAGChunk(text="zulu different")])
        results = store.search("alpha", k=2, include_scores=True)
        assert isinstance(results[0], tuple)
        chunk, score = results[0]
        # Exact match ranks first with the maximal cosine similarity (~1.0)...
        assert chunk.text == "alpha"
        assert score == pytest.approx(1.0, abs=1e-4)
        # ...and outranks the dissimilar document.
        assert results[0][1] > results[1][1]

    def test_search_with_where_filter(self, store):
        store.add_documents(
            [
                AI4RAGChunk(text="cat doc", metadata={"category": "animal"}),
                AI4RAGChunk(text="car doc", metadata={"category": "vehicle"}),
            ]
        )
        results = store.search("cat doc", k=5, where={"category": {"$eq": "animal"}})
        assert len(results) == 1
        assert results[0].metadata["category"] == "animal"

    def test_search_with_filter_alias(self, store):
        store.add_documents(
            [
                AI4RAGChunk(text="cat doc", metadata={"category": "animal"}),
                AI4RAGChunk(text="car doc", metadata={"category": "vehicle"}),
            ]
        )
        results = store.search("car doc", k=5, filter={"category": {"$eq": "vehicle"}})
        assert len(results) == 1
        assert results[0].metadata["category"] == "vehicle"

    def test_search_empty_collection(self, store):
        assert store.search("anything", k=5) == []


class TestChromaVectorStoreWindowSearch:
    """``window_search`` expands each hit with its neighbouring chunks."""

    def test_zero_window_size_returns_search_results(self, store):
        store.add_documents(_doc_chunks())
        results = store.window_search("charlie", k=1, window_size=0)
        assert len(results) == 1
        assert results[0].text == "charlie"

    def test_negative_window_size_returns_search_results(self, store):
        store.add_documents(_doc_chunks())
        results = store.window_search("charlie", k=1, window_size=-1)
        assert results[0].text == "charlie"

    def test_window_search_without_scores_merges_neighbours(self, store):
        store.add_documents(_doc_chunks())
        results = store.window_search("charlie", k=1, window_size=1)
        assert isinstance(results[0], AI4RAGChunk)
        # seq 1 (bravo), 2 (charlie), 3 (delta) merged in sequence order.
        assert results[0].text == "bravo charlie delta"

    def test_window_search_with_scores_merges_and_keeps_score(self, store):
        store.add_documents(_doc_chunks())
        results = store.window_search("charlie", k=1, window_size=1, include_scores=True)
        chunk, score = results[0]
        assert isinstance(chunk, AI4RAGChunk)
        assert chunk.text == "bravo charlie delta"
        assert score == pytest.approx(1.0, abs=1e-4)

    def test_window_clamped_at_document_edges(self, store):
        store.add_documents(_doc_chunks())
        # Centered on the first chunk: only seq 0 and 1 exist to the right.
        results = store.window_search("alpha", k=1, window_size=1)
        assert results[0].text == "alpha bravo"


class TestChromaVectorStoreWindowExtendAndMerge:
    """``_window_extend_and_merge`` validation and merging."""

    def test_missing_document_id_raises(self, store):
        chunk = AI4RAGChunk(text="x", metadata={"sequence_number": 1})
        with pytest.raises(ValueError, match="document_id"):
            store._window_extend_and_merge(chunk, window_size=2)

    def test_missing_sequence_number_raises(self, store):
        chunk = AI4RAGChunk(text="x", metadata={"document_id": "docA"})
        with pytest.raises(ValueError, match="sequence_number"):
            store._window_extend_and_merge(chunk, window_size=2)

    def test_basic_merge(self, store):
        store.add_documents(_doc_chunks())
        center = AI4RAGChunk(text="charlie", metadata={"document_id": "docA", "sequence_number": 2})
        merged = store._window_extend_and_merge(center, window_size=1)
        assert isinstance(merged, AI4RAGChunk)
        assert merged.text == "bravo charlie delta"


class TestChromaVectorStoreGetWindowDocuments:
    """``_get_window_documents`` fetches a contiguous slice of one document."""

    def test_returns_chunks_within_range(self, store):
        store.add_documents(_doc_chunks())
        results = store._get_window_documents("docA", [1, 2, 3])
        assert all(isinstance(chunk, AI4RAGChunk) for chunk in results)
        assert sorted(chunk.metadata["sequence_number"] for chunk in results) == [1, 2, 3]

    def test_filters_by_document_id(self, store):
        chunks = _doc_chunks()
        chunks.append(AI4RAGChunk(text="foreign", metadata={"document_id": "docB", "sequence_number": 2}))
        store.add_documents(chunks)
        results = store._get_window_documents("docA", [0, 1, 2])
        texts = {chunk.text for chunk in results}
        assert "foreign" not in texts
        assert all(chunk.metadata["document_id"] == "docA" for chunk in results)


class TestChromaVectorStoreLifecycle:
    """``count``, ``clear``, ``delete``, and ``clean_collection``."""

    def test_count_reflects_contents(self, store):
        assert store.count() == 0
        store.add_documents([AI4RAGChunk(text="alpha"), AI4RAGChunk(text="bravo")])
        assert store.count() == 2

    def test_clear_removes_all_entries(self, store):
        store.add_documents([AI4RAGChunk(text="alpha"), AI4RAGChunk(text="bravo")])
        store.clear()
        assert store.count() == 0

    def test_clear_on_empty_is_noop(self, store):
        store.clear()
        assert store.count() == 0

    def test_delete_by_ids(self, store):
        chunks = [AI4RAGChunk(text="alpha"), AI4RAGChunk(text="bravo")]
        ids = store.add_documents(chunks)
        store.delete([ids[0]])
        assert store.count() == 1
        remaining = store.search("bravo", k=5)
        assert remaining[0].text == "bravo"

    def test_clean_collection_drops_collection(self, embedding_model):
        vector_store = ChromaVectorStore(embedding_model=embedding_model)
        vector_store.add_documents([AI4RAGChunk(text="alpha")])
        name = vector_store.collection_name
        vector_store.clean_collection()
        existing = {collection.name for collection in vector_store._client.list_collections()}
        assert name not in existing


class TestChromaVectorStoreClose:
    """``close()`` releases OS resources without destroying shared ephemeral data."""

    def test_ephemeral_data_survives_close_and_reopen(self, embedding_model):
        """Regression: closing an ephemeral store must not wipe its shared in-memory data.

        chromadb backs every ``EphemeralClient`` with one process-wide,
        reference-counted in-memory ``System``. A ``close()`` that decremented that
        count to zero would stop the ``System`` and discard all collections — wiping
        data a later store still depends on, exactly as happens across HPO trials
        that reuse a collection by name (see ``ChromaVectorStore.close``). This
        asserts a closed-then-reopened ephemeral store still sees its documents.
        """
        collection_name = "ai4rag_reuse_regression"
        with ChromaVectorStore(embedding_model=embedding_model, collection_name=collection_name) as store:
            store.add_documents([AI4RAGChunk(text="alpha"), AI4RAGChunk(text="bravo")])
            assert store.count() == 2

        # A fresh store over the same collection — as a subsequent HPO trial would
        # open — must still find the data the first (now-closed) store wrote.
        reopened = ChromaVectorStore(embedding_model=embedding_model, collection_name=collection_name)
        try:
            assert reopened.count() == 2
            assert reopened.search("alpha", k=1)[0].text == "alpha"
        finally:
            reopened.clean_collection()

    def test_close_is_noop_for_ephemeral_client(self, embedding_model, mocker):
        """An ephemeral client holds no OS resource; close() must not tear it down.

        Calling the underlying ``client.close()`` would decrement the shared
        ``System``'s refcount and risk destroying in-memory data other stores use,
        so the store must skip it entirely for the ephemeral mode.
        """
        store = ChromaVectorStore(embedding_model=embedding_model, collection_name="ai4rag_noop_close")
        spy = mocker.spy(store._client, "close")
        try:
            store.close()
            spy.assert_not_called()
        finally:
            store.clean_collection()

    def test_close_releases_persistent_client(self, embedding_model, mocker):
        """A persistent client holds a SQLite file lock; close() must release it."""
        persistent = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.PersistentClient")
        store = ChromaVectorStore(
            embedding_model=embedding_model,
            config=ChromaConfig(persist_directory="/tmp/ai4rag-chroma-persist"),
            collection_name="ai4rag_persist_close",
        )

        store.close()

        persistent.return_value.close.assert_called_once()

    def test_close_releases_http_client(self, embedding_model, mocker):
        """An HTTP client holds client-side sockets; close() must release them."""
        http = mocker.patch("ai4rag.rag.vector_store.chroma.chromadb.HttpClient")
        store = ChromaVectorStore(
            embedding_model=embedding_model,
            config=ChromaConfig(host="chroma.local", port=9000),
            collection_name="ai4rag_http_close",
        )

        store.close()

        http.return_value.close.assert_called_once()
