# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Unit tests for the Milvus store running against a real embedded Milvus Lite.

These tests exercise :class:`MilvusVectorStore` end-to-end against a real, local
Milvus Lite database (a temporary ``.db`` file) rather than mocking the client.
Milvus Lite ships with ``pymilvus[milvus-lite]`` and needs no server, so this
suite runs in the standard unit tier and is the primary guard that the store's
BM25 hybrid schema is actually supported by the installed Milvus Lite version.

Isolation is achieved by giving each store its own temporary database directory
(and an auto-generated collection name), removed on teardown.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.config import MilvusLiteConfig
from ai4rag.rag.vector_store.local_store import temporary_milvus_lite_store
from ai4rag.rag.vector_store.milvus import MilvusVectorStore

_EMBEDDING_DIMENSION = 16


class DeterministicEmbeddingModel(BaseEmbeddingModel):
    """Hash-based embedding model: identical text always yields the same vector.

    Each text maps to a fixed-length vector derived from its SHA-256 digest, so a
    query embedded from the exact text of a stored chunk lands on that chunk
    (cosine distance ~0) without contacting any embedding service. Components stay
    strictly positive to avoid a zero vector.
    """

    def __init__(self) -> None:
        super().__init__(client=None, model_id="deterministic", params={"embedding_dimension": _EMBEDDING_DIMENSION})

    @staticmethod
    def _vector(text: str) -> list[float]:
        out: list[float] = []
        counter = 0
        while len(out) < _EMBEDDING_DIMENSION:
            digest = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            out.extend(1.0 + byte / 255.0 for byte in digest)
            counter += 1
        return out[:_EMBEDDING_DIMENSION]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


@pytest.fixture(scope="module")
def embedding_model() -> DeterministicEmbeddingModel:
    return DeterministicEmbeddingModel()


@pytest.fixture
def sample_chunks() -> list[AI4RAGChunk]:
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Milvus Lite is an embedded vector database.",
        "Retrieval augmented generation improves grounded answers.",
        "A fox is a small wild animal related to dogs.",
    ]
    return [
        AI4RAGChunk(text=text, metadata={"document_id": "doc", "sequence_number": i}) for i, text in enumerate(texts)
    ]


@pytest.fixture
def store(embedding_model):
    """A Milvus Lite store on a private temporary database, dropped on teardown."""
    tmp_dir = tempfile.mkdtemp(prefix="ai4rag-milvus-lite-test-")
    vector_store = MilvusVectorStore(
        embedding_model=embedding_model, config=MilvusLiteConfig(db_path=str(Path(tmp_dir) / "s.db"))
    )
    try:
        yield vector_store
    finally:
        vector_store.clean_collection()
        vector_store.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


class TestMilvusLiteCollection:
    def test_collection_is_created_with_bm25_schema(self, store):
        """Constructing the store creates its collection, including the BM25 sparse field."""
        assert store._client.has_collection(store.collection_name)
        # The BM25 hybrid schema requires a sparse field; its presence proves
        # Milvus Lite accepted the BM25 Function-backed schema.
        field_names = {field["name"] for field in store._client.describe_collection(store.collection_name)["fields"]}
        assert {"chunk_id", "content", "vector", "metadata", "sparse"}.issubset(field_names)


class TestMilvusLiteAddAndSearch:
    def test_add_documents_then_vector_search(self, store, sample_chunks):
        store.add_documents(sample_chunks)

        results = store.search(sample_chunks[1].text, k=1)

        assert len(results) == 1
        assert results[0].text == sample_chunks[1].text

    def test_search_include_scores_returns_float_pairs(self, store, sample_chunks):
        store.add_documents(sample_chunks)

        results = store.search(sample_chunks[0].text, k=2, include_scores=True)

        assert len(results) == 2
        for chunk, score in results:
            assert isinstance(chunk, AI4RAGChunk)
            assert isinstance(score, float)

    def test_add_documents_deduplicates_by_chunk_id(self, store, sample_chunks):
        # Re-adding the same chunks (identical chunk_id) must not create duplicates.
        store.add_documents(sample_chunks)
        store.add_documents(sample_chunks)

        results = store.search(sample_chunks[0].text, k=10)
        assert len(results) == len(sample_chunks)

    def test_add_empty_documents_is_noop(self, store):
        store.add_documents([])
        assert store.search("anything", k=5) == []


class TestMilvusLiteHybridSearch:
    """Milvus Lite must support dense + BM25 hybrid search with server-side fusion."""

    def test_hybrid_search_rrf(self, store, sample_chunks):
        store.add_documents(sample_chunks)

        results = store.search("fox", k=2, search_mode="hybrid", ranker_strategy="rrf")

        assert results
        assert any("fox" in chunk.text for chunk in results)

    def test_hybrid_search_weighted(self, store, sample_chunks):
        store.add_documents(sample_chunks)

        results = store.search(
            "vector database", k=2, search_mode="hybrid", ranker_strategy="weighted", ranker_alpha=0.5
        )

        assert results


class TestMilvusLiteLifecycle:
    def test_clean_collection_drops_collection(self, embedding_model):
        tmp_dir = tempfile.mkdtemp(prefix="ai4rag-milvus-lite-test-")
        try:
            vector_store = MilvusVectorStore(
                embedding_model=embedding_model, config=MilvusLiteConfig(db_path=str(Path(tmp_dir) / "s.db"))
            )
            name = vector_store.collection_name
            assert vector_store._client.has_collection(name)

            vector_store.clean_collection()

            assert not vector_store._client.has_collection(name)
            vector_store.close()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_temporary_milvus_lite_store_indexes_and_cleans_up(self, embedding_model, sample_chunks):
        captured_dir = {}

        with temporary_milvus_lite_store(embedding_model) as vector_store:
            vector_store.add_documents(sample_chunks)
            results = vector_store.search(sample_chunks[0].text, k=1)
            assert results
            assert results[0].text == sample_chunks[0].text
            # Record the on-disk database directory so we can assert it is removed.
            captured_dir["path"] = Path(vector_store._config.db_path).parent

        assert not captured_dir["path"].exists(), "temporary Milvus Lite directory must be removed on exit"
