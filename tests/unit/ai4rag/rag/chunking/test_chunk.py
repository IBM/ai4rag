# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk


class TestAI4RAGChunkId:
    """Test suite for AI4RAGChunk.chunk_id deterministic generation."""

    def test_chunk_id_is_deterministic(self):
        chunk = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 1})
        assert chunk.chunk_id == chunk.chunk_id

    def test_identical_chunks_produce_same_id(self):
        chunk_a = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 1})
        chunk_b = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 1})
        assert chunk_a.chunk_id == chunk_b.chunk_id

    def test_different_text_produces_different_id(self):
        chunk_a = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 1})
        chunk_b = AI4RAGChunk(text="world", metadata={"document_id": "doc1", "sequence_number": 1})
        assert chunk_a.chunk_id != chunk_b.chunk_id

    def test_different_document_id_produces_different_id(self):
        chunk_a = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 1})
        chunk_b = AI4RAGChunk(text="hello", metadata={"document_id": "doc2", "sequence_number": 1})
        assert chunk_a.chunk_id != chunk_b.chunk_id

    def test_different_sequence_number_produces_different_id(self):
        chunk_a = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 1})
        chunk_b = AI4RAGChunk(text="hello", metadata={"document_id": "doc1", "sequence_number": 2})
        assert chunk_a.chunk_id != chunk_b.chunk_id

    def test_missing_metadata_keys_default_gracefully(self):
        chunk = AI4RAGChunk(text="hello")
        assert isinstance(chunk.chunk_id, str)
        assert len(chunk.chunk_id) == 64

    def test_chunk_id_is_sha256_hex(self):
        chunk = AI4RAGChunk(text="test", metadata={"document_id": "d", "sequence_number": 0})
        assert len(chunk.chunk_id) == 64
        assert all(c in "0123456789abcdef" for c in chunk.chunk_id)

    @pytest.mark.parametrize(
        "text",
        ["", "a", "hello world", "unicode: äöüß☃", "a" * 10_000],
        ids=["empty", "single-char", "words", "unicode", "large"],
    )
    def test_chunk_id_handles_various_text(self, text):
        chunk = AI4RAGChunk(text=text, metadata={"document_id": "doc1", "sequence_number": 0})
        assert len(chunk.chunk_id) == 64
