# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

import pytest
from langchain_core.documents import Document

from ai4rag.rag.chunking.base_chunker import BaseChunker


class ConcreteChunker(BaseChunker[str]):
    """Concrete implementation of BaseChunker for testing purposes."""

    def split_documents(self, documents: list[str]) -> list[str]:
        """Simple split implementation for testing."""
        return [doc[:5] for doc in documents if doc]

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary representation."""
        return {"type": "concrete", "param": "value"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConcreteChunker":
        """Create instance from dictionary."""
        return cls()


class TestBaseChunker:
    """Test suite for BaseChunker abstract base class."""

    def test_base_chunker_cannot_be_instantiated(self):
        """Test that BaseChunker cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_base_chunker_initialization_via_subclass(self):
        """Test that BaseChunker can be instantiated via a concrete subclass."""
        chunker = ConcreteChunker()
        assert isinstance(chunker, BaseChunker)

    def test_split_documents_is_abstract(self):
        """Test that split_documents is abstract and must be implemented."""

        # Create a subclass without implementing split_documents
        class IncompleteChunker(BaseChunker[str]):
            def to_dict(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> "IncompleteChunker":
                return cls()

        with pytest.raises(TypeError) as exc_info:
            IncompleteChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_to_dict_is_abstract(self):
        """Test that to_dict is abstract and must be implemented."""

        # Create a subclass without implementing to_dict
        class IncompleteChunker(BaseChunker[str]):
            def split_documents(self, documents: list[str]) -> list[str]:
                return documents

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> "IncompleteChunker":
                return cls()

        with pytest.raises(TypeError) as exc_info:
            IncompleteChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_from_dict_is_abstract(self):
        """Test that from_dict is abstract and must be implemented."""

        # Create a subclass without implementing from_dict
        class IncompleteChunker(BaseChunker[str]):
            def split_documents(self, documents: list[str]) -> list[str]:
                return documents

            def to_dict(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as exc_info:
            IncompleteChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_concrete_chunker_split_documents(self):
        """Test that concrete implementation's split_documents works."""
        chunker = ConcreteChunker()
        documents = ["hello world", "test document", "another one"]
        result = chunker.split_documents(documents)
        assert result == ["hello", "test ", "anoth"]

    def test_concrete_chunker_to_dict(self):
        """Test that concrete implementation's to_dict works."""
        chunker = ConcreteChunker()
        result = chunker.to_dict()
        assert isinstance(result, dict)
        assert result["type"] == "concrete"

    def test_concrete_chunker_from_dict(self):
        """Test that concrete implementation's from_dict works."""
        chunker = ConcreteChunker.from_dict({"type": "concrete", "param": "value"})
        assert isinstance(chunker, ConcreteChunker)
        assert isinstance(chunker, BaseChunker)

    def test_base_chunker_is_generic(self):
        """Test that BaseChunker supports generic types."""
        # Test with string type
        chunker_str = ConcreteChunker()
        assert isinstance(chunker_str, BaseChunker)

        # Test that we can create different type-specific chunkers
        class IntChunker(BaseChunker[int]):
            def split_documents(self, documents: list[int]) -> list[int]:
                return [x // 2 for x in documents]

            def to_dict(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> "IntChunker":
                return cls()

        chunker_int = IntChunker()
        assert isinstance(chunker_int, BaseChunker)
        result = chunker_int.split_documents([10, 20, 30])
        assert result == [5, 10, 15]


class TestBaseChunkerStaticMethods:
    """Test suite for static methods on BaseChunker."""

    def test_set_document_id_in_metadata_if_missing(self):
        documents = [
            Document(page_content="Content 1", metadata={}),
            Document(page_content="Content 2", metadata={}),
        ]
        BaseChunker._set_document_id_in_metadata_if_missing(documents)
        for doc in documents:
            assert "document_id" in doc.metadata
            assert isinstance(doc.metadata["document_id"], str)

    def test_set_document_id_preserves_existing(self):
        documents = [Document(page_content="Content", metadata={"document_id": "existing"})]
        BaseChunker._set_document_id_in_metadata_if_missing(documents)
        assert documents[0].metadata["document_id"] == "existing"

    def test_set_sequence_number_in_metadata(self):
        chunks = [
            Document(page_content="Chunk 1", metadata={"document_id": "doc1", "start_index": 0}),
            Document(page_content="Chunk 2", metadata={"document_id": "doc1", "start_index": 10}),
        ]
        result = BaseChunker._set_sequence_number_in_metadata(chunks)
        assert result[0].metadata["sequence_number"] == 1
        assert result[1].metadata["sequence_number"] == 2

    def test_set_sequence_number_sorts_by_start_index(self):
        chunks = [
            Document(page_content="Chunk 2", metadata={"document_id": "doc1", "start_index": 10}),
            Document(page_content="Chunk 1", metadata={"document_id": "doc1", "start_index": 0}),
        ]
        result = BaseChunker._set_sequence_number_in_metadata(chunks)
        assert result[0].metadata["start_index"] == 0
        assert result[1].metadata["start_index"] == 10
