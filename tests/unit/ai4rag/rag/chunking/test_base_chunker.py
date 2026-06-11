# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, Sequence

import pytest
from docling_core.types.doc import DoclingDocument

from ai4rag.rag.chunking.base_chunker import BaseChunker
from ai4rag.rag.chunking.chunk import AI4RAGChunk


class ConcreteChunker(BaseChunker):
    """Concrete implementation of BaseChunker for testing purposes."""

    def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
        return [AI4RAGChunk(text=doc.name[:5], metadata={}) for doc in documents]

    def to_dict(self) -> dict[str, Any]:
        return {"type": "concrete", "param": "value"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ConcreteChunker":
        return cls()


class TestBaseChunker:
    """Test suite for BaseChunker abstract base class."""

    def test_base_chunker_cannot_be_instantiated(self):
        with pytest.raises(TypeError) as exc_info:
            BaseChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_base_chunker_initialization_via_subclass(self):
        chunker = ConcreteChunker()
        assert isinstance(chunker, BaseChunker)

    def test_split_documents_is_abstract(self):

        class IncompleteChunker(BaseChunker):
            def to_dict(self) -> dict[str, Any]:
                return {}

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> "IncompleteChunker":
                return cls()

        with pytest.raises(TypeError) as exc_info:
            IncompleteChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_to_dict_is_abstract(self):

        class IncompleteChunker(BaseChunker):
            def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
                return []

            @classmethod
            def from_dict(cls, d: dict[str, Any]) -> "IncompleteChunker":
                return cls()

        with pytest.raises(TypeError) as exc_info:
            IncompleteChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_from_dict_is_abstract(self):

        class IncompleteChunker(BaseChunker):
            def split_documents(self, documents: Sequence[DoclingDocument]) -> list[AI4RAGChunk]:
                return []

            def to_dict(self) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as exc_info:
            IncompleteChunker()
        assert "Can't instantiate abstract class" in str(exc_info.value)

    def test_concrete_chunker_split_documents(self):
        chunker = ConcreteChunker()
        docs = [DoclingDocument(name="hello_world"), DoclingDocument(name="test_document")]
        result = chunker.split_documents(docs)
        assert len(result) == 2
        assert all(isinstance(c, AI4RAGChunk) for c in result)
        assert result[0].text == "hello"
        assert result[1].text == "test_"

    def test_concrete_chunker_to_dict(self):
        chunker = ConcreteChunker()
        result = chunker.to_dict()
        assert isinstance(result, dict)
        assert result["type"] == "concrete"

    def test_concrete_chunker_from_dict(self):
        chunker = ConcreteChunker.from_dict({"type": "concrete", "param": "value"})
        assert isinstance(chunker, ConcreteChunker)
        assert isinstance(chunker, BaseChunker)
