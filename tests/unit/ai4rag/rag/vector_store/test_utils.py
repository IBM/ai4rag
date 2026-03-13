# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest
from langchain_core.documents import Document

from ai4rag.rag.vector_store.utils import merge_window_into_a_document


class TestMergeWindowIntoADocument:
    """Test suite for merge_window_into_a_document function."""

    def test_merge_single_document(self):
        """Test merging a single document returns the same document."""
        doc = Document(page_content="Test content", metadata={"id": 1})
        result = merge_window_into_a_document([doc])

        assert isinstance(result, Document)
        assert result.page_content == "Test content"
        assert result.metadata == {"id": 1}

    def test_merge_multiple_documents_no_overlap(self):
        """Test merging multiple documents without overlapping text."""
        docs = [
            Document(page_content="First chunk", metadata={"id": 1}),
            Document(page_content="Second chunk", metadata={"id": 2}),
            Document(page_content="Third chunk", metadata={"id": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result, Document)
        assert result.page_content == "First chunk Second chunk Third chunk"

    def test_merge_multiple_documents_with_overlap(self):
        """Test merging documents with overlapping text."""
        docs = [
            Document(page_content="The quick brown", metadata={"id": 1}),
            Document(page_content="brown fox jumps", metadata={"id": 2}),
            Document(page_content="jumps over the", metadata={"id": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result, Document)
        assert result.page_content == "The quick brown fox jumps over the"

    def test_merge_documents_with_partial_overlap(self):
        """Test merging documents with partial overlapping text."""
        docs = [
            Document(page_content="Hello world", metadata={"seq": 1}),
            Document(page_content="world of Python", metadata={"seq": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result, Document)
        assert result.page_content == "Hello world of Python"

    def test_merge_metadata_single_values(self):
        """Test metadata merging with single unique values."""
        docs = [
            Document(page_content="A", metadata={"doc_id": "doc1", "seq": 1}),
            Document(page_content="B", metadata={"doc_id": "doc1", "seq": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["doc_id"] == "doc1"
        assert isinstance(result.metadata["seq"], list)
        assert sorted(result.metadata["seq"]) == [1, 2]

    def test_merge_metadata_multiple_values(self):
        """Test metadata merging with multiple different values."""
        docs = [
            Document(page_content="A", metadata={"doc_id": "doc1", "tag": "a"}),
            Document(page_content="B", metadata={"doc_id": "doc1", "tag": "b"}),
            Document(page_content="C", metadata={"doc_id": "doc1", "tag": "c"}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["doc_id"] == "doc1"
        assert isinstance(result.metadata["tag"], list)
        assert sorted(result.metadata["tag"]) == ["a", "b", "c"]

    def test_merge_metadata_with_lists(self):
        """Test metadata merging when metadata contains lists."""
        docs = [
            Document(page_content="A", metadata={"tags": ["tag1", "tag2"]}),
            Document(page_content="B", metadata={"tags": ["tag2", "tag3"]}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result.metadata["tags"], list)
        assert sorted(result.metadata["tags"]) == ["tag1", "tag2", "tag3"]

    def test_merge_empty_documents(self):
        """Test merging documents with empty content."""
        docs = [
            Document(page_content="", metadata={"id": 1}),
            Document(page_content="", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == ""

    def test_merge_documents_complex_overlap(self):
        """Test merging with complex overlapping patterns."""
        docs = [
            Document(page_content="The cat sat", metadata={"pos": 1}),
            Document(page_content="cat sat on", metadata={"pos": 2}),
            Document(page_content="on the mat", metadata={"pos": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "The cat sat on the mat"

    def test_merge_documents_no_common_suffix_prefix(self):
        """Test merging documents with no common suffix/prefix."""
        docs = [
            Document(page_content="apple", metadata={"fruit": True}),
            Document(page_content="banana", metadata={"fruit": True}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "apple banana"

    def test_merge_documents_exact_duplicates(self):
        """Test merging documents with exact duplicate text."""
        docs = [
            Document(page_content="duplicate", metadata={"id": 1}),
            Document(page_content="duplicate", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "duplicate"

    def test_merge_metadata_mixed_types(self):
        """Test metadata merging with mixed data types."""
        docs = [
            Document(page_content="A", metadata={"count": 1, "name": "first"}),
            Document(page_content="B", metadata={"count": 2, "name": "second"}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result.metadata["count"], list)
        assert sorted(result.metadata["count"]) == [1, 2]
        assert isinstance(result.metadata["name"], list)
        assert sorted(result.metadata["name"]) == ["first", "second"]

    def test_merge_long_overlapping_sections(self):
        """Test merging with long overlapping sections."""
        docs = [
            Document(page_content="This is a long sentence with many words", metadata={"id": 1}),
            Document(page_content="sentence with many words and more content", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "This is a long sentence with many words and more content"

    def test_merge_preserves_whitespace(self):
        """Test that merging properly handles whitespace."""
        docs = [
            Document(page_content="First", metadata={"id": 1}),
            Document(page_content="Second", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "First Second"
        assert "  " not in result.page_content

    def test_merge_documents_with_special_characters(self):
        """Test merging documents containing special characters."""
        docs = [
            Document(page_content="Hello, world!", metadata={"id": 1}),
            Document(page_content="world! How are", metadata={"id": 2}),
            Document(page_content="are you?", metadata={"id": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "Hello, world! How are you?"

    def test_merge_metadata_preserves_order(self):
        """Test that metadata values are sorted consistently."""
        docs = [
            Document(page_content="A", metadata={"seq": 5}),
            Document(page_content="B", metadata={"seq": 1}),
            Document(page_content="C", metadata={"seq": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["seq"] == [1, 3, 5]

    def test_merge_two_documents_complete_overlap(self):
        """Test merging two documents where second is completely contained in first."""
        docs = [
            Document(page_content="The complete text", metadata={"id": 1}),
            Document(page_content="text", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.page_content == "The complete text"

    def test_merge_documents_case_sensitive(self):
        """Test that merging is case-sensitive."""
        docs = [
            Document(page_content="Hello World", metadata={"id": 1}),
            Document(page_content="world is great", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        # "World" != "world", so no overlap
        assert result.page_content == "Hello World world is great"

    def test_merge_with_numeric_metadata(self):
        """Test merging with various numeric metadata types."""
        docs = [
            Document(page_content="A", metadata={"int": 1, "float": 1.5}),
            Document(page_content="B", metadata={"int": 2, "float": 2.5}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["int"] == [1, 2]
        assert result.metadata["float"] == [1.5, 2.5]
