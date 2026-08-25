# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import re
from types import SimpleNamespace

import pytest

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.vector_store.utils import (
    COLLECTION_NAME_PREFIX,
    generate_collection_name,
    merge_window_into_a_document,
    resolve_collection_name,
    resolve_embedding_dimension,
    sanitize_collection_name,
    validate_search_params,
)


class TestSanitizeCollectionName:

    def test_alphanumeric_passthrough(self):
        assert sanitize_collection_name("abc123") == "abc123"

    def test_underscores_preserved(self):
        assert sanitize_collection_name("my_collection") == "my_collection"

    def test_hyphens_replaced(self):
        assert sanitize_collection_name("my-collection") == "my_collection"

    def test_dots_replaced(self):
        assert sanitize_collection_name("v1.2.3") == "v1_2_3"

    def test_spaces_replaced(self):
        assert sanitize_collection_name("hello world") == "hello_world"

    def test_empty_string(self):
        assert sanitize_collection_name("") == ""


class TestGenerateCollectionName:
    """Test suite for the generate_collection_name naming convention."""

    _PATTERN = re.compile(r"^ai4rag_\d{14}_[a-z0-9]{8}$")

    def test_matches_naming_convention(self):
        name = generate_collection_name()
        assert self._PATTERN.match(name), f"{name!r} does not match ai4rag_<timestamp>_<8 random chars>"

    def test_successive_calls_are_unique(self):
        names = {generate_collection_name() for _ in range(50)}
        assert len(names) == 50

    def test_suffix_is_lowercase_alphanumeric(self):
        name = generate_collection_name()
        suffix = name.rsplit("_", maxsplit=1)[-1]
        assert len(suffix) == 8
        assert suffix == suffix.lower()
        assert suffix.isalnum()


class TestResolveCollectionName:
    """Single entry point that generates, guards, and sanitizes collection names."""

    _GENERATED_PATTERN = re.compile(r"^ai4rag_\d{14}_[a-z0-9]{8}$")

    def test_none_generates_compliant_name(self):
        name = resolve_collection_name(None)
        assert self._GENERATED_PATTERN.match(name), f"{name!r} is not a generated ai4rag name"

    def test_compliant_name_passes_through(self):
        assert resolve_collection_name("ai4rag_my_col") == "ai4rag_my_col"

    def test_compliant_name_is_sanitized(self):
        # The mandatory prefix is kept; unsafe characters become underscores.
        assert resolve_collection_name("ai4rag-my.col") == "ai4rag_my_col"

    def test_missing_prefix_is_rejected(self):
        with pytest.raises(ValueError, match=f"must start with '{COLLECTION_NAME_PREFIX}'"):
            resolve_collection_name("my_col")

    def test_prefix_check_precedes_sanitization(self):
        # A name that would only satisfy the prefix *after* sanitization must still
        # be rejected: the guard runs on the raw input, not the cleaned form.
        with pytest.raises(ValueError, match=f"must start with '{COLLECTION_NAME_PREFIX}'"):
            resolve_collection_name("xxai4rag_col")

    def test_over_length_name_is_rejected(self):
        too_long = COLLECTION_NAME_PREFIX + "_" + "a" * 64
        with pytest.raises(ValueError, match="exceeds the maximum length"):
            resolve_collection_name(too_long)

    def test_max_length_name_is_accepted(self):
        exactly_63 = COLLECTION_NAME_PREFIX + "_" + "a" * (63 - len(COLLECTION_NAME_PREFIX) - 1)
        assert len(exactly_63) == 63
        assert resolve_collection_name(exactly_63) == exactly_63


class TestMergeWindowIntoADocument:
    """Test suite for merge_window_into_a_document function."""

    def test_merge_single_document(self):
        """Test merging a single document returns the same document."""
        doc = AI4RAGChunk(text="Test content", metadata={"id": 1})
        result = merge_window_into_a_document([doc])

        assert isinstance(result, AI4RAGChunk)
        assert result.text == "Test content"
        assert result.metadata == {"id": 1}

    def test_merge_multiple_documents_no_overlap(self):
        """Test merging multiple documents without overlapping text."""
        docs = [
            AI4RAGChunk(text="First chunk", metadata={"id": 1}),
            AI4RAGChunk(text="Second chunk", metadata={"id": 2}),
            AI4RAGChunk(text="Third chunk", metadata={"id": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result, AI4RAGChunk)
        assert result.text == "First chunk Second chunk Third chunk"

    def test_merge_multiple_documents_with_overlap(self):
        """Test merging documents with overlapping text."""
        docs = [
            AI4RAGChunk(text="The quick brown", metadata={"id": 1}),
            AI4RAGChunk(text="brown fox jumps", metadata={"id": 2}),
            AI4RAGChunk(text="jumps over the", metadata={"id": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result, AI4RAGChunk)
        assert result.text == "The quick brown fox jumps over the"

    def test_merge_documents_with_partial_overlap(self):
        """Test merging documents with partial overlapping text."""
        docs = [
            AI4RAGChunk(text="Hello world", metadata={"seq": 1}),
            AI4RAGChunk(text="world of Python", metadata={"seq": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result, AI4RAGChunk)
        assert result.text == "Hello world of Python"

    def test_merge_metadata_single_values(self):
        """Test metadata merging with single unique values."""
        docs = [
            AI4RAGChunk(text="A", metadata={"doc_id": "doc1", "seq": 1}),
            AI4RAGChunk(text="B", metadata={"doc_id": "doc1", "seq": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["doc_id"] == "doc1"
        assert isinstance(result.metadata["seq"], list)
        assert sorted(result.metadata["seq"]) == [1, 2]

    def test_merge_metadata_multiple_values(self):
        """Test metadata merging with multiple different values."""
        docs = [
            AI4RAGChunk(text="A", metadata={"doc_id": "doc1", "tag": "a"}),
            AI4RAGChunk(text="B", metadata={"doc_id": "doc1", "tag": "b"}),
            AI4RAGChunk(text="C", metadata={"doc_id": "doc1", "tag": "c"}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["doc_id"] == "doc1"
        assert isinstance(result.metadata["tag"], list)
        assert sorted(result.metadata["tag"]) == ["a", "b", "c"]

    def test_merge_metadata_with_lists(self):
        """Test metadata merging when metadata contains lists."""
        docs = [
            AI4RAGChunk(text="A", metadata={"tags": ["tag1", "tag2"]}),
            AI4RAGChunk(text="B", metadata={"tags": ["tag2", "tag3"]}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result.metadata["tags"], list)
        assert sorted(result.metadata["tags"]) == ["tag1", "tag2", "tag3"]

    def test_merge_empty_documents(self):
        """Test merging documents with empty content."""
        docs = [
            AI4RAGChunk(text="", metadata={"id": 1}),
            AI4RAGChunk(text="", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == ""

    def test_merge_documents_complex_overlap(self):
        """Test merging with complex overlapping patterns."""
        docs = [
            AI4RAGChunk(text="The cat sat", metadata={"pos": 1}),
            AI4RAGChunk(text="cat sat on", metadata={"pos": 2}),
            AI4RAGChunk(text="on the mat", metadata={"pos": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "The cat sat on the mat"

    def test_merge_documents_no_common_suffix_prefix(self):
        """Test merging documents with no common suffix/prefix."""
        docs = [
            AI4RAGChunk(text="apple", metadata={"fruit": True}),
            AI4RAGChunk(text="banana", metadata={"fruit": True}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "apple banana"

    def test_merge_documents_exact_duplicates(self):
        """Test merging documents with exact duplicate text."""
        docs = [
            AI4RAGChunk(text="duplicate", metadata={"id": 1}),
            AI4RAGChunk(text="duplicate", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "duplicate"

    def test_merge_metadata_mixed_types(self):
        """Test metadata merging with mixed data types."""
        docs = [
            AI4RAGChunk(text="A", metadata={"count": 1, "name": "first"}),
            AI4RAGChunk(text="B", metadata={"count": 2, "name": "second"}),
        ]
        result = merge_window_into_a_document(docs)

        assert isinstance(result.metadata["count"], list)
        assert sorted(result.metadata["count"]) == [1, 2]
        assert isinstance(result.metadata["name"], list)
        assert sorted(result.metadata["name"]) == ["first", "second"]

    def test_merge_long_overlapping_sections(self):
        """Test merging with long overlapping sections."""
        docs = [
            AI4RAGChunk(text="This is a long sentence with many words", metadata={"id": 1}),
            AI4RAGChunk(text="sentence with many words and more content", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "This is a long sentence with many words and more content"

    def test_merge_preserves_whitespace(self):
        """Test that merging properly handles whitespace."""
        docs = [
            AI4RAGChunk(text="First", metadata={"id": 1}),
            AI4RAGChunk(text="Second", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "First Second"
        assert "  " not in result.text

    def test_merge_documents_with_special_characters(self):
        """Test merging documents containing special characters."""
        docs = [
            AI4RAGChunk(text="Hello, world!", metadata={"id": 1}),
            AI4RAGChunk(text="world! How are", metadata={"id": 2}),
            AI4RAGChunk(text="are you?", metadata={"id": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "Hello, world! How are you?"

    def test_merge_metadata_preserves_order(self):
        """Test that metadata values are sorted consistently."""
        docs = [
            AI4RAGChunk(text="A", metadata={"seq": 5}),
            AI4RAGChunk(text="B", metadata={"seq": 1}),
            AI4RAGChunk(text="C", metadata={"seq": 3}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["seq"] == [1, 3, 5]

    def test_merge_two_documents_complete_overlap(self):
        """Test merging two documents where second is completely contained in first."""
        docs = [
            AI4RAGChunk(text="The complete text", metadata={"id": 1}),
            AI4RAGChunk(text="text", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.text == "The complete text"

    def test_merge_documents_case_sensitive(self):
        """Test that merging is case-sensitive."""
        docs = [
            AI4RAGChunk(text="Hello World", metadata={"id": 1}),
            AI4RAGChunk(text="world is great", metadata={"id": 2}),
        ]
        result = merge_window_into_a_document(docs)

        # "World" != "world", so no overlap
        assert result.text == "Hello World world is great"

    def test_merge_with_numeric_metadata(self):
        """Test merging with various numeric metadata types."""
        docs = [
            AI4RAGChunk(text="A", metadata={"int": 1, "float": 1.5}),
            AI4RAGChunk(text="B", metadata={"int": 2, "float": 2.5}),
        ]
        result = merge_window_into_a_document(docs)

        assert result.metadata["int"] == [1, 2]
        assert result.metadata["float"] == [1.5, 2.5]


class TestValidateSearchParams:
    """Backend-agnostic validation of the search mode / hybrid ranker combination."""

    def test_vector_mode_without_ranker_params_passes(self):
        # None must be a valid, complete call — the default pure-vector search.
        assert validate_search_params("vector", None, None, None) is None

    @pytest.mark.parametrize("strategy", ["rrf", "weighted", "normalized"])
    def test_hybrid_mode_with_valid_strategy_passes(self, strategy):
        assert validate_search_params("hybrid", strategy, None, None) is None

    def test_hybrid_rrf_with_k_passes(self):
        assert validate_search_params("hybrid", "rrf", 60, None) is None

    def test_hybrid_weighted_with_alpha_passes(self):
        assert validate_search_params("hybrid", "weighted", None, 0.5) is None

    def test_invalid_search_mode_rejected(self):
        with pytest.raises(ValueError, match="Invalid search_mode"):
            validate_search_params("invalid", None, None, None)

    def test_ranker_strategy_on_vector_mode_rejected(self):
        with pytest.raises(ValueError, match="only valid when search_mode='hybrid'"):
            validate_search_params("vector", "rrf", None, None)

    def test_ranker_k_on_vector_mode_rejected(self):
        with pytest.raises(ValueError, match="ranker_k=60 is only valid when search_mode='hybrid'"):
            validate_search_params("vector", None, 60, None)

    def test_ranker_alpha_on_vector_mode_rejected(self):
        with pytest.raises(ValueError, match="ranker_alpha=0.5 is only valid when search_mode='hybrid'"):
            validate_search_params("vector", None, None, 0.5)

    def test_hybrid_without_strategy_rejected(self):
        with pytest.raises(ValueError, match="ranker_strategy must be set"):
            validate_search_params("hybrid", None, None, None)

    def test_hybrid_with_invalid_strategy_rejected(self):
        with pytest.raises(ValueError, match="Invalid ranker_strategy"):
            validate_search_params("hybrid", "bogus", None, None)

    def test_ranker_k_with_non_rrf_strategy_rejected(self):
        with pytest.raises(ValueError, match="ranker_k=60 is only valid when ranker_strategy='rrf'"):
            validate_search_params("hybrid", "weighted", 60, None)

    def test_ranker_alpha_with_non_weighted_strategy_rejected(self):
        with pytest.raises(ValueError, match="ranker_alpha=0.5 is only valid when ranker_strategy='weighted'"):
            validate_search_params("hybrid", "rrf", None, 0.5)

    def test_neutral_sentinels_are_not_treated_as_set(self):
        # An empty strategy, k<=0, and alpha==1 are the "unset" sentinels and must
        # not trip the vector-mode guards.
        assert validate_search_params("vector", "", 0, 1) is None


class TestResolveEmbeddingDimension:
    """Reading the dense vector dimension from either params shape."""

    def test_dict_params(self):
        model = SimpleNamespace(params={"embedding_dimension": 384})
        assert resolve_embedding_dimension(model) == 384

    def test_object_params(self):
        model = SimpleNamespace(params=SimpleNamespace(embedding_dimension=768))
        assert resolve_embedding_dimension(model) == 768
