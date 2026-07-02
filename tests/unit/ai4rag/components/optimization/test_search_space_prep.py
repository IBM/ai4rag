# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai4rag.components.optimization.search_space_preparation import (
    SUPPORTED_METRICS,
    SearchSpaceReport,
    _validate_chunk_sizes,
    _validate_chunking_methods,
    _validate_model_list,
    prepare_search_space_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_report() -> SearchSpaceReport:
    """Return a minimal SearchSpaceReport for serialization tests."""
    return SearchSpaceReport(
        search_space={
            "foundation_model": ["model-a", "model-b"],
            "embedding_model": ["emb-a"],
            "chunk_size": [256, 512],
        },
        selected_models={
            "foundation_model": ["model-a"],
            "embedding_model": ["emb-a"],
        },
    )


@pytest.fixture()
def mock_ogx_client() -> MagicMock:
    """Return a bare MagicMock standing in for OgxClient."""
    return MagicMock()


# ---------------------------------------------------------------------------
# _validate_model_list
# ---------------------------------------------------------------------------


class TestValidateModelList:
    """Tests for the _validate_model_list helper."""

    def test_none_is_valid(self):
        """None means 'use server defaults' and must not raise."""
        _validate_model_list(None, "embedding_models")  # no exception

    def test_valid_list_passes(self):
        """A list of non-empty strings must be accepted."""
        _validate_model_list(["model-a", "model-b"], "generation_models")  # no exception

    def test_empty_string_raises_type_error(self):
        """An empty string inside the list must raise TypeError."""
        with pytest.raises(TypeError, match=r"generation_models\[1\] must be a non-empty string"):
            _validate_model_list(["model-a", "", "model-c"], "generation_models")

    def test_non_list_raises_type_error(self):
        """A non-list value must raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            _validate_model_list("not-a-list", "embedding_models")  # type: ignore[arg-type]

    def test_empty_list_passes(self):
        """An empty list is valid (no models specified)."""
        _validate_model_list([], "embedding_models")  # no exception


# ---------------------------------------------------------------------------
# SearchSpaceReport.save_json
# ---------------------------------------------------------------------------


class TestSearchSpaceReportSaveJson:
    """Tests for JSON serialization of SearchSpaceReport."""

    def test_save_json_creates_file(self, simple_report, tmp_path: Path):
        """save_json must create a readable JSON file at the given path."""
        import json

        out_file = tmp_path / "report.json"
        simple_report.save_json(out_file)

        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert isinstance(data, dict)

    def test_save_json_creates_parent_directories(self, simple_report, tmp_path: Path):
        """save_json must create intermediate directories if they do not exist."""
        out_file = tmp_path / "nested" / "dir" / "report.json"
        simple_report.save_json(out_file)

        assert out_file.exists()

    def test_save_json_preserves_search_space_keys(self, simple_report, tmp_path: Path):
        """All top-level search_space keys must appear in the serialized JSON."""
        import json

        out_file = tmp_path / "report.json"
        simple_report.save_json(out_file)

        data = json.loads(out_file.read_text())
        assert "foundation_model" in data
        assert "embedding_model" in data
        assert "chunk_size" in data


# ---------------------------------------------------------------------------
# prepare_search_space_report -- input validation only
# ---------------------------------------------------------------------------


class TestPrepareSearchSpaceReportValidation:
    """Test input validation in prepare_search_space_report.

    These tests verify that the function rejects bad inputs before
    reaching any heavy I/O or OGX calls.
    """

    def test_invalid_metric_raises_value_error(self, mock_ogx_client):
        """An unsupported metric string must raise ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                metric="invalid_metric",
            )

    def test_supported_metrics_constant(self):
        """SUPPORTED_METRICS must contain the three canonical metrics."""
        assert "faithfulness" in SUPPORTED_METRICS
        assert "answer_correctness" in SUPPORTED_METRICS
        assert "context_correctness" in SUPPORTED_METRICS

    def test_invalid_embedding_models_raises_type_error(self, mock_ogx_client):
        """An empty string in embedding_models must raise TypeError."""
        with pytest.raises(TypeError, match="non-empty string"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                embedding_models=["good-model", ""],
            )

    def test_invalid_generation_models_raises_type_error(self, mock_ogx_client):
        """An empty string in generation_models must raise TypeError."""
        with pytest.raises(TypeError, match="non-empty string"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                generation_models=["", "model-b"],
            )

    def test_non_list_models_raises_type_error(self, mock_ogx_client):
        """A non-list value for model lists must raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                embedding_models="not-a-list",  # type: ignore[arg-type]
            )

    def test_invalid_chunking_methods_raises_type_error(self, mock_ogx_client):
        """A non-list value for chunking_methods must raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                chunking_methods="recursive",  # type: ignore[arg-type]
            )

    def test_empty_chunking_methods_raises_value_error(self, mock_ogx_client):
        """An empty list for chunking_methods must raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                chunking_methods=[],
            )


# ---------------------------------------------------------------------------
# _validate_chunking_methods
# ---------------------------------------------------------------------------


class TestValidateChunkingMethods:
    """Tests for the ``_validate_chunking_methods`` helper."""

    def test_none_is_valid(self):
        """None means 'use defaults' and must not raise."""
        _validate_chunking_methods(None)

    def test_valid_list_passes(self):
        """A list of non-empty strings must be accepted."""
        _validate_chunking_methods(["recursive", "hybrid"])

    def test_single_element_passes(self):
        """A single-element list must be accepted."""
        _validate_chunking_methods(["recursive"])

    def test_non_list_raises_type_error(self):
        """A non-list value must raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            _validate_chunking_methods("recursive")  # type: ignore[arg-type]

    def test_empty_list_raises_value_error(self):
        """An empty list must raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_chunking_methods([])

    def test_empty_string_raises_type_error(self):
        """An empty string element must raise TypeError."""
        with pytest.raises(TypeError, match=r"chunking_methods\[0\] must be a non-empty string"):
            _validate_chunking_methods([""])

    def test_non_string_raises_type_error(self):
        """A non-string element must raise TypeError."""
        with pytest.raises(TypeError, match=r"chunking_methods\[1\] must be a non-empty string"):
            _validate_chunking_methods(["recursive", 42])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# prepare_search_space_report — unsupported chunking_methods
# ---------------------------------------------------------------------------


class TestUnsupportedChunkingMethods:
    """Test that providing chunking methods not in the search space raises."""

    def test_unsupported_method_raises_value_error(self, mock_ogx_client):
        """A chunking method not in ChunkingConstraints.METHODS must raise ValueError.

        Validation is now performed eagerly before any I/O so no mocking is needed.
        """
        with pytest.raises(ValueError, match="Unsupported chunking methods"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                chunking_methods=["semantic"],
            )


# ---------------------------------------------------------------------------
# _validate_chunk_sizes
# ---------------------------------------------------------------------------


class TestValidateChunkSizes:
    """Tests for the ``_validate_chunk_sizes`` helper."""

    def test_none_is_valid(self):
        """None means 'use defaults' and must not raise."""
        _validate_chunk_sizes(None)

    def test_valid_list_passes(self):
        """A list of positive integers must be accepted."""
        _validate_chunk_sizes([256, 512, 1024])

    def test_single_element_passes(self):
        """A single-element list must be accepted."""
        _validate_chunk_sizes([512])

    def test_non_list_raises_type_error(self):
        """A non-list value must raise TypeError."""
        with pytest.raises(TypeError, match="must be a list"):
            _validate_chunk_sizes(512)  # type: ignore[arg-type]

    def test_empty_list_raises_value_error(self):
        """An empty list must raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            _validate_chunk_sizes([])

    def test_zero_raises_type_error(self):
        """Zero is not a positive integer and must raise TypeError."""
        with pytest.raises(TypeError, match=r"chunk_sizes\[0\] must be a positive integer"):
            _validate_chunk_sizes([0])

    def test_negative_raises_type_error(self):
        """A negative value must raise TypeError."""
        with pytest.raises(TypeError, match=r"chunk_sizes\[1\] must be a positive integer"):
            _validate_chunk_sizes([512, -1])

    def test_float_raises_type_error(self):
        """A float value must raise TypeError even if it looks like an integer."""
        with pytest.raises(TypeError, match=r"chunk_sizes\[0\] must be a positive integer"):
            _validate_chunk_sizes([512.0])  # type: ignore[list-item]

    def test_below_min_raises_value_error(self):
        """A size below ChunkingConstraints.MIN_CHUNK_SIZE must raise ValueError."""
        from ai4rag.utils.constants import ChunkingConstraints

        below_min = ChunkingConstraints.MIN_CHUNK_SIZE - 1
        with pytest.raises(ValueError, match="out of range"):
            _validate_chunk_sizes([below_min])

    def test_above_max_raises_value_error(self):
        """A size above ChunkingConstraints.MAX_CHUNK_SIZE must raise ValueError."""
        from ai4rag.utils.constants import ChunkingConstraints

        above_max = ChunkingConstraints.MAX_CHUNK_SIZE + 1
        with pytest.raises(ValueError, match="out of range"):
            _validate_chunk_sizes([above_max])

    def test_boundary_values_pass(self):
        """MIN_CHUNK_SIZE and MAX_CHUNK_SIZE themselves must be accepted."""
        from ai4rag.utils.constants import ChunkingConstraints

        _validate_chunk_sizes([ChunkingConstraints.MIN_CHUNK_SIZE, ChunkingConstraints.MAX_CHUNK_SIZE])


# ---------------------------------------------------------------------------
# prepare_search_space_report — chunk_sizes input validation
# ---------------------------------------------------------------------------


class TestPrepareSearchSpaceReportChunkSizesValidation:
    """Test chunk_sizes input validation in prepare_search_space_report."""

    def test_non_list_chunk_sizes_raises_type_error(self, mock_ogx_client):
        """A non-list chunk_sizes must raise TypeError before any I/O."""
        with pytest.raises(TypeError, match="must be a list"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                chunk_sizes=512,  # type: ignore[arg-type]
            )

    def test_empty_chunk_sizes_raises_value_error(self, mock_ogx_client):
        """An empty chunk_sizes list must raise ValueError before any I/O."""
        with pytest.raises(ValueError, match="must not be empty"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                chunk_sizes=[],
            )

    def test_non_positive_chunk_size_raises_type_error(self, mock_ogx_client):
        """A non-positive integer in chunk_sizes must raise TypeError before any I/O."""
        with pytest.raises(TypeError, match="must be a positive integer"):
            prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                chunk_sizes=[0],
            )
