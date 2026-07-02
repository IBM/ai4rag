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


class TestSpeedPresetSearchSpace:
    """Test that preset='speed' constrains the search space."""

    def test_speed_preset_constrains_chunk_sizes(self, mock_ogx_client):
        """When preset='speed', the report must contain only chunk_size [128, 256]."""
        from unittest.mock import patch

        from ai4rag.components.optimization import search_space_preparation as mod
        from ai4rag.search_space.src.parameter import Parameter

        fake_fm = MagicMock()
        fake_fm.model_id = "fm-a"

        fake_em = MagicMock()
        fake_em.model_id = "em-a"
        # Required by _rule_chunk_size_within_embedding_context_length
        # which compares chunk_size against embedding context_length.
        fake_em.params.context_length = 8192

        search_space_items = {
            "chunking_method": Parameter(name="chunking_method", values=["recursive", "hybrid"]),
            "chunk_size": Parameter(name="chunk_size", values=[512, 1024, 2048]),
            "foundation_model": Parameter(name="foundation_model", values=[fake_fm]),
            "embedding_model": Parameter(name="embedding_model", values=[fake_em]),
        }
        fake_search_space = MagicMock()
        fake_search_space._search_space = search_space_items
        fake_search_space.__getitem__ = lambda self, key: search_space_items[key]

        fake_benchmark_df = MagicMock(spec=mod.pd.DataFrame)
        fake_benchmark_df.__len__ = lambda self: 1

        with (
            patch.object(mod, "prepare_search_space_with_ogx", return_value=fake_search_space),
            patch.object(mod.pd, "read_json", return_value=fake_benchmark_df),
            patch.object(mod, "load_docling_documents", return_value=[]),
            patch.object(mod, "BenchmarkData"),
        ):
            report = prepare_search_space_report(
                test_data_path="dummy.json",
                extracted_text_path="dummy_dir",
                ogx_client=mock_ogx_client,
                preset="speed",
            )

        assert report.search_space["chunk_size"] == [128, 256]


class TestUnsupportedChunkingMethods:
    """Test that providing chunking methods not in the search space raises."""

    def test_unsupported_method_raises_value_error(self, mock_ogx_client):
        """A chunking method absent from the search space must raise ValueError."""
        from unittest.mock import patch

        from ai4rag.components.optimization import search_space_preparation as mod

        search_space_items = {
            "chunking_method": MagicMock(
                values=["recursive", "hybrid"], all_values=MagicMock(return_value=["recursive", "hybrid"])
            ),
            "chunk_size": MagicMock(values=[256], all_values=MagicMock(return_value=[256])),
            "foundation_model": MagicMock(values=[MagicMock()]),
            "embedding_model": MagicMock(values=[MagicMock()]),
        }
        fake_search_space = MagicMock()
        fake_search_space._search_space = search_space_items
        fake_search_space.__getitem__ = lambda self, key: search_space_items[key]

        fake_benchmark_df = MagicMock(spec=mod.pd.DataFrame)
        fake_benchmark_df.__len__ = lambda self: 1

        with (
            patch.object(mod, "prepare_search_space_with_ogx", return_value=fake_search_space),
            patch.object(mod.pd, "read_json", return_value=fake_benchmark_df),
            patch.object(mod, "load_docling_documents", return_value=[]),
            patch.object(mod, "BenchmarkData"),
        ):
            with pytest.raises(ValueError, match="Unsupported chunking methods"):
                prepare_search_space_report(
                    test_data_path="dummy.json",
                    extracted_text_path="dummy_dir",
                    ogx_client=mock_ogx_client,
                    chunking_methods=["semantic"],
                )
