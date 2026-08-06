# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ai4rag.components.optimization.search_space_preparation import (
    SearchSpaceReport,
    _validate_model_list,
    prepare_search_space_report,
)
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames
from ai4rag.utils.validators import validate_model_list

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


# ---------------------------------------------------------------------------
# Search space filtering
# ---------------------------------------------------------------------------


class TestPrepareSearchSpaceReportFiltering:
    """Test that rules are applied and the report reflects only valid combinations."""

    def _make_search_space(self, chunking_methods, chunk_sizes) -> AI4RAGSearchSpace:
        mock_em = MagicMock()
        mock_em.params.context_length = None  # prevent _rule_chunk_size_within_embedding_context_length from failing
        return AI4RAGSearchSpace(
            params=[
                Parameter(name=AI4RAGParamNames.FOUNDATION_MODEL, values=(MagicMock(),)),
                Parameter(name=AI4RAGParamNames.EMBEDDING_MODEL, values=(mock_em,)),
                Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=tuple(chunking_methods)),
                Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=tuple(chunk_sizes)),
            ]
        )

    def test_recursive_with_too_small_chunk_sizes_yields_empty_search_space(self, mocker):
        """chunk_sizes=[128, 256] with recursive and default overlaps (0, 128, 256) produce no valid combinations.

        - overlap=0   is filtered by _rule_chunk_overlap_for_chunking_method (recursive needs overlap > 0)
        - overlap=128 is filtered by _rule_chunk_size_bigger_than_chunk_overlap (256 > 2*128 is False)
        - overlap=256 is filtered by _rule_chunk_size_bigger_than_chunk_overlap (256 > 2*256 is False)
        """
        search_space = self._make_search_space(["recursive"], [128, 256])

        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.prepare_search_space_with_ogx",
            return_value=search_space,
        )
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.pd.read_json",
            return_value=MagicMock(),
        )
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.load_docling_documents",
            return_value=[],
        )
        mocker.patch("ai4rag.components.optimization.search_space_preparation.BenchmarkData")
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation._serialize_model",
            return_value={"model_id": "mock"},
        )

        result = prepare_search_space_report(
            test_data_path="dummy.json",
            extracted_text_path="dummy_dir",
            ogx_client=MagicMock(),
            chunking_methods=["recursive"],
            chunk_sizes=[128, 256],
        )

        assert result.search_space["chunk_size"] == []
        assert result.search_space["chunk_overlap"] == []
        assert result.search_space["chunking_method"] == []


# ---------------------------------------------------------------------------
# validate_model_list shared location
# ---------------------------------------------------------------------------


class TestValidateModelListShared:
    """Verify that the shared validator is accessible from both locations."""

    def test_shared_validator_is_same_function(self):
        """The alias in search_space_preparation must point to the shared function."""
        assert _validate_model_list is validate_model_list


# ---------------------------------------------------------------------------
# pre_validated_search_space parameter
# ---------------------------------------------------------------------------


class TestPrepareSearchSpaceReportPreValidated:
    """Test the pre_validated_search_space bypass path."""

    def _make_search_space(self) -> AI4RAGSearchSpace:
        mock_em = MagicMock()
        mock_em.params.context_length = None
        return AI4RAGSearchSpace(
            params=[
                Parameter(name=AI4RAGParamNames.FOUNDATION_MODEL, values=(MagicMock(),)),
                Parameter(name=AI4RAGParamNames.EMBEDDING_MODEL, values=(mock_em,)),
                Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=("recursive",)),
                Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=(512,)),
                Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, values=(128,)),
            ]
        )

    def test_skips_ogx_call_when_pre_validated(self, mocker):
        """prepare_search_space_with_ogx must not be called when pre_validated_search_space is given."""
        search_space = self._make_search_space()
        mock_prepare = mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.prepare_search_space_with_ogx",
        )
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.pd.read_json",
            return_value=MagicMock(),
        )
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.load_docling_documents",
            return_value=[],
        )
        mocker.patch("ai4rag.components.optimization.search_space_preparation.BenchmarkData")
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation._serialize_model",
            return_value={"model_id": "mock"},
        )

        prepare_search_space_report(
            test_data_path="dummy.json",
            extracted_text_path="dummy_dir",
            ogx_client=MagicMock(),
            pre_validated_search_space=search_space,
        )

        mock_prepare.assert_not_called()

    def test_uses_provided_search_space(self, mocker):
        """The report must reflect parameters from the pre-validated search space."""
        search_space = self._make_search_space()
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.pd.read_json",
            return_value=MagicMock(),
        )
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.load_docling_documents",
            return_value=[],
        )
        mocker.patch("ai4rag.components.optimization.search_space_preparation.BenchmarkData")
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation._serialize_model",
            return_value={"model_id": "mock"},
        )

        result = prepare_search_space_report(
            test_data_path="dummy.json",
            extracted_text_path="dummy_dir",
            ogx_client=MagicMock(),
            pre_validated_search_space=search_space,
        )

        assert result.search_space["chunk_size"] == [512]
        assert result.search_space["chunk_overlap"] == [128]
        assert result.search_space["chunking_method"] == ["recursive"]

    def test_still_loads_documents(self, mocker):
        """Documents must still be loaded even when pre_validated_search_space is given (needed for MPS)."""
        search_space = self._make_search_space()
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.pd.read_json",
            return_value=MagicMock(),
        )
        mock_load_docs = mocker.patch(
            "ai4rag.components.optimization.search_space_preparation.load_docling_documents",
            return_value=[],
        )
        mocker.patch("ai4rag.components.optimization.search_space_preparation.BenchmarkData")
        mocker.patch(
            "ai4rag.components.optimization.search_space_preparation._serialize_model",
            return_value={"model_id": "mock"},
        )

        prepare_search_space_report(
            test_data_path="dummy.json",
            extracted_text_path="dummy_dir",
            ogx_client=MagicMock(),
            pre_validated_search_space=search_space,
        )

        mock_load_docs.assert_called_once_with("dummy_dir")
