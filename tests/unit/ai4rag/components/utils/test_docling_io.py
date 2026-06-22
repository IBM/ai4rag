# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for :mod:`ai4rag.components.utils.docling_io` -- DoclingDocument file loading."""

from __future__ import annotations

import pytest


class TestLoadDoclingDocuments:
    """Test suite for :func:`load_docling_documents`."""

    @pytest.fixture
    def mock_load_from_json(self, mocker):
        """Patch ``DoclingDocument.load_from_json`` to return a lightweight stub."""
        mock = mocker.patch("ai4rag.components.utils.docling_io.DoclingDocument.load_from_json")
        mock.side_effect = lambda p: mocker.MagicMock(name=f"DoclingDocument({p.name})")
        return mock

    # ------------------------------------------------------------------
    # Single file
    # ------------------------------------------------------------------

    def test_loads_single_json_file(self, tmp_path, mock_load_from_json):
        """A path pointing to a single ``.json`` file loads exactly one document."""
        json_file = tmp_path / "doc.json"
        json_file.write_text("{}")

        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(json_file)

        assert len(result) == 1
        mock_load_from_json.assert_called_once_with(json_file)

    def test_loads_single_file_with_string_path(self, tmp_path, mock_load_from_json):
        """Accepting a plain ``str`` path should work identically to ``Path``."""
        json_file = tmp_path / "doc.json"
        json_file.write_text("{}")

        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(str(json_file))

        assert len(result) == 1
        mock_load_from_json.assert_called_once_with(json_file)

    # ------------------------------------------------------------------
    # Directory
    # ------------------------------------------------------------------

    def test_loads_all_json_files_from_directory(self, tmp_path, mock_load_from_json):
        """All ``.json`` files in a directory should be loaded, sorted by filename."""
        (tmp_path / "b_doc.json").write_text("{}")
        (tmp_path / "a_doc.json").write_text("{}")
        (tmp_path / "c_doc.json").write_text("{}")

        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path)

        assert len(result) == 3
        # Verify sorted order: a_doc, b_doc, c_doc.
        call_paths = [call.args[0].name for call in mock_load_from_json.call_args_list]
        assert call_paths == ["a_doc.json", "b_doc.json", "c_doc.json"]

    def test_ignores_non_json_files_in_directory(self, tmp_path, mock_load_from_json):
        """Only files with a ``.json`` extension should be loaded."""
        (tmp_path / "doc.json").write_text("{}")
        (tmp_path / "readme.md").write_text("# Readme")
        (tmp_path / "data.txt").write_text("data")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")

        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path)

        assert len(result) == 1
        mock_load_from_json.assert_called_once()

    def test_ignores_subdirectories_with_json_suffix(self, tmp_path, mock_load_from_json):
        """A subdirectory named ``something.json`` must not be treated as a file."""
        subdir = tmp_path / "fake.json"
        subdir.mkdir()

        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path)

        assert len(result) == 0
        mock_load_from_json.assert_not_called()

    def test_case_insensitive_json_extension(self, tmp_path, mock_load_from_json):
        """Extensions like ``.JSON`` or ``.Json`` should be matched."""
        (tmp_path / "upper.JSON").write_text("{}")
        (tmp_path / "mixed.Json").write_text("{}")

        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path)

        assert len(result) == 2

    # ------------------------------------------------------------------
    # Empty directory
    # ------------------------------------------------------------------

    def test_returns_empty_list_for_empty_directory(self, tmp_path, mock_load_from_json):
        """An empty directory should produce an empty list without errors."""
        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path)

        assert result == []
        mock_load_from_json.assert_not_called()

    # ------------------------------------------------------------------
    # Non-existent path
    # ------------------------------------------------------------------

    def test_returns_empty_list_for_nonexistent_path(self, tmp_path, mock_load_from_json):
        """A path that does not exist should return an empty list (no exception)."""
        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path / "does_not_exist")

        assert result == []
        mock_load_from_json.assert_not_called()

    def test_returns_empty_list_for_nonexistent_file(self, tmp_path, mock_load_from_json):
        """A specific non-existent file path should return an empty list."""
        from ai4rag.components.utils.docling_io import load_docling_documents

        result = load_docling_documents(tmp_path / "missing.json")

        assert result == []
        mock_load_from_json.assert_not_called()
