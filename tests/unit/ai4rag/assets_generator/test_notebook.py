# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai4rag.components.assets_generator import Notebook, NotebookCell

# ---------------------------------------------------------------------------
# NotebookCell
# ---------------------------------------------------------------------------


class TestNotebookCellToDict:
    """Verify that ``NotebookCell.to_dict()`` produces the correct Jupyter cell JSON structure."""

    def test_code_cell_includes_execution_fields(self):
        """Code cells must carry ``execution_count`` and ``outputs`` keys."""
        cell = NotebookCell(cell_type="code", source="print('hi')")
        result = cell.to_dict()

        assert result["cell_type"] == "code"
        assert result["source"] == "print('hi')"
        assert result["metadata"] == {}
        assert result["execution_count"] is None
        assert result["outputs"] == []

    def test_markdown_cell_omits_execution_fields(self):
        """Markdown cells must not contain ``execution_count`` or ``outputs``."""
        cell = NotebookCell(cell_type="markdown", source="# Title")
        result = cell.to_dict()

        assert result["cell_type"] == "markdown"
        assert result["source"] == "# Title"
        assert "execution_count" not in result
        assert "outputs" not in result

    def test_custom_metadata_preserved(self):
        """Explicit metadata passed at construction must survive round-trip."""
        meta = {"tags": ["setup"]}
        cell = NotebookCell(cell_type="code", source="x = 1", metadata=meta)

        assert cell.to_dict()["metadata"] == {"tags": ["setup"]}

    def test_default_metadata_is_empty_dict(self):
        """When no metadata is supplied, the default must be an empty dict (not None)."""
        cell = NotebookCell(cell_type="markdown", source="")
        assert cell.to_dict()["metadata"] == {}


class TestNotebookCellFormatSource:
    """Verify placeholder substitution in ``NotebookCell.format_source()``."""

    def test_list_source_substitution(self):
        """Placeholders in a list-of-lines source are replaced correctly."""
        cell = NotebookCell(cell_type="code", source=["x = {VALUE}", "y = {OTHER}"])
        result = cell.format_source({"VALUE": "42", "OTHER": "99"})

        assert result is cell, "format_source must return self for chaining"
        assert cell.source == ["x = 42", "y = 99"]

    def test_list_source_missing_placeholder_replaced_with_empty(self):
        """Missing placeholders must be replaced with empty strings, never raise."""
        cell = NotebookCell(cell_type="code", source=["val = {MISSING}"])
        cell.format_source({})

        assert cell.source == ["val = "]

    def test_string_source_substitution(self):
        """A plain string source is formatted in place."""
        cell = NotebookCell(cell_type="markdown", source="Hello {NAME}!")
        cell.format_source({"NAME": "World"})

        assert cell.source == "Hello World!"

    def test_no_placeholders_is_noop(self):
        """Source without any placeholders is returned unchanged."""
        cell = NotebookCell(cell_type="code", source=["import os"])
        cell.format_source({"UNUSED": "value"})

        assert cell.source == ["import os"]

    def test_partial_placeholders_with_list_source(self):
        """Only the placeholders present in the mapping are substituted; others become empty."""
        cell = NotebookCell(cell_type="code", source=["{A} + {B} = {C}"])
        cell.format_source({"A": "1", "C": "3"})

        assert cell.source == ["1 +  = 3"]


# ---------------------------------------------------------------------------
# Notebook
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_notebook() -> Notebook:
    """A minimal notebook with one code and one markdown cell."""
    return Notebook(
        cells=[
            NotebookCell(cell_type="markdown", source="# Header"),
            NotebookCell(cell_type="code", source="x = 1"),
        ]
    )


class TestNotebookToDict:
    """Verify ``Notebook.to_dict()`` produces a valid Jupyter notebook dict."""

    def test_structure_has_required_keys(self, simple_notebook: Notebook):
        """Top-level dict must contain cells, metadata, nbformat, and nbformat_minor."""
        result = simple_notebook.to_dict()

        assert set(result.keys()) == {"cells", "metadata", "nbformat", "nbformat_minor"}
        assert result["nbformat"] == 4
        assert result["nbformat_minor"] == 4
        assert len(result["cells"]) == 2

    def test_metadata_contains_kernelspec(self, simple_notebook: Notebook):
        """Default metadata must include kernelspec and language_info."""
        meta = simple_notebook.to_dict()["metadata"]

        assert "kernelspec" in meta
        assert meta["kernelspec"]["name"] == "python3"
        assert "language_info" in meta
        assert meta["language_info"]["name"] == "python"

    def test_empty_notebook(self):
        """An empty notebook (no cells) produces a valid dict with an empty cells list."""
        nb = Notebook()
        result = nb.to_dict()

        assert result["cells"] == []


class TestNotebookSave:
    """Verify ``Notebook.save()`` writes well-formed JSON to disk."""

    def test_save_creates_valid_json(self, tmp_path: Path, simple_notebook: Notebook):
        """Saved file must be loadable as JSON and match ``to_dict()``."""
        out = tmp_path / "test.ipynb"
        returned = simple_notebook.save(out)

        assert returned is simple_notebook, "save() must return self for chaining"
        assert out.exists()

        with out.open() as f:
            data = json.load(f)

        assert data == simple_notebook.to_dict()

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """Saving to a nested path that does not exist yet must create intermediates."""
        out = tmp_path / "deep" / "nested" / "notebook.ipynb"
        Notebook().save(out)

        assert out.exists()


class TestNotebookLoadBundled:
    """Verify ``Notebook.load()`` from the bundled package templates."""

    def test_load_ogx_indexing_template(self):
        """Loading the real bundled template must produce a Notebook with cells."""
        nb = Notebook.load("ogx_indexing_template.ipynb")

        assert isinstance(nb, Notebook)
        assert len(nb.cells) > 0
        assert all(isinstance(c, NotebookCell) for c in nb.cells)

    def test_loaded_template_preserves_cell_types(self):
        """Each loaded cell must have a valid cell_type of 'code' or 'markdown'."""
        nb = Notebook.load("ogx_indexing_template.ipynb")

        for cell in nb.cells:
            assert cell.cell_type in ("code", "markdown")

    def test_loaded_code_cells_have_execution_fields(self):
        """Code cells loaded from a template must carry execution_count and outputs."""
        nb = Notebook.load("ogx_indexing_template.ipynb")
        code_cells = [c for c in nb.cells if c.cell_type == "code"]

        assert len(code_cells) > 0
        for cell in code_cells:
            assert hasattr(cell, "execution_count")
            assert hasattr(cell, "outputs")

    def test_round_trip_preserves_content(self, tmp_path: Path):
        """Load a bundled template, save it, reload it, and verify cells match."""
        original = Notebook.load("ogx_indexing_template.ipynb")
        path = tmp_path / "round_trip.ipynb"
        original.save(path)

        reloaded = Notebook.load("round_trip.ipynb", templates_dir=tmp_path)

        assert len(reloaded.cells) == len(original.cells)
        for orig_cell, new_cell in zip(original.cells, reloaded.cells):
            assert orig_cell.to_dict() == new_cell.to_dict()


class TestNotebookLoadCustomDir:
    """Verify ``Notebook.load()`` with a custom ``templates_dir``."""

    @pytest.fixture
    def custom_template(self, tmp_path: Path) -> Path:
        """Create a minimal valid notebook JSON in a temp directory."""
        nb_data = {
            "cells": [
                {
                    "cell_type": "code",
                    "source": ["print('custom')"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                }
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.12.0"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        path = tmp_path / "custom_template.ipynb"
        with path.open("w") as f:
            json.dump(nb_data, f)
        return tmp_path

    def test_load_from_custom_dir(self, custom_template: Path):
        """Loading from a custom directory must use that directory, not bundled data."""
        nb = Notebook.load("custom_template.ipynb", templates_dir=custom_template)

        assert len(nb.cells) == 1
        assert nb.cells[0].source == ["print('custom')"]
        assert nb.nbformat_minor == 5

    def test_load_preserves_original_metadata(self, custom_template: Path):
        """Metadata from the loaded file must override constructor defaults."""
        nb = Notebook.load("custom_template.ipynb", templates_dir=custom_template)

        assert nb.metadata["language_info"]["version"] == "3.12.0"
