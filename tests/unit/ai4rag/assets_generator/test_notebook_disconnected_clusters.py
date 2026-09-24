# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Test that indexing notebook includes disconnected cluster documentation."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def indexing_notebook() -> dict:
    """Load the MaaS indexing template notebook."""
    notebook_path = Path(__file__).parents[4] / "ai4rag/assets_generator/notebook_templates/maas_indexing_template.ipynb"
    with open(notebook_path) as f:
        return json.load(f)


class TestDisconnectedClusterDocumentation:
    """Verify disconnected cluster support is documented in indexing notebook."""

    def test_notebook_has_disconnected_cluster_prerequisites_section(self, indexing_notebook):
        """Notebook must include a 'Prerequisites for Disconnected Clusters' section."""
        sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]

        assert any("Prerequisites for Disconnected Clusters" in s for s in sources), \
            "Notebook missing 'Prerequisites for Disconnected Clusters' section"

    def test_notebook_explains_model_requirements(self, indexing_notebook):
        """Notebook must describe format-specific offline model requirements."""
        sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        full_text = "\n".join(sources)

        assert "Docling" in full_text, "Notebook must mention Docling artifacts"
        assert "DOCLING_ARTIFACTS_PATH" in full_text, "Notebook must mention DOCLING_ARTIFACTS_PATH env var"
        assert "TXT-only corpora" in full_text, "Notebook must explain TXT-only offline requirements"
        assert "Markdown-only corpora" in full_text, "Notebook must explain Markdown-only offline requirements"
        assert "no Docling ML artifacts are required" in full_text, (
            "Notebook must state that TXT-only and Markdown-only corpora do not need Docling ML artifacts"
        )

    def test_notebook_includes_environment_setup_code(self, indexing_notebook):
        """Notebook must include code cells to set environment variables."""
        code_sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        full_code = "\n".join(code_sources)

        assert "DOCLING_ARTIFACTS_PATH" in full_code, "Notebook must include code to configure DOCLING_ARTIFACTS_PATH"

    def test_extract_text_includes_docling_artifacts_path(self, indexing_notebook):
        """The extract_text() call must include docling_artifacts_path parameter."""
        code_sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code"
        ]

        extract_text_calls = [s for s in code_sources if "extract_text(" in s]
        assert len(extract_text_calls) > 0, "Notebook must include an extract_text() call"

        assert any("docling_artifacts_path" in call for call in extract_text_calls), \
            "extract_text() call must include docling_artifacts_path parameter"

    def test_notebook_includes_appendix_with_download_instructions(self, indexing_notebook):
        """Notebook must include model-backed offline download instructions only."""
        sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        full_text = "\n".join(sources)

        assert "Appendix" in full_text, "Notebook must include an appendix section"
        assert "Model-backed formats or features" in full_text
        assert "Do not use a `.txt` file as an artifact-validation trigger" in full_text
        assert "docling-tools models download" in full_text
        assert "oc rsync ~/.cache/docling/models/" in full_text
        assert "DOCLING_ARTIFACTS_PATH=/opt/app-root/docling-artifacts" in full_text
        assert 'suffix=".txt"' not in full_text
        assert "download all Docling artifacts" not in full_text
        assert "rsync -av" not in full_text

    def test_notebook_cells_are_valid_json(self, indexing_notebook):
        """All cells must be valid JSON with required fields."""
        for i, cell in enumerate(indexing_notebook["cells"]):
            assert "cell_type" in cell, f"Cell {i} missing cell_type"
            assert cell["cell_type"] in ("code", "markdown"), f"Cell {i} has invalid type: {cell['cell_type']}"
            assert "source" in cell, f"Cell {i} missing source"
            assert "metadata" in cell, f"Cell {i} missing metadata"

            if cell["cell_type"] == "code":
                assert "execution_count" in cell, f"Code cell {i} missing execution_count"
                assert "outputs" in cell, f"Code cell {i} missing outputs"
