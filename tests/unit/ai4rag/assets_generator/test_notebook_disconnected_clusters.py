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
    notebook_path = (
        Path(__file__).parents[4] / "ai4rag/assets_generator/notebook_templates/maas_indexing_template.ipynb"
    )
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

        assert any(
            "Prerequisites for Disconnected Clusters" in s for s in sources
        ), "Notebook missing 'Prerequisites for Disconnected Clusters' section"

    def test_notebook_explains_model_requirements(self, indexing_notebook):
        """Notebook must explain which models are required offline."""
        sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        full_text = "\n".join(sources)

        assert "Docling" in full_text, "Notebook must mention Docling artifacts"
        assert "DOCLING_ARTIFACTS_PATH" in full_text, "Notebook must mention DOCLING_ARTIFACTS_PATH env var"

    def test_notebook_includes_environment_setup_code(self, indexing_notebook):
        """Notebook must include code cells to set environment variables."""
        code_sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        full_code = "\n".join(code_sources)

        assert "DOCLING_ARTIFACTS_PATH" in full_code, "Notebook must include code to configure DOCLING_ARTIFACTS_PATH"

    def test_notebook_selects_artifacts_after_data_discovery(self, indexing_notebook):
        """Only the models required by the discovered corpus are validated."""
        code_sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        full_code = "\n".join(code_sources)

        assert full_code.index("result = discover_documents") < full_code.index("needs_ocr =")
        assert 'ocr_extensions = {{".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}}' in full_code
        assert 'audio_extensions = {{".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}}' in full_code
        assert "PDF or image data requires DOCLING_ARTIFACTS_PATH" in full_code
        assert "Audio data requires HF_MODEL_DIR" in full_code
        assert "model_config = json.load(config_file)" in full_code
        assert 'model_type != "whisper"' in full_code
        assert "Audio data supports only Hugging Face Whisper models" in full_code
        assert "Use a workbench image with the Docling artifacts bundle" in full_code

    def test_offline_docling_validation_is_a_dedicated_cell(self, indexing_notebook):
        """Connected runners can skip only offline Docling validation."""
        cells = indexing_notebook["cells"]
        validation_heading_index = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown"
            and "### Validate Offline Configuration" in "".join(cell["source"])
        )
        validation_cell = cells[validation_heading_index + 1]
        validation_source = "".join(validation_cell["source"])

        assert validation_cell["cell_type"] == "code"
        assert "PDF or image data requires DOCLING_ARTIFACTS_PATH" in validation_source
        assert "RapidOCR models are missing" in validation_source

        shared_cell = cells[validation_heading_index - 1]
        shared_source = "".join(shared_cell["source"])
        assert shared_cell["cell_type"] == "code"
        assert "needs_ocr =" in shared_source
        assert "DoclingExtractionConfig" in shared_source
        assert "RapidOCR models are missing" not in shared_source

    def test_extract_text_includes_docling_artifacts_path(self, indexing_notebook):
        """The extract_text() call must include docling_artifacts_path parameter."""
        code_sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code"
        ]

        extract_text_calls = [s for s in code_sources if "extract_text(" in s]
        assert len(extract_text_calls) > 0, "Notebook must include an extract_text() call"

        assert any(
            "docling_artifacts_path" in call for call in extract_text_calls
        ), "extract_text() call must include docling_artifacts_path parameter"
        assert any("docling_config=docling_config" in call for call in extract_text_calls)

    def test_notebook_includes_appendix_with_download_instructions(self, indexing_notebook):
        """Notebook must include appendix with offline download instructions."""
        sources = [
            "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        full_text = "\n".join(sources)

        assert "Appendix" in full_text, "Notebook must include an appendix section"
        assert "Download" in full_text, "Appendix must include download instructions"

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
