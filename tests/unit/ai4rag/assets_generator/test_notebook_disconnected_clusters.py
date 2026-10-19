# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Test that indexing notebook includes disconnected cluster documentation."""

import json
import tomllib
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

    def test_notebook_installs_compatible_text_extraction_dependencies(self, indexing_notebook):
        """The text-extraction extra carries compatible OCR dependency constraints for the notebook."""
        install_cell = next(
            "".join(cell["source"])
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code" and "ai4rag[text-extraction]" in "".join(cell["source"])
        )

        assert "ai4rag[text-extraction]" in install_cell

        pyproject_path = Path(__file__).parents[4] / "pyproject.toml"
        project_config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        text_extraction_dependencies = project_config["project"]["optional-dependencies"]["text-extraction"]
        assert "rapidocr~=3.9.2" in text_extraction_dependencies
        assert "onnxruntime~=1.25.0" in text_extraction_dependencies

    def test_notebook_has_disconnected_cluster_prerequisites_section(self, indexing_notebook):
        """Notebook must include a 'Prerequisites for Disconnected Clusters' section."""
        sources = [
            ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]

        assert any(
            "Prerequisites for Disconnected Clusters" in s for s in sources
        ), "Notebook missing 'Prerequisites for Disconnected Clusters' section"

    def test_notebook_explains_model_requirements(self, indexing_notebook):
        """Notebook must explain which models are required offline."""
        sources = [
            ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "markdown"
        ]
        full_text = "\n".join(sources)

        assert "Docling" in full_text, "Notebook must mention Docling artifacts"
        assert "DOCLING_ARTIFACTS_PATH" in full_text, "Notebook must mention DOCLING_ARTIFACTS_PATH env var"

    def test_notebook_includes_environment_setup_code(self, indexing_notebook):
        """Notebook must include code cells to set environment variables."""
        code_sources = [
            ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        full_code = "\n".join(code_sources)

        assert "DOCLING_ARTIFACTS_PATH" in full_code, "Notebook must include code to configure DOCLING_ARTIFACTS_PATH"

    def test_notebook_selects_artifacts_after_data_discovery(self, indexing_notebook):
        """Only the models required by the discovered corpus are validated."""
        code_sources = [
            ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
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

    def test_model_validation_is_shared_and_only_offline_setup_is_tagged(self, indexing_notebook):
        """Both environments validate the pinned models; only offline flags are skipped."""
        cells = indexing_notebook["cells"]
        offline_cells = [cell for cell in cells if "disconnected" in cell["metadata"].get("tags", [])]
        assert len(offline_cells) == 1
        assert 'os.environ["HF_HUB_OFFLINE"] = "1"' in "".join(offline_cells[0]["source"])
        validation_cell = next(
            cell
            for cell in cells
            if cell["cell_type"] == "code" and "docling_config = DoclingExtractionConfig(" in "".join(cell["source"])
        )
        validation_source = "".join(validation_cell["source"])

        assert validation_cell["cell_type"] == "code"
        assert "PDF or image data requires DOCLING_ARTIFACTS_PATH" in validation_source
        assert "models are missing" in validation_source
        assert not {"connected", "disconnected"}.intersection(validation_cell["metadata"].get("tags", []))

        shared_cell = next(
            cell for cell in cells if cell["cell_type"] == "code" and "needs_ocr =" in "".join(cell["source"])
        )
        shared_source = "".join(shared_cell["source"])
        assert shared_cell["cell_type"] == "code"
        assert "needs_ocr =" in shared_source
        assert "ocr_language =" in shared_source
        assert not {"connected", "disconnected"}.intersection(shared_cell["metadata"].get("tags", []))

    def test_extract_text_includes_docling_artifacts_path(self, indexing_notebook):
        """The extract_text() call must include docling_artifacts_path parameter."""
        code_sources = [
            ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
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
            ("".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
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
            assert cell["cell_type"] in (
                "code",
                "markdown",
            ), f"Cell {i} has invalid type: {cell['cell_type']}"
            assert "source" in cell, f"Cell {i} missing source"
            assert "metadata" in cell, f"Cell {i} missing metadata"

            if cell["cell_type"] == "code":
                assert "execution_count" in cell, f"Code cell {i} missing execution_count"
                assert "outputs" in cell, f"Code cell {i} missing outputs"


def test_graph_notebook_tags_only_disconnected_setup():
    """Shared model validation runs in both environments after document discovery."""
    notebook_path = (
        Path(__file__).parents[4]
        / "ai4rag/assets_generator/notebook_templates/mass_creating_knowledge_graph_template.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    expected_offline_sources = (
        "%pip install --force-reinstall",
        'os.environ["HF_HUB_OFFLINE"] = "1"',
    )
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        environment_tags = {"connected", "disconnected"}.intersection(cell["metadata"].get("tags", []))
        if cell["cell_type"] == "code" and any(fragment in source for fragment in expected_offline_sources):
            assert environment_tags == {"disconnected"}
        else:
            assert not environment_tags
    for fragment in expected_offline_sources:
        assert any(fragment in "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
