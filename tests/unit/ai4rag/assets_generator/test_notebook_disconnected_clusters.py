# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Test that indexing notebook includes disconnected cluster documentation."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

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

    def test_notebook_installs_pinned_ocr_dependencies_after_ai4rag(self, indexing_notebook):
        """OCR notebooks need the same RapidOCR and ONNX Runtime pins as the pipeline image."""
        install_cell = next(
            "".join(cell["source"])
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code" and "ai4rag[text-extraction]" in "".join(cell["source"])
        )

        assert "rapidocr==3.9.2" in install_cell
        assert "--force-reinstall --no-deps" in install_cell
        assert "onnxruntime==1.25.0" in install_cell
        assert install_cell.index("ai4rag[text-extraction]") < install_cell.index("rapidocr==3.9.2")
        assert install_cell.index("rapidocr==3.9.2") < install_cell.index("onnxruntime==1.25.0")

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
        assert "matching PP-OCRv4 Docling artifacts bundle" in full_code
        assert "rapidocr_model_paths" in full_code
        assert "ocr_rec_keys_path" in full_code
        assert "do_ocr=needs_ocr" in full_code
        assert "ocr_lang=ocr_language" in full_code

    def test_offline_docling_validation_is_a_dedicated_cell(self, indexing_notebook):
        """Connected runners can skip only offline Docling validation."""
        cells = indexing_notebook["cells"]
        validation_heading_index = next(
            index
            for index, cell in enumerate(cells)
            if cell["cell_type"] == "markdown" and "### Validate Offline Configuration" in "".join(cell["source"])
        )
        validation_cell = cells[validation_heading_index + 1]
        validation_source = "".join(validation_cell["source"])

        assert validation_cell["cell_type"] == "code"
        assert "PDF or image data requires DOCLING_ARTIFACTS_PATH" in validation_source
        assert "RapidOCR {{ocr_language}} models are missing" in validation_source

        shared_cell = cells[validation_heading_index - 1]
        shared_source = "".join(shared_cell["source"])
        assert shared_cell["cell_type"] == "code"
        assert "needs_ocr =" in shared_source
        assert "generation_language = {LANGUAGE}" in shared_source
        assert "RapidOCR models are missing" not in shared_source
        assert "DoclingExtractionConfig" in validation_source

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

    def test_generated_notebook_uses_conditional_ocr_and_pipeline_model_paths(self, tmp_path, monkeypatch):
        """Text-only input skips artifacts; OCR input pins the selected pipeline PP-OCRv4 bundle."""
        from ai4rag.assets_generator.notebook import Notebook
        from ai4rag.assets_generator.templates import create_placeholder_mapping

        artifacts = tmp_path / "docling"
        model_paths = {
            "english": {
                "ocr_det_model_path": "onnx/PP-OCRv4/det/en_PP-OCRv3_det_mobile.onnx",
                "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                "ocr_rec_model_path": "onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx",
                "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile/en_dict.txt",
            },
            "chinese": {
                "ocr_det_model_path": "onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx",
                "ocr_cls_model_path": "onnx/PP-OCRv4/cls/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
                "ocr_rec_model_path": "onnx/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile.onnx",
                "ocr_rec_keys_path": "paddle/PP-OCRv4/rec/ch_PP-OCRv4_rec_mobile/ppocr_keys_v1.txt",
            },
        }

        def formatted_cells(language: dict) -> list[dict]:
            notebook = Notebook.load("maas_indexing_template.ipynb")
            mapping = create_placeholder_mapping(
                {"template_id": "simple_rag", "settings": {"generation": {"language": language}}}
            )
            for cell in notebook.cells:
                cell.format_source(mapping)
            return [cell.to_dict() for cell in notebook.cells]

        def execute_discovery(cells: list[dict], documents: list[dict]) -> dict:
            source = "".join(cells[18]["source"])
            namespace = {
                "Path": Path,
                "os": os,
                "json": json,
                "result": SimpleNamespace(to_dict=lambda: {"documents": documents}),
            }
            exec(source, namespace)  # noqa: S102 - execute the generated notebook cell under test
            return namespace

        english_cells = formatted_cells({"code": "en", "name": "English"})
        monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
        text_only = execute_discovery(english_cells, [{"key": "docs/readme.txt"}])
        assert text_only["needs_ocr"] is False
        assert text_only["docling_artifacts_path"] is None
        assert text_only["ocr_language"] == "english"

        text_only["DoclingExtractionConfig"] = lambda **kwargs: kwargs
        exec("".join(english_cells[20]["source"]), text_only)  # noqa: S102 - execute generated config cell
        assert text_only["docling_config"]["do_ocr"] is False
        assert text_only["rapidocr_model_paths"] == {}

        default_language_cells = formatted_cells({})
        default_language = execute_discovery(default_language_cells, [{"key": "docs/readme.txt"}])
        assert default_language["ocr_language"] == "english"

        for bundle in model_paths.values():
            for relative_path in bundle.values():
                path = artifacts / "RapidOcr" / relative_path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
        monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(artifacts))
        chinese_cells = formatted_cells({"code": "zh", "name": "Chinese"})
        pdf = execute_discovery(chinese_cells, [{"key": "docs/scan.pdf"}])
        assert pdf["needs_ocr"] is True
        assert pdf["ocr_language"] == "chinese"
        pdf["DoclingExtractionConfig"] = lambda **kwargs: kwargs
        exec("".join(chinese_cells[20]["source"]), pdf)  # noqa: S102 - execute generated config cell

        config = pdf["docling_config"]
        assert config["do_ocr"] is True
        assert config["ocr_lang"] == "chinese"
        assert {
            key: str(artifacts / "RapidOcr" / relative_path) for key, relative_path in model_paths["chinese"].items()
        } == pdf["rapidocr_model_paths"]
        assert config["ocr_rec_keys_path"] == str(artifacts / "RapidOcr" / model_paths["chinese"]["ocr_rec_keys_path"])

        image = execute_discovery(english_cells, [{"key": "docs/photo.tiff"}])
        assert image["needs_ocr"] is True
        image["DoclingExtractionConfig"] = lambda **kwargs: kwargs
        exec("".join(english_cells[20]["source"]), image)  # noqa: S102 - execute generated config cell
        assert image["docling_config"]["do_ocr"] is True
        assert image["docling_config"]["ocr_lang"] == "english"
        assert {
            key: str(artifacts / "RapidOcr" / relative_path) for key, relative_path in model_paths["english"].items()
        } == image["rapidocr_model_paths"]

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
