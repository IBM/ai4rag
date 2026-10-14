# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Test that indexing notebook includes disconnected cluster documentation."""

import json
import os
import tomllib
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

    def test_notebook_installs_pinned_text_extraction_dependencies(self, indexing_notebook):
        """The text-extraction extra carries the OCR dependency pins used by the notebook."""
        install_cell = next(
            "".join(cell["source"])
            for cell in indexing_notebook["cells"]
            if cell["cell_type"] == "code" and "ai4rag[text-extraction]" in "".join(cell["source"])
        )

        assert "ai4rag[text-extraction]" in install_cell

        pyproject_path = Path(__file__).parents[4] / "pyproject.toml"
        project_config = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
        text_extraction_dependencies = project_config["project"]["optional-dependencies"]["text-extraction"]
        assert "rapidocr==3.9.2" in text_extraction_dependencies
        assert "onnxruntime==1.25.0" in text_extraction_dependencies

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
        assert "ocr_lang=ocr_language" in full_code
        assert all(name in full_code for name in ("ocr_det_model_path", "ocr_cls_model_path"))
        assert all(name in full_code for name in ("ocr_rec_model_path", "ocr_rec_keys_path"))

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
        assert "RapidOCR models are missing" in validation_source

        shared_cell = cells[validation_heading_index - 1]
        shared_source = "".join(shared_cell["source"])
        assert shared_cell["cell_type"] == "code"
        assert "needs_ocr =" in shared_source
        assert "DoclingExtractionConfig" in shared_source
        assert "RapidOCR models are missing" not in shared_source

    def test_generated_notebook_uses_conditional_ocr_and_matching_pp_ocrv4_paths(self, tmp_path, monkeypatch):
        """Text-only input skips RapidOCR; PDF/image input uses the selected pinned model bundle."""
        from ai4rag.assets_generator.notebook import Notebook
        from ai4rag.assets_generator.templates import create_placeholder_mapping

        bundle_paths = {
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
        notebook_path = (
            Path(__file__).parents[4] / "ai4rag/assets_generator/notebook_templates/maas_indexing_template.ipynb"
        )

        def formatted_cells(language: dict) -> list[dict]:
            notebook = Notebook.load(notebook_path.name, templates_dir=notebook_path.parent)
            mapping = create_placeholder_mapping(
                {"template_id": "simple_rag", "settings": {"generation": {"language": language}}}
            )
            for cell in notebook.cells:
                cell.format_source(mapping)
            return [cell.to_dict() for cell in notebook.cells]

        def cell_source(cell: dict) -> str:
            source = cell["source"]
            return "".join(source) if isinstance(source, list) else source

        def run_extraction_setup(cells: list[dict], documents: list[dict]) -> dict:
            namespace = {
                "Path": Path,
                "os": os,
                "json": json,
                "result": SimpleNamespace(to_dict=lambda: {"documents": documents}),
                "DoclingExtractionConfig": lambda **kwargs: kwargs,
            }
            discovery_source = next(
                cell_source(cell) for cell in cells if "discovered_documents = result.to_dict()" in cell_source(cell)
            )
            validation_source = next(
                cell_source(cell) for cell in cells if "RapidOCR models are missing below" in cell_source(cell)
            )
            exec(discovery_source, namespace)  # noqa: S102 - execute generated notebook setup under test
            exec(validation_source, namespace)  # noqa: S102 - execute generated notebook validation under test
            return namespace

        monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
        text_only = run_extraction_setup(
            formatted_cells({"code": "en", "name": "English"}), [{"key": "docs/readme.txt"}]
        )
        assert text_only["needs_ocr"] is False
        assert text_only["docling_config"]["do_ocr"] is False
        assert text_only["rapidocr_model_paths"] == {}

        artifacts = tmp_path / "docling"
        for paths in bundle_paths.values():
            for relative_path in paths.values():
                model_path = artifacts / "RapidOcr" / relative_path
                model_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.touch()
        monkeypatch.setenv("DOCLING_ARTIFACTS_PATH", str(artifacts))

        chinese_pdf = run_extraction_setup(
            formatted_cells({"code": "zh", "name": "Chinese"}), [{"key": "docs/scanned.pdf"}]
        )
        assert chinese_pdf["needs_ocr"] is True
        assert chinese_pdf["docling_config"]["do_ocr"] is True
        assert chinese_pdf["docling_config"]["ocr_lang"] == "chinese"
        expected_chinese_paths = {
            key: str(artifacts / "RapidOcr" / relative_path) for key, relative_path in bundle_paths["chinese"].items()
        }
        assert chinese_pdf["rapidocr_model_paths"] == expected_chinese_paths
        assert {key: chinese_pdf["docling_config"][key] for key in expected_chinese_paths} == expected_chinese_paths

        english_image = run_extraction_setup(
            formatted_cells({"code": "en", "name": "English"}), [{"key": "docs/photo.tiff"}]
        )
        assert english_image["needs_ocr"] is True
        assert english_image["docling_config"]["do_ocr"] is True
        assert english_image["docling_config"]["ocr_lang"] == "english"
        assert english_image["rapidocr_model_paths"] == {
            key: str(artifacts / "RapidOcr" / relative_path) for key, relative_path in bundle_paths["english"].items()
        }

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
