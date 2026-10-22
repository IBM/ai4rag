# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Checks for the simple MaaS indexing notebook flow."""

import json
import os
import string
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[4]


@pytest.fixture
def indexing_notebook() -> dict:
    """Load the MaaS indexing notebook template."""
    path = ROOT / "ai4rag/assets_generator/notebook_templates/maas_indexing_template.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def _source(cell: dict) -> str:
    """Return a notebook cell's text."""
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def test_maas_notebook_uses_one_mode_setting_and_no_tags(indexing_notebook, monkeypatch):
    """Run All should follow one explicit mode, without tag filtering."""
    cells = indexing_notebook["cells"]
    assert all(not cell["metadata"].get("tags") for cell in cells)
    mode_cells = [cell for cell in cells if "cluster_mode = " in _source(cell)]
    assert len(mode_cells) == 1
    source = _source(mode_cells[0]).format()

    for mode in ("connected", "disconnected"):
        monkeypatch.delenv("NOTEBOOK_CLUSTER_MODE", raising=False)
        monkeypatch.delenv("RHOAI_NOTEBOOK_CLUSTER_MODE", raising=False)
        monkeypatch.setenv("RHOAI_NOTEBOOK_CLUSTER_MODE", mode)
        namespace = {"os": os}
        exec(source, namespace)
        assert namespace["cluster_mode"] == mode
        assert (os.getenv("HF_HUB_OFFLINE") == "1") is (mode == "disconnected")


def test_notebook_prepares_extraction_after_discovery(indexing_notebook):
    """Model details stay in ai4rag, not in notebook cells."""
    code = "\n".join(_source(cell) for cell in indexing_notebook["cells"] if cell["cell_type"] == "code")
    assert code.index("result = discover_documents(") < code.index("extraction = prepare_extraction(")
    assert "docling_config=extraction.config" in code
    assert "docling_artifacts_path=extraction.artifacts_path" in code
    assert "hf_hub_download" not in code
    assert "rapidocr_bundles" not in code
    assert "ocr_language =" not in code
    assert "inspect.signature" not in code


def test_disconnected_instructions_are_in_one_early_cell(indexing_notebook):
    """Offline prerequisites appear before document discovery."""
    cells = indexing_notebook["cells"]
    instruction_cells = [cell for cell in cells if "### Disconnected cluster setup" in _source(cell)]
    assert len(instruction_cells) == 1
    instructions = _source(instruction_cells[0])
    assert "DOCLING_ARTIFACTS_PATH" in instructions
    assert "HF_MODEL_DIR" in instructions
    assert "package mirror" in instructions
    assert cells.index(instruction_cells[0]) < next(
        index for index, cell in enumerate(cells) if "result = discover_documents(" in _source(cell)
    )


def test_text_extraction_extra_has_ocr_dependencies(indexing_notebook):
    """The installed extra provides the notebook helper's dependencies."""
    code = "\n".join(_source(cell) for cell in indexing_notebook["cells"] if cell["cell_type"] == "code")
    assert "ai4rag[text-extraction]" in code
    project_config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project_config["project"]["optional-dependencies"]["text-extraction"]
    assert "rapidocr~=3.9.2" in dependencies
    assert "onnxruntime~=1.25.0" in dependencies
    assert "huggingface-hub>=0.30,<2" in dependencies


@pytest.mark.parametrize("template", ["maas_indexing", "mass_creating_knowledge_graph"])
def test_generated_code_cells_render_and_compile(template):
    """Template replacement leaves ordinary Python cells syntactically valid."""
    path = ROOT / f"ai4rag/assets_generator/notebook_templates/{template}_template.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    formatter = string.Formatter()
    fields = {field for cell in notebook["cells"] for _, field, _, _ in formatter.parse(_source(cell)) if field}
    values = {field: "1" for field in fields}
    values.update(
        LANGUAGE='{"code": "en"}',
        INPUT_DATA_KEYS='["docs/"]',
        EMBEDDING_PARAMS="{}",
        AI4RAG_VERSION="0.24.0",
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        rendered = _source(cell).format(**values)
        if not rendered.startswith("%"):
            compile(rendered, "<generated notebook cell>", "exec")


def test_graph_notebook_uses_shared_extraction_and_one_mode_setting(monkeypatch):
    """The graph notebook follows the same connected and offline flow."""
    path = ROOT / "ai4rag/assets_generator/notebook_templates/mass_creating_knowledge_graph_template.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    assert all(not cell["metadata"].get("tags") for cell in cells)
    code = "\n".join(_source(cell) for cell in cells if cell["cell_type"] == "code")
    assert code.count("cluster_mode = ") == 1
    assert code.index("result = discover_documents(") < code.index("extraction = prepare_extraction(")
    assert "docling_config=extraction.config" in code
    assert "docling_artifacts_path=extraction.artifacts_path" in code
    assert "hf_hub_download" not in code
    assert "rapidocr_bundles" not in code
    assert "ocr_language =" not in code
    assert "max_extraction_workers=1" in code
    mode_cell = next(cell for cell in cells if "cluster_mode = " in _source(cell))
    for mode in ("connected", "disconnected"):
        monkeypatch.setenv("NOTEBOOK_CLUSTER_MODE", mode)
        namespace = {"os": os}
        exec(_source(mode_cell).format(), namespace)
        assert namespace["cluster_mode"] == mode
