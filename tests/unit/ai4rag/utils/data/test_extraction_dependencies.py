# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Dependency contract for standalone extraction and generated notebooks."""

import tomllib
from pathlib import Path


def test_text_extraction_extra_has_ocr_dependencies():
    """The extraction extra supplies the public helper's OCR dependencies."""
    project = tomllib.loads((Path(__file__).parents[5] / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["optional-dependencies"]["text-extraction"]
    assert "rapidocr~=3.9.2" in dependencies
    assert "onnxruntime~=1.25.0" in dependencies
    assert "huggingface-hub>=0.30,<2" in dependencies
