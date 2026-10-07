# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for the offline local Hugging Face ASR adapter."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from ai4rag.utils.data.local_hf_asr_pipeline import (
    LocalHuggingFaceAsrPipeline,
    LocalHuggingFaceAsrPipelineOptions,
    _validate_model_directory,
)


def _model_directory(tmp_path):
    """Create the minimum validated local Transformers model layout."""
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    return tmp_path


def test_validate_model_directory_accepts_complete_local_model(tmp_path):
    """A complete local model directory is accepted without contacting a hub."""
    model_dir = _model_directory(tmp_path)

    assert _validate_model_directory(str(model_dir)) == str(model_dir.resolve())


def test_validate_model_directory_rejects_missing_path():
    """Audio extraction without a configured model path must fail closed."""
    with pytest.raises(FileNotFoundError, match="HF_MODEL_DIR"):
        _validate_model_directory(None)


def test_validate_model_directory_reports_missing_files(tmp_path):
    """An incomplete modelcar mount produces a useful error."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        _validate_model_directory(str(tmp_path))


def test_load_transcriber_uses_local_files_only(monkeypatch, tmp_path):
    """The adapter must pass local-only loading to both Transformers loaders."""
    model_dir = _model_directory(tmp_path)
    processor = SimpleNamespace(tokenizer=object(), feature_extractor=object())
    model = mock.MagicMock()
    transcriber = mock.MagicMock()
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=mock.MagicMock(return_value=processor)),
        AutoModelForSpeechSeq2Seq=SimpleNamespace(from_pretrained=mock.MagicMock(return_value=model)),
        pipeline=mock.MagicMock(return_value=transcriber),
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    adapter = LocalHuggingFaceAsrPipeline(LocalHuggingFaceAsrPipelineOptions(asr_model_path=str(model_dir)))

    assert adapter._load_transcriber() is transcriber  # pylint: disable=protected-access
    transformers.AutoProcessor.from_pretrained.assert_called_once_with(str(model_dir.resolve()), local_files_only=True)
    transformers.AutoModelForSpeechSeq2Seq.from_pretrained.assert_called_once_with(
        str(model_dir.resolve()), local_files_only=True
    )
    model.eval.assert_called_once_with()


def test_build_document_preserves_timestamped_transcript(tmp_path):
    """Transformers segments become Docling text items with timing metadata."""
    model_dir = _model_directory(tmp_path)
    adapter = LocalHuggingFaceAsrPipeline(LocalHuggingFaceAsrPipelineOptions(asr_model_path=str(model_dir)))
    adapter._transcriber = mock.MagicMock(  # pylint: disable=protected-access
        return_value={
            "chunks": [
                {"text": " First segment ", "timestamp": (1.0, 2.0)},
                {"text": "", "timestamp": (2.0, 3.0)},
                {"text": "No end", "timestamp": (3.0, None)},
            ]
        }
    )
    result = SimpleNamespace(input=SimpleNamespace(file=tmp_path / "recording.mp3", document_hash="0" * 64))

    adapter._build_document(result)  # pylint: disable=protected-access

    assert result.document.texts[0].text == "First segment"
    source = result.document.texts[0].source[0]
    assert (source.start_time, source.end_time) == (1.0, 2.0)
