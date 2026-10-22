# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Behavioral tests for notebook extraction model preparation."""

import ast
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from ai4rag.utils.data import text_extraction as extraction


def _documents(*keys: str) -> list[dict]:
    """Build discovery-style descriptors."""
    return [{"key": key, "size_bytes": 1} for key in keys]


@pytest.mark.parametrize("disconnected", [False, True])
def test_text_only_needs_no_models(monkeypatch, disconnected):
    """Text-only corpora must not require local bundles or downloads."""
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
    monkeypatch.delenv("HF_MODEL_DIR", raising=False)
    prepared = extraction.prepare_extraction(_documents("report.txt", "notes.md"), disconnected=disconnected)
    assert not prepared.config.do_ocr
    assert not prepared.needs_asr
    assert prepared.artifacts_path is None


@pytest.mark.parametrize("language,bundle", [("en", "english"), ("zh", "chinese")])
@pytest.mark.parametrize("extension", [".pdf", ".png"])
def test_local_ocr_uses_pinned_paths(tmp_path, language, bundle, extension):
    """PDFs and images use the same four PP-OCRv4 paths as the pipeline."""
    expected = extraction._RAPIDOCR_BUNDLES[bundle]
    for relative in expected.values():
        model_file = tmp_path / "RapidOcr" / relative
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()

    prepared = extraction.prepare_extraction(
        _documents(f"folder/document{extension}"),
        language=language,
        disconnected=True,
        docling_artifacts_path=str(tmp_path),
    )
    assert prepared.needs_ocr
    assert prepared.config.do_ocr
    assert prepared.config.ocr_lang == (bundle,)
    assert prepared.artifacts_path == str(tmp_path)
    for setting, relative in expected.items():
        assert getattr(prepared.config, setting) == str(tmp_path / "RapidOcr" / relative)


def test_disconnected_ocr_requires_bundle(monkeypatch):
    """An offline PDF fails before extraction when no bundle is mounted."""
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
    with pytest.raises(FileNotFoundError, match="DOCLING_ARTIFACTS_PATH"):
        extraction.prepare_extraction(_documents("scan.pdf"), disconnected=True)


def test_incomplete_bundle_lists_missing_files(tmp_path):
    """A partial local bundle is not silently replaced with other models."""
    with pytest.raises(FileNotFoundError, match="RapidOCR english models are missing"):
        extraction.prepare_extraction(_documents("scan.pdf"), docling_artifacts_path=str(tmp_path))


def test_connected_ocr_downloads_only_selected_bundle(tmp_path, monkeypatch):
    """Connected runs fetch the four selected files at one pinned revision."""
    monkeypatch.delenv("DOCLING_ARTIFACTS_PATH", raising=False)
    calls = []
    hub = ModuleType("huggingface_hub")

    def fake_download(**kwargs):
        calls.append(kwargs)
        file_path = tmp_path / kwargs["filename"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
        return str(file_path)

    hub.hf_hub_download = fake_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    prepared = extraction.prepare_extraction(_documents("scan.jpg"), language="zh")
    assert prepared.config.do_ocr
    assert prepared.artifacts_path is None
    assert {call["filename"] for call in calls} == {
        f"RapidOcr/{path}" for path in extraction._RAPIDOCR_BUNDLES["chinese"].values()
    }
    assert {call["revision"] for call in calls} == {extraction._OCR_ARTIFACT_REVISION}


@pytest.mark.parametrize("disconnected", [False, True])
def test_audio_reuses_local_whisper_validation(tmp_path, monkeypatch, disconnected):
    """Audio needs a complete local Whisper model, including in mixed corpora."""
    monkeypatch.delenv("HF_MODEL_DIR", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).touch()
    prepared = extraction.prepare_extraction(
        _documents("voice.mp3", "notes.txt"), disconnected=disconnected, asr_model_path=str(tmp_path)
    )
    assert prepared.needs_asr
    assert not prepared.needs_ocr
    assert prepared.config.asr_model_path == str(tmp_path)


def test_disconnected_audio_requires_local_whisper(monkeypatch):
    """Offline audio fails before extraction when no model is mounted."""
    monkeypatch.delenv("HF_MODEL_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="HF_MODEL_DIR"):
        extraction.prepare_extraction(_documents("voice.mp3"), disconnected=True)


def test_connected_audio_downloads_pinned_whisper(tmp_path, monkeypatch):
    """Connected audio downloads and validates the pinned Whisper snapshot."""
    monkeypatch.delenv("HF_MODEL_DIR", raising=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).touch()
    calls = []
    hub = ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    prepared = extraction.prepare_extraction(_documents("voice.wav"))
    assert prepared.needs_asr
    assert prepared.config.asr_model_path == str(tmp_path)
    assert calls == [
        {
            "repo_id": extraction._ASR_MODEL_REPO,
            "revision": extraction._ASR_MODEL_REVISION,
            "allow_patterns": ["*.json", "*.txt", "*.safetensors"],
        }
    ]


def test_connected_audio_prefers_env_model_without_download(tmp_path, monkeypatch):
    """An existing local model takes priority over a connected download."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (tmp_path / name).touch()
    monkeypatch.setenv("HF_MODEL_DIR", str(tmp_path))
    hub = ModuleType("huggingface_hub")

    def unexpected_download(**_kwargs):
        pytest.fail("Local Whisper models must not trigger a download")

    hub.snapshot_download = unexpected_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    prepared = extraction.prepare_extraction(_documents("voice.wav"))
    assert prepared.config.asr_model_path == str(tmp_path)


def test_mixed_corpus_requires_both_bundles(tmp_path):
    """A PDF plus audio enables OCR and validates the local Whisper model."""
    ocr_root = tmp_path / "docling"
    for relative in extraction._RAPIDOCR_BUNDLES["english"].values():
        model_file = ocr_root / "RapidOcr" / relative
        model_file.parent.mkdir(parents=True, exist_ok=True)
        model_file.touch()
    whisper_root = tmp_path / "whisper"
    whisper_root.mkdir()
    (whisper_root / "config.json").write_text(json.dumps({"model_type": "whisper"}), encoding="utf-8")
    for name in ("preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        (whisper_root / name).touch()

    prepared = extraction.prepare_extraction(
        _documents("scan.pdf", "voice.wav"),
        disconnected=True,
        docling_artifacts_path=str(ocr_root),
        asr_model_path=str(whisper_root),
    )
    assert prepared.config.do_ocr
    assert prepared.needs_asr
    assert prepared.config.asr_model_path == str(whisper_root)


def test_pipeline_bundle_paths_match_when_pipeline_checkout_is_available():
    """Compare the full path mapping with the sibling pipeline checkout, if present."""
    component = (
        Path(__file__).parents[6]
        / "pipelines-components/components/data_processing/autorag/text_extraction/component.py"
    )
    if not component.is_file():
        pytest.skip("Pipeline checkout is not available")
    tree = ast.parse(component.read_text(encoding="utf-8"))
    names = {"RAPIDOCR_BUNDLES", "LAYOUT_OCR_EXTENSIONS", "ASR_EXTENSIONS", "CHINESE_ALIASES"}
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name) and target.id in names
    }
    assert names == assignments.keys()
    assert extraction._RAPIDOCR_BUNDLES == assignments["RAPIDOCR_BUNDLES"]
    assert extraction.OCR_EXTENSIONS == assignments["LAYOUT_OCR_EXTENSIONS"]
    assert extraction.AUDIO_EXTENSIONS == assignments["ASR_EXTENSIONS"]
    assert extraction._CHINESE_OCR_LANGS == assignments["CHINESE_ALIASES"]
