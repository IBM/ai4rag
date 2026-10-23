# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for the offline local Hugging Face ASR adapter."""

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from ai4rag.utils.data.local_hf_asr_pipeline import (
    LocalHuggingFaceAsrPipeline,
    LocalHuggingFaceAsrPipelineOptions,
    _decode_audio,
    _validate_model_directory,
)


def _model_directory(tmp_path):
    """Create the minimum validated local Transformers model layout."""
    for name in ("config.json", "preprocessor_config.json", "tokenizer_config.json", "model.safetensors"):
        content = json.dumps({"model_type": "whisper"}) if name == "config.json" else "{}"
        (tmp_path / name).write_text(content, encoding="utf-8")
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
    (tmp_path / "config.json").write_text('{"model_type": "whisper"}', encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="preprocessor_config.json"):
        _validate_model_directory(str(tmp_path))


@pytest.mark.parametrize("model_type", ["", "wav2vec2"])
def test_validate_model_directory_rejects_non_whisper_model(tmp_path, model_type):
    """Only local Whisper Transformers models are accepted for audio extraction."""
    model_dir = _model_directory(tmp_path)
    (model_dir / "config.json").write_text(json.dumps({"model_type": model_type}), encoding="utf-8")

    with pytest.raises(ValueError, match="only Hugging Face Whisper models"):
        _validate_model_directory(str(model_dir))


def test_validate_model_directory_rejects_invalid_model_config(tmp_path):
    """A malformed model configuration must fail before loading Transformers."""
    model_dir = _model_directory(tmp_path)
    (model_dir / "config.json").write_text("not json", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid config.json"):
        _validate_model_directory(str(model_dir))


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


def test_build_document_preserves_timestamped_transcript(monkeypatch, tmp_path):
    """Decoded waveforms reach ASR and retain document and segment provenance."""
    model_dir = _model_directory(tmp_path)
    adapter = LocalHuggingFaceAsrPipeline(LocalHuggingFaceAsrPipelineOptions(asr_model_path=str(model_dir)))
    waveform = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    decode_audio = mock.MagicMock(return_value=waveform)
    monkeypatch.setattr("ai4rag.utils.data.local_hf_asr_pipeline._decode_audio", decode_audio)
    transcriber = mock.MagicMock(
        return_value={
            "chunks": [
                {"text": " First segment ", "timestamp": (1.0, 2.0)},
                {"text": "", "timestamp": (2.0, 3.0)},
                {"text": "No end", "timestamp": (3.0, None)},
            ]
        }
    )
    transcriber.feature_extractor.sampling_rate = 22050
    adapter._transcriber = transcriber  # pylint: disable=protected-access
    result = SimpleNamespace(input=SimpleNamespace(file=tmp_path / "recording.mp3", document_hash="0" * 64))

    adapter._build_document(result)  # pylint: disable=protected-access

    decode_audio.assert_called_once_with(result.input.file, 22050)
    transcriber.assert_called_once_with({"raw": waveform, "sampling_rate": 22050}, return_timestamps=True)
    assert result.document.origin.filename == "recording.mp3"
    assert result.document.origin.mimetype == "audio/mpeg"
    assert result.document.texts[0].text == "First segment"
    source = result.document.texts[0].source[0]
    assert (source.start_time, source.end_time) == (1.0, 2.0)


@pytest.mark.parametrize("sampling_rate", [16000, 22050])
def test_decode_audio_opens_file_directly(monkeypatch, tmp_path, sampling_rate):
    """FFmpeg receives a seekable filename and emits the requested mono waveform."""
    audio = tmp_path / "recording with spaces.m4a"
    expected = np.array([0.25, -0.5, 0.75], dtype="<f4")
    run = mock.MagicMock(return_value=SimpleNamespace(stdout=expected.tobytes()))
    monkeypatch.setattr("ai4rag.utils.data.local_hf_asr_pipeline.subprocess.run", run)

    waveform = _decode_audio(audio, sampling_rate)

    np.testing.assert_array_equal(waveform, expected)
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    command = run.call_args.args[0]
    assert command[command.index("-i") + 1] == str(audio)
    assert command[command.index("-ar") + 1] == str(sampling_rate)
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-f") + 1] == "f32le"
    assert "pipe:0" not in command
    assert run.call_args.kwargs == {"stdin": subprocess.DEVNULL, "capture_output": True, "check": True}


def test_decode_audio_reports_missing_ffmpeg(monkeypatch, tmp_path):
    """Missing system FFmpeg is reported as an actionable dependency error."""
    run = mock.MagicMock(side_effect=FileNotFoundError("ffmpeg"))
    monkeypatch.setattr("ai4rag.utils.data.local_hf_asr_pipeline.subprocess.run", run)

    with pytest.raises(RuntimeError, match="ffmpeg executable on PATH"):
        _decode_audio(tmp_path / "recording.m4a", 16000)


def test_decode_audio_preserves_ffmpeg_error(monkeypatch, tmp_path):
    """Decoder failures include the filename and FFmpeg's diagnostic output."""
    error = subprocess.CalledProcessError(1, ["ffmpeg"], stderr=b"Invalid data found when processing input\n")
    run = mock.MagicMock(side_effect=error)
    monkeypatch.setattr("ai4rag.utils.data.local_hf_asr_pipeline.subprocess.run", run)

    with pytest.raises(RuntimeError, match="recording.m4a.*Invalid data found") as exc_info:
        _decode_audio(tmp_path / "recording.m4a", 16000)

    assert exc_info.value.__cause__ is error


def test_decode_audio_reports_chnl_limitation(monkeypatch, tmp_path):
    """Unsupported M4A channel boxes produce an actionable conversion message."""
    error = subprocess.CalledProcessError(
        183,
        ["ffmpeg"],
        stderr=b"Unsupported 'chnl' box with version 1, flags: 0error reading header\n",
    )
    run = mock.MagicMock(side_effect=error)
    monkeypatch.setattr("ai4rag.utils.data.local_hf_asr_pipeline.subprocess.run", run)

    with pytest.raises(
        RuntimeError,
        match=(
            "Cannot decode 'recording.m4a': this M4A uses a version-1 'chnl' "
            "channel-layout box that the installed FFmpeg does not support. "
            "Convert the file to standard AAC-LC M4A, MP3, or WAV, or upgrade "
            "the runtime image to FFmpeg 8.1.2 or newer."
        ),
    ) as exc_info:
        _decode_audio(tmp_path / "recording.m4a", 16000)

    assert exc_info.value.__cause__ is error


def test_decode_audio_rejects_empty_output(monkeypatch, tmp_path):
    """FFmpeg can exit successfully yet produce no samples for an unreadable file."""
    run = mock.MagicMock(return_value=SimpleNamespace(stdout=b""))
    monkeypatch.setattr("ai4rag.utils.data.local_hf_asr_pipeline.subprocess.run", run)

    with pytest.raises(ValueError, match="no audio samples.*recording.m4a"):
        _decode_audio(tmp_path / "recording.m4a", 16000)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is required for audio decoding")
@pytest.mark.parametrize("extension", ["m4a", "mp3", "wav"])
def test_decode_audio_with_real_ffmpeg(tmp_path, extension):
    """Trailing-metadata M4A and existing formats decode to mono at the model rate."""
    audio = tmp_path / f"recording.{extension}"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=5",
            "-ac",
            "2",
            str(audio),
        ],
        capture_output=True,
        check=True,
    )
    if extension == "m4a":
        encoded = audio.read_bytes()
        assert encoded.index(b"mdat") < encoded.index(b"moov")

    waveform = _decode_audio(audio, 16000)

    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert waveform.size / 16000 == pytest.approx(5, abs=0.1)
    assert np.isfinite(waveform).all()
    assert np.max(np.abs(waveform)) > 0.01
