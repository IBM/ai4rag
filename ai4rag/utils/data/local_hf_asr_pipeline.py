# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Offline Hugging Face Whisper backend for Docling audio extraction."""

import json
import mimetypes
from pathlib import Path

from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import AsrPipelineOptions
from docling.pipeline.asr_pipeline import AsrPipeline
from docling.pipeline.base_pipeline import BasePipeline
from docling_core.types.doc import ContentLayer, DocItemLabel, DoclingDocument, DocumentOrigin, TrackSource
from pydantic import ConfigDict


class LocalHuggingFaceAsrPipelineOptions(AsrPipelineOptions):
    """Options for :class:`LocalHuggingFaceAsrPipeline`.

    Parameters
    ----------
    asr_model_path
        Directory containing an approved Hugging Face Transformers Whisper
        model. ``None`` is permitted while constructing a converter so
        non-audio runs do not require audio model artifacts; audio conversion
        then fails before any remote lookup is attempted.
    """

    model_config = ConfigDict(extra="forbid")

    asr_model_path: str | None = None


class LocalHuggingFaceAsrPipeline(AsrPipeline):  # pylint: disable=super-init-not-called,non-parent-init-called
    """Docling ASR pipeline that loads only a local Transformers model.

    The native Docling ASR initializer creates an ``openai-whisper`` model,
    whose checkpoint format differs from the approved Transformers modelcar.
    This pipeline intentionally initializes :class:`BasePipeline` directly and
    lazily loads the local model on the first audio document.
    """

    def __init__(  # pylint: disable=super-init-not-called,non-parent-init-called
        self, pipeline_options: LocalHuggingFaceAsrPipelineOptions
    ):
        BasePipeline.__init__(self, pipeline_options)
        self.keep_backend = True
        self.pipeline_options = pipeline_options
        self._transcriber = None

    def _load_transcriber(self):
        """Load the approved local ASR model without any network fallback."""
        if self._transcriber is not None:
            return self._transcriber

        model_dir = _validate_model_directory(self.pipeline_options.asr_model_path)

        # Keep imports lazy: audio dependencies are optional and non-audio
        # extraction must not import the Transformers inference stack.
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_dir, local_files_only=True)
        model.eval()
        self._transcriber = pipeline(
            task="automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=-1,
            chunk_length_s=30,
        )
        return self._transcriber

    def _build_document(self, conv_res: ConversionResult) -> ConversionResult:
        """Transcribe one audio input and retain segment timing provenance."""
        audio = Path(conv_res.input.file)
        output = self._load_transcriber()(str(audio), return_timestamps=True)

        conv_res.document = DoclingDocument(
            name=audio.stem,
            origin=DocumentOrigin(
                filename=audio.name,
                mimetype=mimetypes.guess_type(audio.name)[0] or "audio/x-wav",
                binary_hash=conv_res.input.document_hash,
            ),
        )
        for chunk in output.get("chunks", []):
            timestamp = chunk.get("timestamp")
            if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
                continue
            start, end = timestamp
            text = str(chunk.get("text", "")).strip()
            if not text or start is None or end is None:
                continue
            conv_res.document.add_text(
                label=DocItemLabel.TEXT,
                text=text,
                content_layer=ContentLayer.BODY,
                source=TrackSource(
                    start_time=float(start),
                    end_time=max(float(end), float(start) + 0.001),
                ),
            )
        return conv_res


def _validate_model_directory(raw_path: str | None) -> str:
    """Validate a local Transformers model directory for offline ASR."""
    if not raw_path:
        raise FileNotFoundError(
            "Audio extraction requires HF_MODEL_DIR to point to the approved local Whisper model directory. "
            "Runtime model downloads are disabled."
        )

    model_dir = Path(raw_path).expanduser().resolve()
    config_path = model_dir / "config.json"
    if not model_dir.is_dir() or not config_path.is_file():
        raise FileNotFoundError(
            f"HF_MODEL_DIR={model_dir} is not a local Transformers model directory containing config.json."
        )

    try:
        with config_path.open(encoding="utf-8") as config_file:
            model_config = json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"HF_MODEL_DIR={model_dir} has an invalid config.json.") from exc

    model_type = model_config.get("model_type", "") if isinstance(model_config, dict) else ""
    if model_type != "whisper":
        raise ValueError(
            "Audio extraction supports only Hugging Face Whisper models; "
            f"HF_MODEL_DIR={model_dir} declares model_type={model_type!r}."
        )

    required = ("preprocessor_config.json", "tokenizer_config.json")
    missing = [name for name in required if not (model_dir / name).is_file()]
    has_weights = any(
        (model_dir / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    )
    if not model_dir.is_dir() or missing or not has_weights:
        details = missing + ([] if has_weights else ["model weights"])
        raise FileNotFoundError(
            f"HF_MODEL_DIR={model_dir} is not a complete local Transformers Whisper model directory. "
            f"Missing: {', '.join(details)}."
        )
    return str(model_dir)
