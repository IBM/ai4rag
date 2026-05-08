# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path

import pytest

from ai4rag.utils.event_handler import LogLevel
from ai4rag.utils.event_handler.event_handler import KFPEventHandler, LocalEventHandler

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PAYLOAD = {
    "pattern_name": "Pattern1",
    "scores": {
        "scores": {
            "answer_correctness": {"mean": 0.8, "ci_low": 0.7, "ci_high": 0.9},
            "faithfulness": {"mean": 0.9, "ci_low": 0.85, "ci_high": 0.95},
        },
        "question_scores": {
            "answer_correctness": {"q0": 0.8, "q1": 0.8},
            "faithfulness": {"q0": 0.9, "q1": 0.9},
        },
    },
    "execution_time": 42,
    "final_score": 0.9,
    "schema_version": "1.0",
    "producer": "ai4rag",
    "settings": {
        "vector_store": {"datasource_type": "local_chroma", "collection_name": "col_1"},
        "chunking": {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        "embedding": {"model_id": "em-1", "distance_metric": "cosine", "embedding_params": {}},
        "retrieval": {"method": "simple", "number_of_chunks": 3, "search_mode": "vector"},
        "generation": {
            "model_id": "fm-1",
            "context_template_text": "{document}",
            "user_message_text": "Q: {question}",
            "system_message_text": "You are helpful.",
        },
    },
    "iteration": 1,
}

EVALUATION_RESULTS = [
    {
        "question": "What is topic_0 about?",
        "correct_answers": ["topic_0 is about AI."],
        "answer": "It is about AI.",
        "answer_contexts": [{"text": "AI content", "document_id": "doc_1"}],
        "scores": {"answer_correctness": 0.8, "faithfulness": 0.9},
    }
]


# ---------------------------------------------------------------------------
# LocalEventHandler
# ---------------------------------------------------------------------------


class TestLocalEventHandlerInit:
    """Tests for LocalEventHandler.__init__."""

    def test_init_without_output_path(self):
        """output_path defaults to None."""
        handler = LocalEventHandler()

        assert handler.output_path is None

    def test_init_with_string_output_path(self, tmp_path):
        """String output_path is converted to Path."""
        handler = LocalEventHandler(output_path=str(tmp_path))

        assert isinstance(handler.output_path, Path)
        assert handler.output_path == tmp_path

    def test_init_with_path_output_path(self, tmp_path):
        """Path output_path is stored as Path."""
        handler = LocalEventHandler(output_path=tmp_path)

        assert handler.output_path == tmp_path


class TestLocalEventHandlerOnStatusChange:
    """Tests for LocalEventHandler.on_status_change."""

    def test_on_status_change_logs_message(self, mocker):
        """Calls logger.debug with level, step and message."""
        mock_logger = mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="test message", step="chunking")

        mock_logger.debug.assert_called_once()

    def test_on_status_change_with_step(self, mocker):
        """Step value is forwarded to the log call."""
        mock_logger = mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="msg", step="embedding")

        call_args = mock_logger.debug.call_args[0]
        assert "embedding" in call_args

    def test_on_status_change_without_step(self, mocker):
        """Omitting step does not raise and logs None."""
        mock_logger = mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler()

        handler.on_status_change(level=LogLevel.WARNING, message="msg")

        mock_logger.debug.assert_called_once()
        call_args = mock_logger.debug.call_args[0]
        assert None in call_args


class TestLocalEventHandlerOnPatternCreationWithoutOutputPath:
    """Tests for LocalEventHandler.on_pattern_creation when no output_path is set."""

    def test_only_logs_no_files_written(self, mocker):
        """No filesystem activity when output_path is None."""
        mock_logger = mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler()

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        mock_logger.debug.assert_called_once()

    def test_does_not_raise(self):
        """Calling without output_path must not raise."""
        handler = LocalEventHandler()

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)


class TestLocalEventHandlerOnPatternCreationWithOutputPath:
    """Tests for LocalEventHandler.on_pattern_creation when output_path is configured."""

    def test_creates_pattern_directory(self, tmp_path, mocker):
        """Creates a subdirectory named after pattern_name."""
        mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler(output_path=tmp_path)

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        assert (tmp_path / "Pattern1").is_dir()

    def test_writes_pattern_json(self, tmp_path, mocker):
        """pattern.json is written with the payload content."""
        mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler(output_path=tmp_path)

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        pattern_file = tmp_path / "Pattern1" / "pattern.json"
        assert pattern_file.exists()
        with open(pattern_file, encoding="utf-8") as f:
            written = json.load(f)
        assert written["pattern_name"] == "Pattern1"
        assert written["final_score"] == 0.9

    def test_writes_evaluation_results_json(self, tmp_path, mocker):
        """evaluation_results.json is written with the evaluation records."""
        mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler(output_path=tmp_path)

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        results_file = tmp_path / "Pattern1" / "evaluation_results.json"
        assert results_file.exists()
        with open(results_file, encoding="utf-8") as f:
            written = json.load(f)
        assert written == EVALUATION_RESULTS

    def test_uses_default_pattern_name_when_missing(self, tmp_path, mocker):
        """Falls back to 'default_pattern_name' when payload has no pattern_name."""
        mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler(output_path=tmp_path)
        payload_without_name = {k: v for k, v in PAYLOAD.items() if k != "pattern_name"}

        handler.on_pattern_creation(payload=payload_without_name, evaluation_results=EVALUATION_RESULTS)

        assert (tmp_path / "default_pattern_name").is_dir()

    def test_raises_when_pattern_directory_already_exists(self, tmp_path, mocker):
        """Raises when the pattern directory already exists (exist_ok=False)."""
        mocker.patch("ai4rag.utils.event_handler.event_handler.logger")
        handler = LocalEventHandler(output_path=tmp_path)

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        with pytest.raises(FileExistsError):
            handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)


# ---------------------------------------------------------------------------
# KFPEventHandler
# ---------------------------------------------------------------------------


class TestKFPEventHandlerInit:
    """Tests for KFPEventHandler.__init__."""

    def test_status_changes_starts_empty(self):
        """status_changes list is empty after construction."""
        handler = KFPEventHandler()

        assert handler.status_changes == []

    def test_patterns_starts_empty(self):
        """patterns list is empty after construction."""
        handler = KFPEventHandler()

        assert handler.patterns == []

    def test_last_step_starts_as_none(self):
        """_last_step is None after construction."""
        handler = KFPEventHandler()

        assert handler._last_step is None


class TestKFPEventHandlerOnStatusChange:
    """Tests for KFPEventHandler.on_status_change."""

    def test_entry_is_stored(self):
        """Each call appends one entry to status_changes."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="started", step="chunking")

        assert len(handler.status_changes) == 1

    def test_stored_entry_contains_level_message_step(self):
        """Stored entry contains correct level, message and step values."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="started", step="chunking")

        entry = handler.status_changes[0]
        assert entry["level"] == LogLevel.INFO
        assert entry["message"] == "started"
        assert entry["step"] == "chunking"

    def test_step_is_updated_on_each_call(self):
        """Passing a new step updates _last_step."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="a", step="chunking")
        handler.on_status_change(level=LogLevel.INFO, message="b", step="embedding")

        assert handler._last_step == "embedding"

    def test_missing_step_uses_last_known_step(self):
        """When step is omitted, the previously seen step is reused."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="first", step="chunking")
        handler.on_status_change(level=LogLevel.INFO, message="second")

        assert handler.status_changes[1]["step"] == "chunking"

    def test_missing_step_when_no_previous_step_stores_none(self):
        """When step has never been set, the stored step is None."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="first")

        assert handler.status_changes[0]["step"] is None

    def test_multiple_calls_accumulate(self):
        """All calls are kept in status_changes in insertion order."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="a", step="chunking")
        handler.on_status_change(level=LogLevel.WARNING, message="b", step="embedding")
        handler.on_status_change(level=LogLevel.ERROR, message="c")

        assert len(handler.status_changes) == 3
        assert handler.status_changes[0]["message"] == "a"
        assert handler.status_changes[1]["message"] == "b"
        assert handler.status_changes[2]["message"] == "c"

    def test_last_step_carries_across_multiple_stepless_calls(self):
        """Step set once is reused for all subsequent calls that omit it."""
        handler = KFPEventHandler()

        handler.on_status_change(level=LogLevel.INFO, message="a", step="generation")
        handler.on_status_change(level=LogLevel.INFO, message="b")
        handler.on_status_change(level=LogLevel.INFO, message="c")

        assert handler.status_changes[1]["step"] == "generation"
        assert handler.status_changes[2]["step"] == "generation"


class TestKFPEventHandlerOnPatternCreation:
    """Tests for KFPEventHandler.on_pattern_creation."""

    def test_pattern_is_stored(self):
        """Each call appends one entry to patterns."""
        handler = KFPEventHandler()

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        assert len(handler.patterns) == 1

    def test_stored_pattern_contains_payload(self):
        """Stored entry exposes the original payload under 'payload' key."""
        handler = KFPEventHandler()

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        assert handler.patterns[0]["payload"] is PAYLOAD

    def test_stored_pattern_contains_evaluation_results(self):
        """Stored entry exposes evaluation_results under 'evaluation_results' key."""
        handler = KFPEventHandler()

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)

        assert handler.patterns[0]["evaluation_results"] is EVALUATION_RESULTS

    def test_multiple_patterns_accumulate(self):
        """All on_pattern_creation calls are kept in insertion order."""
        handler = KFPEventHandler()
        second_payload = {**PAYLOAD, "pattern_name": "Pattern2"}

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS)
        handler.on_pattern_creation(payload=second_payload, evaluation_results=[])

        assert len(handler.patterns) == 2
        assert handler.patterns[0]["payload"]["pattern_name"] == "Pattern1"
        assert handler.patterns[1]["payload"]["pattern_name"] == "Pattern2"

    def test_extra_kwargs_are_stored(self):
        """Additional keyword arguments passed via **kwargs are included in the stored entry."""
        handler = KFPEventHandler()

        handler.on_pattern_creation(payload=PAYLOAD, evaluation_results=EVALUATION_RESULTS, extra_field="value")

        assert handler.patterns[0]["extra_field"] == "value"
