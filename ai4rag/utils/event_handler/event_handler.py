# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path
from typing import Any

from ai4rag import logger

from .base_event_handler import BaseEventHandler, EvaluationRecord, LogLevel, PatternPayload


class LocalEventHandler(BaseEventHandler):
    """Implementation used for local development and testing purposes.

    Parameters
    ----------
    output_path : str | Path | None, default=None
        Location where results for all patterns will be saved.
    """

    def __init__(self, output_path: str | Path | None = None):
        self.output_path = Path(output_path) if output_path else None

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        logger.info("LocalEventHandler ::: %s ::: %s ::: %s", level, step, message)

    def on_pattern_creation(
        self, payload: PatternPayload, evaluation_results: list[EvaluationRecord], **kwargs
    ) -> None:
        logger.info("LocalEventHandler ::: Pattern creation ::: %s", payload)
        pattern_name = payload.get("pattern_name", "default_pattern_name")

        if self.output_path:
            dir_path = self.output_path / pattern_name
            dir_path.mkdir(exist_ok=False, parents=True)

            evaluation_results_path = dir_path / "evaluation_results.json"
            with open(evaluation_results_path, encoding="utf-8", mode="w") as file:
                json.dump(evaluation_results, file)

            with open(dir_path / "pattern.json", encoding="utf-8", mode="w") as file2:
                json.dump(payload, file2)


class KFPEventHandler(BaseEventHandler):
    """Event handler that aggregates status changes and created patterns for post-experiment use.

    All status changes are collected in :attr:`status_changes` and all pattern results in
    :attr:`patterns`. When ``step`` is omitted from :meth:`on_status_change`, the last known
    step value is reused so every entry always carries a step label.

    To be used within kubeflow-pipelines components.
    """

    def __init__(self):
        self.status_changes: list[dict] = []
        self.patterns: list[dict[str, Any]] = []
        self._last_step: str | None = None

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        if step is not None:
            self._last_step = step
        self.status_changes.append({"level": level, "message": message, "step": self._last_step})

    def on_pattern_creation(
        self, payload: PatternPayload, evaluation_results: list[EvaluationRecord], **kwargs
    ) -> None:
        self.patterns.append({"payload": payload, "evaluation_results": evaluation_results, **kwargs})
