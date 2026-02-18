# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path

from ai4rag import logger
from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LogLevel


class LocalEventHandler(BaseEventHandler):
    """Stream logs to the console"""

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path

    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        logger.debug("On status change: %s %s %s", level, message, step)

    def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
        # log only scores for evaluation_results and filter model's answers
        file_content = {"data": {f"q{idx}": el["scores"] for idx, el in enumerate(evaluation_results)}}
        logger.debug("On pattern creation payload: %s\nEvaluation results: %s", payload, evaluation_results)
        pattern_name = kwargs.get("pattern_name", None)

        file_content["payload"] = payload

        if self.output_path:
            dir_path = Path(self.output_path) / pattern_name
            dir_path.mkdir(exist_ok=False, parents=True)

            evaluation_results_path = dir_path / "evaluation_results.json"
            with open(evaluation_results_path, mode="w") as file:
                json.dump(file_content, file)
            logger.debug("Writing evaluation json in location: %s", evaluation_results_path)
