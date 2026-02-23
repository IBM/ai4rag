# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from pathlib import Path

from ai4rag import logger

from .base_event_handler import BaseEventHandler, LogLevel


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

    def on_pattern_creation(self, payload: dict, evaluation_results: list, **kwargs) -> None:
        logger.info("LocalEventHandler ::: Pattern creation ::: ", payload)
        pattern_name = payload.get("rag_pattern", {}).get("name", "default_pattern_name")

        if self.output_path:
            dir_path = self.output_path / pattern_name
            dir_path.mkdir(exist_ok=False, parents=True)

            evaluation_results_path = dir_path / "evaluation_results.json"
            with open(evaluation_results_path, mode="w") as file:
                json.dump(evaluation_results, file)

            with open(dir_path / "pattern.json", mode="w") as file2:
                json.dump(payload, file2)
