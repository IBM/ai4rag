import json
from pathlib import Path

from ai4rag.utils.event_handler.event_handler import BaseEventHandler, LevelType, AIServiceData
from ai4rag import logger


class LocalEventHandler(BaseEventHandler):
    """Stream logs to the console"""

    def on_status_change(self, level: LevelType, message: str, step: str | None = None) -> None:
        logger.debug("On status change: %s %s %s", level, message, step)

    def on_pattern_creation(
        self, payload: dict, evaluation_results: list, inference_service_data: AIServiceData, **kwargs
    ) -> None:
        # log only scores for evaluation_results and filter model's answers
        file_content = {"data": {el["question_id"]: el["scores"] for el in evaluation_results}}
        logger.debug("On pattern creation payload: %s\nEvaluation results: %s", payload, evaluation_results)
        output_path = kwargs.get("output_path", None)
        pattern_name = kwargs.get("pattern_name", None)

        file_content["payload"] = payload

        if output_path:
            dir_path = Path(output_path) / pattern_name
            dir_path.mkdir(exist_ok=False, parents=True)

            evaluation_results_path = dir_path / "evaluation_results.json"
            with open(evaluation_results_path, mode="w") as file:
                json.dump(file_content, file)
            logger.debug("Writing evaluation json in location: %s", evaluation_results_path)

            if inference_service_data.service_code:
                inference_service_function_path = dir_path / "inference_service.py"
                with open(inference_service_function_path, mode="w") as file:
                    file.write(inference_service_data.service_code)
                logger.debug("Writing inference service code in location: %s", inference_service_function_path)

            inference_service_metadata_path = dir_path / "inference_service_metadata.json"
            with open(inference_service_metadata_path, mode="w") as file:
                json.dump(inference_service_data.service_metadata, file)
            logger.debug("Writing inference service metadata in location: %s", inference_service_metadata_path)
