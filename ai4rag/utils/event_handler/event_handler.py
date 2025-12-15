from abc import ABC, abstractmethod
from typing import Literal, TypeAlias
from dataclasses import dataclass


__all__ = ["BaseEventHandler", "LogLevel", "LevelType", "AIServiceData"]


LevelType: TypeAlias = Literal["info", "warning", "error"]


@dataclass
class AIServiceData:
    """Dataclass representing metadata and code of the given AI service."""

    service_metadata: dict
    service_code: str | None
    vector_store_type: Literal["chroma", "milvus"]


class LogLevel:
    """Available log levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class BaseEventHandler(ABC):
    """
    Abstract class defining interface for streaming results and messages,
    to the service layer.
    """

    @abstractmethod
    def on_status_change(self, level: LevelType, message: str, step: str | None = None) -> None:
        """
        Method called to notify about experiment's status change.

        Parameters
        ----------
        level : LevelType
            Logging level

        message : str
            Text of streamed message

        step : str
            Currently performed step. It should be one of composition steps.
        """

    @abstractmethod
    def on_pattern_creation(
        self, payload: dict, evaluation_results: list, inference_service_data: AIServiceData, **kwargs
    ) -> None:
        """
        Method called when single RAG pattern's evaluation is completed.

        Parameters
        ----------
        payload : dict
            Information about RAG pattern's location and name, calculated scores
            and message.

            Example:

            name --> Pattern 1
            iteration --> which pattern we are evaluating

            payload = {
                "metrics": {
                    "test_data": {
                        "answer_correctness": {
                            "mean": 0.7,
                            "ci_low": 0.6,
                            "ci_high": 0.8,
                        },
                    },
                },
                "context": {
                    "rag_pattern": {
                        "composition_steps" : [
                            "chunking", "embeddings", "vector_store", "retrieval", "generation"
                        ],
                        "duration": 3507.9,
                        "location": {},
                        "name": "Pattern 1",
                        "settings": {
                            "chunking": {
                                "method": "recursive",
                                "chunk_size": 256,
                                "overlap": 128
                            },
                            "embeddings": {
                                "truncate_strategy": "left",
                                "input_size": 384,
                                "model_name": "ibm/slate.30m.english.rtrvr"
                            },
                            "vector_store": {
                                "database": "milvus",
                                "index_name": "XD_1234_index_5678",
                                "distance_metric": "euclidean"
                                "operation": "upsert",
                                "document_schema": {...},
                            },
                            "retrieval": {
                                "method": "simple",
                                "number_of_chunks": 5
                            },
                            "generation": {
                                "model_name": "mistralai/mixtral-8x7b-instruct-v0-1",
                                "parameters": {
                                    "max_new_tokens": 256
                                },
                                "chat_template_messages": {
                                    "system_message_text": "...",
                                    "user_message_text": "...",
                                }
                                "context_template_text": "...",
                            }
                        }
                    }
                    "iteration": 1,
                    "max_combinations": 100
                }
            }

        evaluation_results : dict
            Results from single pattern evaluation.

            Example content:
            "data": [
                {
                    "question_id": "0",
                    "answer": "<model's answer>",
                    "answer_contexts": [
                        {"text": "<content1_text>", "document_id": "document_1.pdf"},
                        {"text": "<content2_text>", "document_id": "document_2.pdf"},
                    ]
                    "scores": {
                        "answer_correctness": 0.79,
                        "context_correctness": 0.65,
                    }
                },
                {
                    "question_id": "1",
                    "answer": "<model's answer>",
                    "answer_contexts": [
                        {"text": "<content3_text>", "document_id": "document_3.pdf"},
                        {"text": "<content4_text>", "document_id": "document_4.pdf"},
                    ]
                    "scores": {
                        "answer_correctness": 0.79,
                        "context_correctness": 0.65,
                    }
                }
            ]

        inference_service_data : AIServiceData
            Data for the AI inference service. It should contain both code and
            metadata for the AI (inference) service.
        """
