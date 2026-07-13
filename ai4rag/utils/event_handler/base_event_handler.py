# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import NotRequired, TypedDict

__all__ = [
    "BaseEventHandler",
    "LogLevel",
    "PatternPayload",
    "EvaluationRecord",
]


class LogLevel(StrEnum):
    """Available log levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


# ---------------------------------------------------------------------------
# TypedDicts for on_pattern_creation – payload
# ---------------------------------------------------------------------------


class AggregateMetricScore(TypedDict):
    """Confidence interval for a single aggregate metric."""

    mean: float | None
    ci_low: float | None
    ci_high: float | None


class AggregateMetricPayload(TypedDict):
    """Single metric entry in the ``metrics`` list of :class:`EvaluationPayload`."""

    name: str
    evaluator: str
    description: str
    scores: AggregateMetricScore
    optimization_metric: NotRequired[bool]


class VectorStoreSettings(TypedDict, total=False):
    """Vector store configuration used by a RAG pattern."""

    provider_id: str
    vector_store_id: str
    provider_type: str


class ChunkingSettings(TypedDict):
    """Chunking parameters used during indexing."""

    method: str
    chunk_size: int
    chunk_overlap: int


class EmbeddingSettings(TypedDict, total=False):
    """Embedding model configuration used during indexing."""

    model_id: str
    embedding_params: dict


class RetrievalSettings(TypedDict, total=False):
    """Retrieval parameters. ``window_size`` and ranker fields are optional."""

    method: str
    number_of_chunks: int
    search_mode: str
    # present only when retrieval_method == "window"
    window_size: int
    # present only when search_mode == "hybrid"
    ranker_strategy: str
    ranker_k: int
    ranker_alpha: float


class GenerationSettings(TypedDict):
    """Foundation model configuration used during generation."""

    model_id: str
    context_template_text: str
    user_message_text: str
    system_message_text: str


class PatternSettings(TypedDict):
    """Full settings block of :class:`PatternPayload`."""

    vector_store_binding: VectorStoreSettings
    chunking: ChunkingSettings
    embedding: EmbeddingSettings
    retrieval: RetrievalSettings
    generation: GenerationSettings


class EvaluationPayload(TypedDict):
    """Evaluation section of :class:`PatternPayload`."""

    metrics: list[AggregateMetricPayload]


class PatternPayload(TypedDict):
    """Payload passed to :meth:`BaseEventHandler.on_pattern_creation`."""

    name: str
    max_combinations: int
    evaluation: EvaluationPayload
    duration_seconds: int
    settings: PatternSettings
    iteration: int


# ---------------------------------------------------------------------------
# TypedDicts for on_pattern_creation – evaluation_results
# ---------------------------------------------------------------------------


class AnswerContext(TypedDict):
    """Single retrieved chunk with its source document."""

    text: str
    document_id: str


class EvaluationMetricRecord(TypedDict):
    """Single metric score for a question."""

    name: str
    evaluator: str
    score: float


class EvaluationRecord(TypedDict):
    """Per-question evaluation entry in the ``evaluation_results`` list."""

    question: str
    correct_answers: list[str]
    answer: str
    answer_contexts: list[AnswerContext]
    metrics: list[EvaluationMetricRecord]


class BaseEventHandler(ABC):
    """
    Abstract class defining interface for streaming results and messages,
    to the service layer.
    """

    @abstractmethod
    def on_status_change(self, level: LogLevel, message: str, step: str | None = None) -> None:
        """
        Method called to notify about experiment's status change.

        Parameters
        ----------
        level : LogLevel
            Logging level

        message : str
            Text of streamed message

        step : str
            Currently performed step. It should be one of composition steps.
        """

    @abstractmethod
    def on_pattern_creation(
        self, payload: PatternPayload, evaluation_results: list[EvaluationRecord], **kwargs
    ) -> None:
        """
        Method called when single RAG pattern's evaluation is completed.

        Parameters
        ----------
        payload : dict
            Information about RAG pattern's location and name, calculated scores
            and message.

            Example content:

            {
                'name': 'Pattern1',
                'max_combinations': 24,
                'evaluation': {
                    'metrics': [
                        {
                            'name': 'answer_correctness',
                            'evaluator': 'unitxt',
                            'description': '',
                            'scores': {'mean': 0.0, 'ci_low': None, 'ci_high': None}
                        },
                        {
                            'name': 'faithfulness',
                            'evaluator': 'unitxt',
                            'description': '',
                            'scores': {'mean': 0.0909, 'ci_low': 0.0145, 'ci_high': 0.016},
                            'optimization_metric': True
                        },
                        {
                            'name': 'context_correctness',
                            'evaluator': 'unitxt',
                            'description': '',
                            'scores': {'mean': 0.0, 'ci_low': None, 'ci_high': None}
                        },
                    ],
                },
                'duration_seconds': 42,
                'settings': {
                    'vector_store_binding': {'provider_id': 'local_chroma', 'vector_store_id': 'ai4rag_20260317092550'},
                    'chunking': {'method': 'recursive', 'chunk_size': 1024, 'chunk_overlap': 256},
                    'embedding': {
                        'model_id': 'mock-em-1',
                        'embedding_params': {'embedding_dimension': 64},
                    },
                    'retrieval': {'method': 'window', 'number_of_chunks': 3, 'search_mode': 'vector', 'window_size': 3},
                    'generation': {
                        'model_id': 'mock-fm-2',
                        'context_template_text': '{document}',
                        'user_message_text': 'Context:\n{reference_documents}\n\nQuestion: {question}',
                        'system_message_text': 'System instruction...'
                    }
                },
                'iteration': 0,
            }


        evaluation_results : dict
            Results from single pattern evaluation.

            Example content:

            [
                {
                    "question": "<question_1>",
                    "answer": "<model's answer>",
                    "answer_contexts": [
                        {"text": "<content1_text>", "document_id": "document_1.pdf"},
                        {"text": "<content2_text>", "document_id": "document_2.pdf"},
                    ],
                    "correct_answers": ["correct_answer_for_question_1"],
                    "metrics": [
                        {"name": "answer_correctness", "evaluator": "unitxt", "score": 0.79},
                        {"name": "faithfulness", "evaluator": "unitxt", "score": 0.55},
                        {"name": "context_correctness", "evaluator": "unitxt", "score": 0.65},
                    ],
                },
                {
                    "question": "<question_2>",
                    "answer": "<model's answer>",
                    "answer_contexts": [
                        {"text": "<content3_text>", "document_id": "document_3.pdf"},
                        {"text": "<content4_text>", "document_id": "document_4.pdf"},
                    ],
                    "correct_answers": ["correct_answer_1_for_question_2", "correct_answer_2_for_question_3"],
                    "metrics": [
                        {"name": "answer_correctness", "evaluator": "unitxt", "score": 0.79},
                        {"name": "faithfulness", "evaluator": "unitxt", "score": 0.55},
                        {"name": "context_correctness", "evaluator": "unitxt", "score": 0.65},
                    ],
                },
            ]
        """
