# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections.abc import Hashable
from typing import Any

__all__ = [
    "ConstantMeta",
    "AI4RAGParamNames",
    "ChunkingConstraints",
    "ExperimentStep",
    "RetrievalConstraints",
    "ChatGenerationConstants",
    "TokenEstimation",
]


class ConstantMeta(type):
    """Metaclass for all instance classes that we desire to have constant attributes."""

    def __new__(mcs, name, bases, class_dict):
        _constant_attributes = [
            val
            for key, val in class_dict.items()
            if not key.startswith("__") and not callable(val) and isinstance(val, Hashable)
        ]
        class_dict["_constant_attributes"] = _constant_attributes

        new_class = super().__new__(mcs, name, bases, class_dict)

        return new_class

    def __setattr__(cls, name, value) -> None:
        raise AttributeError(f"Cannot modify attribute '{name}' after class creation.")

    def __iter__(cls) -> Any:
        yield from cls._constant_attributes

    def __contains__(cls, value: Any) -> bool:
        return value in cls._constant_attributes

    def validate(cls, value: Any) -> Any:
        """Validates if given value exists in defined constants.

        Parameters
        ----------
        value : Any
            Value to search in defined constants.

        Returns
        -------
        Any
            Returns provided value if valid.

        Raises
        ------
        ValueError
            When value doesn't exists in declared constants.
        """
        if value not in cls._constant_attributes:
            raise ValueError(f"Value {value} not found in defined constants.")
        return value


class AI4RAGParamNames(metaclass=ConstantMeta):
    """Parameter's names used in the experiment."""

    CHUNKING = "chunking"
    CHUNKING_METHOD = "chunking_method"
    CHUNK_SIZE = "chunk_size"
    CHUNK_OVERLAP = "chunk_overlap"
    EMBEDDING_MODEL = "embedding_model"
    DISTANCE_METRIC = "distance_metric"
    FOUNDATION_MODEL = "foundation_model"
    TRUNCATE_STRATEGY = "truncate_strategy"
    INPUT_SIZE = "input_size"
    RETRIEVAL = "retrieval"
    RETRIEVAL_METHOD = "retrieval_method"
    WINDOW_SIZE = "window_size"
    NUMBER_OF_CHUNKS = "number_of_chunks"
    SEARCH_MODE = "search_mode"
    RANKER_STRATEGY = "ranker_strategy"
    RANKER_K = "ranker_k"
    RANKER_ALPHA = "ranker_alpha"
    GENERATION = "generation"


class ExperimentStep(metaclass=ConstantMeta):
    """Steps occurring in the experiment engine."""

    MODEL_SELECTION = "model selection"
    OPTIMIZATION = "optimization"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    GENERATION = "generation"
    EVALUATION = "evaluation"


class ChatGenerationConstants(metaclass=ConstantMeta):
    """Constants used for setting the generation (inference) parameters for chat models only."""

    MAX_COMPLETION_TOKENS = 2048
    TEMPERATURE = 0.2


class ChunkingConstraints(metaclass=ConstantMeta):
    """Constants used to define chunking constraints on what below parameters can be."""

    MIN_CHUNK_SIZE = 128
    MAX_CHUNK_SIZE = 2048
    MIN_CHUNK_OVERLAP = 0
    MAX_CHUNK_OVERLAP = 256
    METHODS = ["recursive", "hybrid"]


class TokenEstimation(metaclass=ConstantMeta):
    """Character-based token estimation constant shared by chunkers and embedding models."""

    CHARS_PER_TOKEN = 4


class RetrievalConstraints(metaclass=ConstantMeta):
    """Constants used to define the permissible values for retrieval constraints."""

    MIN_NUMBER_OF_RETRIEVED_CHUNKS = 1
    MAX_NUMBER_OF_RETRIEVED_CHUNKS = 10
    METHODS = ["window", "simple"]
    MIN_WINDOW_SIZE = 0
    MAX_WINDOW_SIZE = 4
    SEARCH_MODES = ["vector", "hybrid"]
    RANKER_STRATEGIES = ["rrf", "weighted", "normalized"]
    MIN_RANKER_K = 1
    MAX_RANKER_K = 100
