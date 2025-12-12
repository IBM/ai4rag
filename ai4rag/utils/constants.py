#
# Copyright IBM Corp. 2025
# SPDX-License-Identifier: Apache-2.0
#
from typing import Any

__all__ = [
    "ConstantMeta",
    "AI4RAGParamNames",
    "GenerationConstants",
    "ChunkingConstraints",
    "ExperimentStep",
    "RetrievalConstraints",
    "SearchSpaceValidationErrors",
    "EventsToReport",
    "DEFAULT_WORD_TO_TOKEN_RATIO",
    "ChatGenerationConstants",
    "DefaultVectorStoreFieldNames",
]

DEFAULT_WORD_TO_TOKEN_RATIO = 1.5


class ConstantMeta(type):
    """Metaclass for all instance classes that we desire to have constant attributes."""

    def __new__(mcs, name, bases, class_dict):
        _constant_attributes = [
            val
            for key, val in class_dict.items()
            if not key.startswith("__") and not callable(val) and type(val) in (str, int, float)
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
    INFERENCE_MODEL_ID = "inference_model_id"
    TRUNCATE_STRATEGY = "truncate_strategy"
    INPUT_SIZE = "input_size"
    RETRIEVAL = "retrieval"
    RETRIEVAL_METHOD = "retrieval_method"
    RETRIEVAL_WINDOW_SIZE = "retrieval_window_size"
    NUMBER_OF_RETRIEVED_CHUNKS = "number_of_retrieved_chunks"
    GENERATION = "generation"


class ExperimentStep(metaclass=ConstantMeta):
    """Steps occurring in the experiment engine."""

    MODEL_SELECTION = "model selection"
    OPTIMIZATION = "optimization"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    GENERATION = "generation"
    EVALUATION = "evaluation"


class GenerationConstants(metaclass=ConstantMeta):
    """Constants used for setting the generation (inference) parameters."""

    MAX_NEW_TOKENS = 1000
    MIN_NEW_TOKENS = 1
    DECODING_METHOD = "greedy"


class ChatGenerationConstants(metaclass=ConstantMeta):
    """Constants used for setting the generation (inference) parameters for chat models only."""

    MAX_COMPLETION_TOKENS = 2048
    TEMPERATURE = 0.2


class ChunkingConstraints(metaclass=ConstantMeta):
    """Constants used to define chunking constraints on what below parameters can be."""

    MIN_CHUNK_SIZE = 128
    MAX_CHUNK_SIZE = 2048
    MIN_CHUNK_OVERLAP = 64
    MAX_CHUNK_OVERLAP = 512
    METHODS = ["recursive", "semantic"]


class RetrievalConstraints(metaclass=ConstantMeta):
    """Constants used to define the permissible values for retrieval constraints."""

    MIN_NUMBER_OF_RETRIEVED_CHUNKS = 1
    MAX_NUMBER_OF_RETRIEVED_CHUNKS = 10
    METHODS = ["window", "simple"]
    MIN_WINDOW_SIZE = 0
    MAX_WINDOW_SIZE = 4


class DefaultVectorStoreFieldNames(metaclass=ConstantMeta):
    """Constants used as field names during index building for Milvus and Elasticsearch"""

    CHUNK_SEQUENCE_NUMBER_FIELD = "sequence_number"
    DENSE_EMBEDDINGS_FIELD = "vector"
    SPARSE_EMBEDDINGS_FIELD = "sparse_embeddings"
    MILVUS_TEXT_FIELD = "text"
    ELASTICSEARCH_TEXT_FIELD = "text_field"
    METADATA_DOCUMENT_NAME_FIELD = "document_id"


class SearchSpaceValidationErrors(metaclass=ConstantMeta):
    """Constants used to define payload validation errors."""

    UNEXPECTED_KEYWORD_ARGUMENT = "unexpected_keyword_argument"
    LITERAL_ERROR = "literal_error"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LESS_THAN_EQUAL = "less_than_equal"
    GREATER_THAN_EQUAL = "greater_than_equal"
    INT_FROM_FLOAT = "int_from_float"
    INT_PARSING = "int_parsing"
    INT_TYPE = "int_type"
    LIST_TYPE = "list_type"


class EventsToReport(metaclass=ConstantMeta):
    """Constants used to name events reported on INFO level."""

    EMBEDDING = "Embedding"
    RETRIEVAL_GENERATION = "Retrieval and Generation"
