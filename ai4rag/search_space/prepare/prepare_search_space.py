# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

import pandas as pd
from openai import OpenAI

from ai4rag import logger
from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints
from ai4rag.search_space.prepare.language_detection import detect_language_with_llm
from ai4rag.search_space.prepare.models import get_embedding_models, get_foundation_models
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = ["prepare_search_space_with_maas"]


def _resolve_models_from_payload(
    validated_payload: AI4RAGConstraints,
    client: OpenAI,
) -> tuple[list, list]:
    """Discover and validate foundation and embedding models from the payload.

    MaaS carries no metadata distinguishing foundation from embedding models, so
    the payload must declare the model ids for both types explicitly. Endpoint
    discovery, instantiation and responsiveness checks are delegated to the
    :mod:`ai4rag.search_space.prepare.models` helpers.

    Parameters
    ----------
    validated_payload : AI4RAGConstraints
        Validated constraint payload specifying which models to include. Both
        ``foundation_models`` and ``embedding_models`` are required.
    client : OpenAI
        Shared serving client used to list models and serve chat/embeddings for
        every instantiated model.

    Returns
    -------
    tuple[list, list]
        A (foundation_models, embedding_models) pair of instantiated model objects.

    Raises
    ------
    SearchSpaceValueError
        When ``client`` is not an :class:`~openai.OpenAI` instance, when the
        payload omits foundation or embedding model ids (type cannot be inferred
        from MaaS metadata), or when a requested model is unavailable or does not
        respond.
    """
    if not validated_payload.foundation_models or not validated_payload.embedding_models:
        raise SearchSpaceValueError(
            "MaaS exposes no metadata to distinguish model types, so model ids must be provided per type. "
            "Provide both 'foundation_models' and 'embedding_models' in the payload."
        )

    foundation_models = get_foundation_models(client, [m.model_id for m in validated_payload.foundation_models])
    embedding_models = get_embedding_models(client, [m.model_id for m in validated_payload.embedding_models])

    return foundation_models, embedding_models


def _apply_language_detection(foundation_models: list, benchmark_data: pd.DataFrame) -> None:
    """Detect language from benchmark questions and set it on each foundation model in-place.

    Parameters
    ----------
    foundation_models : list
        Foundation model objects to update with detected language.
    benchmark_data : pd.DataFrame
        Benchmark data whose "question" column is sampled for detection.
    """
    for fm in foundation_models:
        lang = detect_language_with_llm(
            questions=[str(q) for q in benchmark_data["question"][:10]],
            generation_model=fm,
        )
        if lang is not None:
            fm.language = Language(**lang)
            logger.info("Model %s: language set to %s (%s).", fm.model_id, lang["name"], lang["code"])
        else:
            logger.warning("Model %s: language detection failed, falling back to auto-detect.", fm.model_id)


def prepare_search_space_with_maas(
    payload: dict[str, Any],
    client: OpenAI,
    vector_store_type: str = "milvus",
    benchmark_data: pd.DataFrame | None = None,
) -> AI4RAGSearchSpace:
    """Prepare an AI4RAGSearchSpace using OpenShift MaaS for model validation.

    Foundation and embedding models are discovered and validated via MaaS. Because
    MaaS carries no metadata distinguishing model types, the payload must declare
    the model ids for both foundation and embedding models. Chunking parameters
    (chunking_methods, chunk_sizes, chunk_overlaps) are validated locally against
    ChunkingConstraints and, when provided, override the platform defaults for
    those dimensions.

    Parameters
    ----------
    payload : dict[str, Any]
        A mapping of constraint names to their values. Supported keys:

        - "foundation_models" (list[dict]) — foundation model identifiers
          to include; required.
        - "embedding_models" (list[dict]) — embedding model identifiers
          to include; required.
        - "chunking_methods" (list[str]) — overrides the default
          chunking_method dimension (e.g. ["recursive"]).
          None keeps the platform default.
        - "chunk_sizes" (list[int]) — overrides the default
          chunk_size dimension (e.g. [256, 512]).
          None keeps the platform default.
        - "chunk_overlaps" (list[int]) — overrides the default
          chunk_overlap dimension (e.g. [0, 128]).
          None keeps the platform default.

    client : OpenAI
        General MaaS client used for model discovery and validation.

    vector_store_type : str, default="milvus"
        Type of vector store. When "chroma", hybrid search parameters are
        excluded from the default search space since ChromaDB does not support
        hybrid search.

    benchmark_data : pd.DataFrame | None, default=None
        Benchmark data used for language detection.
        If not given, models will use automatic language detection per session.

    Returns
    -------
    AI4RAGSearchSpace
        A valid AI4RAGSearchSpace used in RAG optimization process.

    Raises
    ------
    pydantic.ValidationError
        Raised when payload contains a non-recognized parameter name,
        when chunking_methods contains unsupported values, or when
        chunk_sizes or chunk_overlaps are out of range.
    SearchSpaceValueError
        Raised when client is not an OpenAI client, or when foundation or
        embedding model ids are missing from the payload.
    """
    logger.info("Preparing search space based on provided constraints: %s.", payload)

    validated_payload = AI4RAGConstraints(**payload)

    foundation_models, embedding_models = _resolve_models_from_payload(validated_payload, client)

    if benchmark_data is not None:
        _apply_language_detection(foundation_models, benchmark_data)

    fms_param = Parameter(name=AI4RAGParamNames.FOUNDATION_MODEL, values=foundation_models)
    ems_param = Parameter(name=AI4RAGParamNames.EMBEDDING_MODEL, values=embedding_models)
    logger.info("Selected foundation models for the experiment: %s.", [m.model_id for m in fms_param.values])
    logger.info("Selected embedding models for the experiment: %s.", [m.model_id for m in ems_param.values])

    extra_params: list[Parameter] = []
    if validated_payload.chunking_methods is not None:
        extra_params.append(
            Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=tuple(validated_payload.chunking_methods))
        )
    if validated_payload.chunk_sizes is not None:
        extra_params.append(Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=tuple(validated_payload.chunk_sizes)))
    if validated_payload.chunk_overlaps is not None:
        extra_params.append(
            Parameter(name=AI4RAGParamNames.CHUNK_OVERLAP, values=tuple(validated_payload.chunk_overlaps))
        )

    return AI4RAGSearchSpace(
        params=[fms_param, ems_param, *extra_params],
        vector_store_type=vector_store_type,
    )
