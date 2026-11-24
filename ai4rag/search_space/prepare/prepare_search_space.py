# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from collections.abc import Sequence
from typing import Any

import pandas as pd
from openai import OpenAI

from ai4rag import logger
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints, SearchSpaceConstraints
from ai4rag.search_space.prepare.language_detection import detect_language_with_llm
from ai4rag.search_space.prepare.models import (
    _validate_model_responsiveness,
    get_embedding_models,
    get_foundation_models,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = ["prepare_search_space_with_maas", "prepare_search_space_with_models"]


def _resolve_models_from_payload(
    validated_payload: AI4RAGConstraints,
    client: OpenAI,
) -> tuple[list[OpenAIFoundationModel], list[OpenAIEmbeddingModel]]:
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
    tuple[list[OpenAIFoundationModel], list[OpenAIEmbeddingModel]]
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


def _apply_language_detection(
    foundation_models: Sequence[OpenAIFoundationModel],
    benchmark_data: pd.DataFrame | None,
    *,
    include_error_details: bool = True,
) -> None:
    """Detect language from benchmark questions and set it on each foundation model in-place.

    Parameters
    ----------
    foundation_models : Sequence[OpenAIFoundationModel]
        Foundation model objects to update with detected language.
    benchmark_data : pd.DataFrame | None
        Benchmark data whose "question" column is sampled for detection.
    include_error_details : bool, default=True
        Include exception messages in language detection logs.
    """
    if benchmark_data is None:
        return
    for fm in foundation_models:
        lang = detect_language_with_llm(
            questions=[str(q) for q in benchmark_data["question"][:10]],
            generation_model=fm,
            include_error_details=include_error_details,
        )
        if lang is not None:
            fm.language = Language(**lang)
            logger.info("Model %s: language set to %s (%s).", fm.model_id, lang["name"], lang["code"])
        else:
            logger.warning("Model %s: language detection failed, falling back to auto-detect.", fm.model_id)


def prepare_search_space_with_maas(
    payload: dict[str, Any],
    client: OpenAI,
    benchmark_data: pd.DataFrame | None = None,
    vector_store_type: str = "milvus",
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

        - "template_ids" (list[str]) — template search dimension; None uses the backend default.
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

    benchmark_data : pd.DataFrame | None, default=None
        Benchmark data used for language detection.
        If not given, models will use automatic language detection per session.

    vector_store_type : str, default="milvus"
        Vector-store provider used by the experiment. It selects defaults that
        are supported by that backend; Neo4j produces graph-only retrieval.

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
    search_space = _build_search_space(validated_payload, foundation_models, embedding_models, vector_store_type)
    _apply_language_detection(foundation_models, benchmark_data)
    return search_space


def prepare_search_space_with_models(
    payload: dict[str, Any],
    *,
    foundation_models: Sequence[OpenAIFoundationModel],
    embedding_models: Sequence[OpenAIEmbeddingModel],
    benchmark_data: pd.DataFrame | None = None,
    vector_store_type: str = "milvus",
) -> AI4RAGSearchSpace:
    """Prepare a search space with initialized models that retain their clients.

    Each supplied model is checked with a minimal chat or embedding request
    through its own client. Model IDs must be unique within each role because
    optimization uses them to distinguish candidates.

    Parameters
    ----------
    payload : dict[str, Any]
        Template and chunking constraints. Supports ``template_ids``,
        ``chunking_methods``, ``chunk_sizes``, and ``chunk_overlaps`` with the
        same validation and defaults as :func:`prepare_search_space_with_maas`.
    foundation_models : Sequence[OpenAIFoundationModel]
        One or more generation candidates in the search space.
    embedding_models : Sequence[OpenAIEmbeddingModel]
        One or more embedding candidates in the search space.
    benchmark_data : pd.DataFrame | None, default=None
        Benchmark data whose ``question`` column is sampled for language
        detection through each foundation model. Successful detection updates
        that model's language and regenerates its user prompt in place.
    vector_store_type : str, default="milvus"
        Vector-store provider that selects the backend-specific defaults.

    Returns
    -------
    AI4RAGSearchSpace
        Search space containing the supplied model objects.

    Raises
    ------
    pydantic.ValidationError
        If the payload contains unsupported keys or invalid constraints.
    SearchSpaceValueError
        If either sequence is empty, contains the wrong wrapper type, or
        repeats a model ID. Also raised if any model fails its probe or a
        template is incompatible with the vector-store provider.
    """
    logger.info("Preparing search space based on provided constraints: %s.", payload)
    validated_payload = SearchSpaceConstraints(**payload)

    _validate_model_candidates(foundation_models, OpenAIFoundationModel, "foundation_models")
    _validate_model_candidates(embedding_models, OpenAIEmbeddingModel, "embedding_models")
    foundation_models, embedding_models = list(foundation_models), list(embedding_models)
    search_space = _build_search_space(validated_payload, foundation_models, embedding_models, vector_store_type)
    _validate_model_responsiveness(foundation_models, embedding_models)
    _apply_language_detection(foundation_models, benchmark_data, include_error_details=False)
    return search_space


def _validate_model_candidates(
    models: Sequence[OpenAIFoundationModel | OpenAIEmbeddingModel],
    model_class: type[OpenAIFoundationModel] | type[OpenAIEmbeddingModel],
    name: str,
) -> None:
    """Validate a model dimension before applying language detection."""
    if not isinstance(models, Sequence) or not models or any(not isinstance(model, model_class) for model in models):
        raise SearchSpaceValueError(f"'{name}' must be a non-empty sequence of {model_class.__name__} instances.")
    if len({model.model_id for model in models}) != len(models):
        raise SearchSpaceValueError(f"'{name}' must contain unique model IDs.")


def _build_search_space(
    validated_payload: SearchSpaceConstraints,
    foundation_models: list[OpenAIFoundationModel],
    embedding_models: list[OpenAIEmbeddingModel],
    vector_store_type: str,
) -> AI4RAGSearchSpace:
    """Assemble the search space and validate its backend compatibility."""
    fms_param = Parameter(name=AI4RAGParamNames.FOUNDATION_MODEL, values=foundation_models)
    ems_param = Parameter(name=AI4RAGParamNames.EMBEDDING_MODEL, values=embedding_models)
    logger.info("Selected foundation models for the experiment: %s.", [m.model_id for m in fms_param.values])
    logger.info("Selected embedding models for the experiment: %s.", [m.model_id for m in ems_param.values])

    extra_params: list[Parameter] = []
    if validated_payload.template_ids is not None:
        extra_params.append(Parameter(name=AI4RAGParamNames.TEMPLATE_ID, values=tuple(validated_payload.template_ids)))
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
