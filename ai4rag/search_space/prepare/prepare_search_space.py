# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

import pandas as pd
from ogx_client import OgxClient

from ai4rag import logger
from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints
from ai4rag.search_space.prepare.language_detection import detect_language_with_llm
from ai4rag.search_space.prepare.ogx_utils import (
    _get_default_ogx_models,
    _validate_availability_and_create_models,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace
from ai4rag.utils.constants import AI4RAGParamNames

__all__ = ["prepare_search_space_with_ogx", "prepare_search_space_custom"]


def _resolve_models_from_payload(
    validated_payload: AI4RAGConstraints,
    client: OgxClient,
) -> tuple[list, list]:
    """Retrieve and validate foundation and embedding models from OGX.

    Parameters
    ----------
    validated_payload : AI4RAGConstraints
        Validated constraint payload specifying which models to include.
    client : OgxClient
        Authenticated OGX client used for model discovery and validation.

    Returns
    -------
    tuple[list, list]
        A ``(foundation_models, embedding_models)`` pair of instantiated
        model objects ready for use in the search space.

    Raises
    ------
    SearchSpaceValueError
        When *client* is not an :class:`OgxClient` instance.
    """
    if not isinstance(client, OgxClient):
        raise SearchSpaceValueError(f"Unrecognized client type: '{client.__class__.__name__}'")

    models = _get_default_ogx_models(client)

    if validated_payload.foundation_models:
        foundation_models = _validate_availability_and_create_models(
            registered_models=models["foundation_models"],
            models_type="llm",
            client=client,
            provided_models_ids=[m.model_id for m in validated_payload.foundation_models],
        )
    else:
        foundation_models = _validate_availability_and_create_models(
            registered_models=models["foundation_models"],
            models_type="llm",
            client=client,
        )

    if validated_payload.embedding_models:
        embedding_models = _validate_availability_and_create_models(
            registered_models=models["embedding_models"],
            models_type="embedding",
            client=client,
            provided_models_ids=[m.model_id for m in validated_payload.embedding_models],
        )
    else:
        embedding_models = _validate_availability_and_create_models(
            registered_models=models["embedding_models"],
            models_type="embedding",
            client=client,
        )

    return foundation_models, embedding_models


def _apply_language_detection(foundation_models: list, benchmark_data: pd.DataFrame) -> None:
    """Detect language from benchmark questions and set it on each foundation model in-place.

    Parameters
    ----------
    foundation_models : list
        Foundation model objects to update with detected language.
    benchmark_data : pd.DataFrame
        Benchmark data whose ``question`` column is sampled for detection.
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


def _build_model_params(foundation_models: list, embedding_models: list) -> tuple[Parameter, Parameter]:
    """Create :class:`Parameter` objects for model lists and log selections.

    Parameters
    ----------
    foundation_models : list
        Foundation model instances to wrap in a Parameter.
    embedding_models : list
        Embedding model instances to wrap in a Parameter.

    Returns
    -------
    tuple[Parameter, Parameter]
        ``(fms_param, ems_param)`` ready for inclusion in a search space.
    """
    fms_param = Parameter(name="foundation_model", values=foundation_models)
    ems_param = Parameter(name="embedding_model", values=embedding_models)
    logger.info("Selected foundation models for the experiment: %s.", [m.model_id for m in fms_param.values])
    logger.info("Selected embedding models for the experiment: %s.", [m.model_id for m in ems_param.values])
    return fms_param, ems_param


def prepare_search_space_with_ogx(
    payload: dict[str, Any],
    client: OgxClient,
    vector_store_type: str = "ogx",
    benchmark_data: pd.DataFrame | None = None,
) -> AI4RAGSearchSpace:
    """
    Prepare AutoRAGSearchSpace.

    Parameters
    ----------
    payload : dict[str, Any]
        A mapping between parameter name and its associated values.

    client : OgxClient
        Client instance for listing and validating available models.

    vector_store_type : str, default="ogx"
        Type of vector store. Supported values: ``"ogx"`` and ``"chroma"``.
        When ``"chroma"``, hybrid search parameters are excluded from the
        default search space since ChromaDB does not support hybrid search.

    benchmark_data : pd.DataFrame | None, default=None
        Benchmark data used for language detection.
        If not given, models with use automatic language detection per session.

    Returns
    -------
    AI4RAGSearchSpace
        A valid AI4RAGSearchSpace used in RAG optimization process.

    Raises
    ------
    SearchSpaceValueError
        Raised when payload contains non-recognized parameter name.
    """
    logger.info("Preparing search space based on provided constraints: %s.", payload)

    validated_payload = AI4RAGConstraints(**payload)

    skipped = [f for f in ("chunking_methods", "chunk_sizes") if getattr(validated_payload, f) is not None]
    if skipped:
        logger.warning(
            "Fields %s are not used by prepare_search_space_with_ogx and will be skipped. "
            "Use prepare_search_space_custom to apply chunking constraints.",
            skipped,
        )

    foundation_models, embedding_models = _resolve_models_from_payload(validated_payload, client)

    if benchmark_data is not None:
        _apply_language_detection(foundation_models, benchmark_data)

    fms_param, ems_param = _build_model_params(foundation_models, embedding_models)

    return AI4RAGSearchSpace(
        params=[fms_param, ems_param],
        vector_store_type=vector_store_type,
    )


def prepare_search_space_custom(
    payload: dict[str, Any],
    client: OgxClient,
    vector_store_type: str = "ogx",
    benchmark_data: pd.DataFrame | None = None,
) -> AI4RAGSearchSpace:
    """Prepare an :class:`AI4RAGSearchSpace` with optional chunking customization.

    Extends :func:`prepare_search_space_with_ogx` by allowing the caller to
    override the default ``chunking_method`` and ``chunk_size`` dimensions of
    the search space via the *payload* dict.  All other parameters and rules
    remain unchanged.

    Parameters
    ----------
    payload : dict[str, Any]
        A mapping of constraint names to their values.  Supports all keys
        accepted by :func:`prepare_search_space_with_ogx` plus:

        - ``"chunking_methods"`` *(list[str])* — overrides the default
          ``chunking_method`` dimension (e.g. ``["recursive"]``).
        - ``"chunk_sizes"`` *(list[int])* — overrides the default
          ``chunk_size`` dimension (e.g. ``[256, 512]``).

        Omitting a key keeps the platform default for that dimension.

    client : OgxClient
        Client instance for listing and validating available models.

    vector_store_type : str, default="ogx"
        Type of vector store. Supported values: ``"ogx"`` and ``"chroma"``.
        When ``"chroma"``, hybrid search parameters are excluded from the
        default search space since ChromaDB does not support hybrid search.

    benchmark_data : pd.DataFrame | None, default=None
        Benchmark data used for language detection.
        If not given, models will use automatic language detection per session.

    Returns
    -------
    AI4RAGSearchSpace
        A valid AI4RAGSearchSpace used in RAG optimization process.

    Raises
    ------
    SearchSpaceValueError
        Raised when payload contains a non-recognized parameter name or
        when *client* is not an :class:`OgxClient`.
    """
    logger.info("Preparing custom search space based on provided constraints: %s.", payload)

    validated_payload = AI4RAGConstraints(**payload)
    foundation_models, embedding_models = _resolve_models_from_payload(validated_payload, client)

    if benchmark_data is not None:
        _apply_language_detection(foundation_models, benchmark_data)

    fms_param, ems_param = _build_model_params(foundation_models, embedding_models)

    extra_params: list[Parameter] = []
    if validated_payload.chunking_methods is not None:
        extra_params.append(
            Parameter(name=AI4RAGParamNames.CHUNKING_METHOD, values=tuple(validated_payload.chunking_methods))
        )
    if validated_payload.chunk_sizes is not None:
        extra_params.append(Parameter(name=AI4RAGParamNames.CHUNK_SIZE, values=tuple(validated_payload.chunk_sizes)))

    return AI4RAGSearchSpace(
        params=[fms_param, ems_param, *extra_params],
        vector_store_type=vector_store_type,
    )
