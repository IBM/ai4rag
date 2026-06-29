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

__all__ = ["prepare_search_space_with_ogx"]


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

    if isinstance(client, OgxClient):
        models = _get_default_ogx_models(client)
        registered_foundation_models = models["foundation_models"]
        registered_embedding_models = models["embedding_models"]
    else:
        raise SearchSpaceValueError(f"Unrecognized client type: '{client.__class__.__name__}'")

    if validated_payload.foundation_models:
        foundation_models = _validate_availability_and_create_models(
            registered_models=registered_foundation_models,
            models_type="llm",
            client=client,
            provided_models_ids=[m.model_id for m in validated_payload.foundation_models],
        )
    else:
        foundation_models = _validate_availability_and_create_models(
            registered_models=registered_foundation_models,
            models_type="llm",
            client=client,
        )

    if validated_payload.embedding_models:
        embedding_models = _validate_availability_and_create_models(
            registered_models=registered_embedding_models,
            models_type="embedding",
            client=client,
            provided_models_ids=[m.model_id for m in validated_payload.embedding_models],
        )
    else:
        embedding_models = _validate_availability_and_create_models(
            registered_models=registered_embedding_models,
            models_type="embedding",
            client=client,
        )

    if benchmark_data is not None:
        for fm in foundation_models:
            lang = detect_language_with_llm(
                questions=[str(q) for q in benchmark_data["question"][:10]],
                generation_model=fm,
            )
            if lang is not None:
                fm.language = Language(**lang)

    fms_param = Parameter(name="foundation_model", values=foundation_models)
    ems_param = Parameter(name="embedding_model", values=embedding_models)

    logger.info("Selected foundation models for the experiment: %s.", [m.model_id for m in fms_param.values])
    logger.info("Selected embedding models for the experiment: %s.", [m.model_id for m in ems_param.values])

    return AI4RAGSearchSpace(
        params=[fms_param, ems_param],
        vector_store_type=vector_store_type,
    )
