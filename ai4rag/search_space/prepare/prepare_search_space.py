# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any

from llama_stack_client import LlamaStackClient

from ai4rag import logger
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints
from ai4rag.search_space.prepare.llama_stack_utils import (
    _are_provided_models_available,
    _get_default_llama_stack_models,
)
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

__all__ = ["prepare_search_space_with_llama_stack"]


def prepare_search_space_with_llama_stack(
    payload: dict[str, Any],
    client: LlamaStackClient,
    vector_store_type: str = "ls_milvus",
) -> AI4RAGSearchSpace:
    """
    Prepare AutoRAGSearchSpace.

    Parameters
    ----------
    payload : dict[str, Any]
        A mapping between parameter name and its associated values.

    client : LlamaStackClient
        Client instance for listing and validating available models.

    vector_store_type : str, default="ls_milvus"
        Type of vector store. Use "ls_<provider_id>" for Llama Stack vector stores
        (e.g., "ls_milvus", "ls_qdrant"). When "chroma", hybrid search parameters
        are excluded from the default search space since ChromaDB does not support hybrid search.

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

    if isinstance(client, LlamaStackClient):
        models = _get_default_llama_stack_models(client)
        default_foundation_models = models["foundation_models"]
        default_embedding_models = models["embedding_models"]
    else:
        raise SearchSpaceValueError(f"Unrecognized client type: '{client.__class__.__name__}'")

    if validated_payload.foundation_models:
        _are_provided_models_available(
            provided_models=validated_payload.foundation_models,
            available_models=default_foundation_models,
            not_responding_models=models["not_responding_foundation_models"],
        )
    if validated_payload.embedding_models:
        _are_provided_models_available(
            provided_models=validated_payload.embedding_models,
            available_models=default_embedding_models,
            not_responding_models=models["not_responding_embedding_models"],
        )

    # Transform user models into llama-stack based models
    if validated_payload.foundation_models is not None:
        fms_param = Parameter(
            name="foundation_model",
            values=[
                LSFoundationModel(
                    model_id=fm.model_id,
                    client=client,
                )
                for fm in validated_payload.foundation_models
            ],
        )
    else:
        fms_param = Parameter(
            name="foundation_model",
            values=default_foundation_models,
        )

    if validated_payload.embedding_models is not None:
        embedding_models_values = []
        for em in validated_payload.embedding_models:
            matched_model = next(filter(lambda x, _id=em.model_id: x.model_id == _id, default_embedding_models), None)
            if matched_model is None:
                raise SearchSpaceValueError(f"Embedding model '{em.model_id}' not found among available models.")
            embedding_models_values.append(
                LSEmbeddingModel(model_id=em.model_id, client=client, params=matched_model.params)
            )
        ems_param = Parameter(
            name="embedding_model",
            values=embedding_models_values,
        )
    else:
        ems_param = Parameter(
            name="embedding_model",
            values=default_embedding_models,
        )

    logger.info("Selected foundation models for the experiment: %s.", [m.model_id for m in fms_param.values])
    logger.info("Selected embedding models for the experiment: %s.", [m.model_id for m in ems_param.values])

    return AI4RAGSearchSpace(
        params=[fms_param, ems_param],
        vector_store_type=vector_store_type,
    )
