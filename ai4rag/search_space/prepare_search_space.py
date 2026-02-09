# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Any, TypedDict

from llama_stack_client import LlamaStackClient
from pydantic import TypeAdapter, ValidationError

from ai4rag import logger
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.search_space.prepare.input_payload_types import AI4RAGConstraints
from ai4rag.search_space.prepare.validation_error_decoder import validation_error_decoder
from ai4rag.search_space.src.exceptions import SearchSpaceValueError
from ai4rag.search_space.src.parameter import Parameter
from ai4rag.search_space.src.search_space import AI4RAGSearchSpace

__all__ = ["prepare_search_space_with_llama_stack"]


class _DefaultModelsResponseType(TypedDict):
    foundation_models: list[LSFoundationModel]
    embedding_models: list[LSEmbeddingModel]


def _get_default_llama_stack_models(client: LlamaStackClient) -> _DefaultModelsResponseType:
    """Get list of default foundation models based on the available ones in llama stack."""

    logger.info("Selecting default foundation models...")
    available_models = client.models.list()
    llms = [model for model in available_models if model.custom_metadata.get("model_type") == "llm"]
    embeddings = [model for model in available_models if model.custom_metadata.get("model_type") == "embedding"]
    foundation_models = [LSFoundationModel(model_id=m.id, client=client) for m in llms]
    embedding_models = [
        LSEmbeddingModel(
            model_id=m.id, client=client, params={"embedding_dimension": m.custom_metadata["embedding_dimension"]}
        )
        for m in embeddings
    ]

    if not foundation_models:
        raise SearchSpaceValueError("There are no available models of type 'llm'.")
    if not embedding_models:
        raise SearchSpaceValueError("There are no available models of type 'embedding'.")

    logger.info("Selected default foundation models: %s.", foundation_models)
    logger.info("Selected default embedding models: %s.", embedding_models)

    return {"foundation_models": foundation_models, "embedding_models": embedding_models}


def _are_provided_models_available(
    provided_models: list, available_models: list[LSFoundationModel | LSEmbeddingModel]
) -> bool:
    """Check whether models provided by the user are available for the experiment."""

    available_ids = [m.model_id for m in available_models]

    for model in provided_models:
        m_id = model.model_id
        if m_id not in available_ids:
            raise SearchSpaceValueError(f"Provided model with model_id: {m_id} is not available for the experiment.")
    return True


def prepare_search_space_with_llama_stack(payload: dict[str, Any], client: LlamaStackClient) -> AI4RAGSearchSpace:
    """
    Prepare AutoRAGSearchSpace.

    Parameters
    ----------
    payload : dict[str, Any]
        A mapping between parameter name and its associated values.

    client : LlamaStackClient
        Client instance for listing and validating available models.

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

    payload_model = TypeAdapter(AI4RAGConstraints)

    try:
        validated_payload = payload_model.validate_python(payload)
    except ValidationError as ve:
        # we want to catch only the first error
        validation_error_decoder(ve.errors()[0])

    if isinstance(client, LlamaStackClient):
        models = _get_default_llama_stack_models(client)
        default_foundation_models = models["foundation_models"]
        default_embedding_models = models["embedding_models"]
    else:
        raise SearchSpaceValueError(f"Unrecognized client type: {client.__class__.__name__}")

    if validated_payload.foundation_models:
        _are_provided_models_available(validated_payload.foundation_models, default_foundation_models)
    if validated_payload.embedding_models:
        _are_provided_models_available(validated_payload.embedding_models, default_embedding_models)

    # Transform user models into llama-stack based models
    if validated_payload.foundation_models is not None:
        fms_param = Parameter(
            name="foundation_model",
            param_type="C",
            values=[
                LSFoundationModel(
                    model_id=fm.model_id,
                    client=client,
                    model_params=fm.parameters.model_dump() if fm.parameters else {},
                )
                for fm in validated_payload.foundation_models
            ],
        )
    else:
        fms_param = Parameter(
            name="foundation_model",
            param_type="C",
            values=default_foundation_models,
        )

    if validated_payload.embedding_models is not None:
        ems_param = Parameter(
            name="embedding_model",
            param_type="C",
            values=[
                LSEmbeddingModel(
                    model_id=em.model_id, client=client, params=em.parameters.model_dump() if em.parameters else {}
                )
                for em in validated_payload.embedding_models
            ],
        )
    else:
        ems_param = Parameter(
            name="embedding_model",
            param_type="C",
            values=default_embedding_models,
        )

    return AI4RAGSearchSpace(
        params=[fms_param, ems_param],
    )
