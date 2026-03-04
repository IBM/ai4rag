# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import TypedDict

from llama_stack_client import LlamaStackClient

from ai4rag import logger
from ai4rag.rag.embedding.llama_stack import LSEmbeddingModel, LSEmbeddingParams
from ai4rag.rag.foundation_models.llama_stack import LSFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


class _DefaultModelsResponseType(TypedDict):
    foundation_models: list[LSFoundationModel]
    embedding_models: list[LSEmbeddingModel]


def _validate_foundation_model(model: LSFoundationModel) -> bool:
    """
    Validate that a foundation model responds correctly with minimal tokens.

    Parameters
    ----------
    model : LSFoundationModel
        Foundation model to validate.

    Returns
    -------
    bool
        True if model responds successfully, False otherwise.
    """
    try:
        # Test with minimal message and 1 token response
        test_messages = [{"role": "user", "content": "Hi"}]
        model.chat(messages=test_messages)
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Foundation model '%s' does not respond and will be excluded from search space.",
            model.model_id,
            exc_info=True,
        )
        return False


def _validate_embedding_model(model: LSEmbeddingModel) -> bool:
    """
    Validate that an embedding model responds correctly with minimal input.

    Parameters
    ----------
    model : LSEmbeddingModel
        Embedding model to validate.

    Returns
    -------
    bool
        True if model responds successfully, False otherwise.
    """
    try:
        # Test with minimal text
        model.embed_query("test")
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Embedding model '%s' does not respond and will be excluded fro msearch space.",
            model.model_id,
            exc_info=True,
        )
        return False


def _get_default_llama_stack_models(client: LlamaStackClient) -> _DefaultModelsResponseType:
    """Get list of default foundation models based on the available ones in llama stack."""

    logger.info("Selecting default foundation models...")
    available_models = client.models.list()
    llms = [model for model in available_models if model.custom_metadata.get("model_type") == "llm"]
    embeddings = [model for model in available_models if model.custom_metadata.get("model_type") == "embedding"]

    # Create model instances
    foundation_models_unvalidated = [LSFoundationModel(model_id=m.id, client=client) for m in llms]
    embedding_models_unvalidated = [
        LSEmbeddingModel(
            model_id=m.id,
            client=client,
            params=LSEmbeddingParams(
                embedding_dimension=m.custom_metadata.get("embedding_dimension")
                or m.metadata.get("embedding_dimension"),
                context_length=m.custom_metadata.get("context_length") or m.metadata.get("context_length"),
            ),
        )
        for m in embeddings
    ]

    # Validate each model
    logger.info("Validating foundation models...")
    foundation_models = [m for m in foundation_models_unvalidated if _validate_foundation_model(m)]

    logger.info("Validating embedding models...")
    embedding_models = [m for m in embedding_models_unvalidated if _validate_embedding_model(m)]

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
            raise SearchSpaceValueError(f"Provided model with model_id: '{m_id}' is not available for the experiment.")
    return True
