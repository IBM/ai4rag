# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import TypedDict

from ogx_client import OgxClient

from ai4rag import logger
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.search_space.prepare.input_payload_types import AI4RAGEmbeddingModel, AI4RAGFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


class _DefaultModelsResponseType(TypedDict):
    foundation_models: list[OGXFoundationModel]
    embedding_models: list[OGXEmbeddingModel]
    not_responding_foundation_models: list[OGXFoundationModel]
    not_responding_embedding_models: list[OGXEmbeddingModel]


def _validate_foundation_model(model: OGXFoundationModel) -> bool:
    """
    Validate that a foundation model responds correctly with minimal tokens.

    Parameters
    ----------
    model : OGXFoundationModel
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


def _validate_embedding_model(model: OGXEmbeddingModel) -> bool:
    """
    Validate that an embedding model responds correctly with minimal input.

    Parameters
    ----------
    model : OGXEmbeddingModel
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
            "Embedding model '%s' does not respond and will be excluded from search space.",
            model.model_id,
            exc_info=True,
        )
        return False


def _get_default_ogx_models(client: OgxClient) -> _DefaultModelsResponseType:
    """Get list of default foundation models based on the available ones in OGX."""

    logger.info("Selecting default foundation models...")
    available_models = client.models.list()
    llms = [model for model in available_models if model.custom_metadata.get("model_type") == "llm"]
    embeddings = [model for model in available_models if model.custom_metadata.get("model_type") == "embedding"]

    # Create model instances
    foundation_models_unvalidated = [OGXFoundationModel(model_id=m.id, client=client) for m in llms]
    embedding_models_unvalidated = [
        OGXEmbeddingModel(
            model_id=m.id,
            client=client,
            params=OGXEmbeddingParams(
                embedding_dimension=getattr(m, "custom_metadata", {}).get("embedding_dimension"),
                context_length=getattr(m, "custom_metadata", {}).get("context_length"),
            ),
        )
        for m in embeddings
    ]

    # Validate each model
    foundation_models = []
    not_responding_foundation_models = []
    logger.info("Validating foundation models...")
    for fm_el in foundation_models_unvalidated:
        if _validate_foundation_model(fm_el):
            foundation_models.append(fm_el)
        else:
            not_responding_foundation_models.append(fm_el)

    embedding_models = []
    not_responding_embedding_models = []
    logger.info("Validating embedding models...")
    for em_el in embedding_models_unvalidated:
        if _validate_embedding_model(em_el):
            embedding_models.append(em_el)
        else:
            not_responding_embedding_models.append(em_el)

    if not foundation_models:
        raise SearchSpaceValueError(
            "There are no available models of type 'llm' or the ones registered are not responding: "
            f"{[m.model_id for m in not_responding_foundation_models]}. "
            "Please look at the full logs."
        )
    if not embedding_models:
        raise SearchSpaceValueError(
            "There are no available models of type 'embedding' or the ones registered are not responding: "
            f"{[m.model_id for m in not_responding_embedding_models]}. "
            "Please look at the full logs."
        )

    logger.info("Available foundation models: %s.", foundation_models)
    logger.info("Available embedding models: %s.", embedding_models)

    return {
        "foundation_models": foundation_models,
        "embedding_models": embedding_models,
        "not_responding_foundation_models": not_responding_foundation_models,
        "not_responding_embedding_models": not_responding_embedding_models,
    }


def _are_provided_models_available(
    provided_models: list[AI4RAGFoundationModel] | list[AI4RAGEmbeddingModel],
    available_models: list[OGXFoundationModel | OGXEmbeddingModel],
    not_responding_models: list[OGXFoundationModel | OGXEmbeddingModel],
) -> None:
    """
    Check whether models provided by the user are available for the experiment.

    Parameters
    ----------
    provided_models : list[AI4RAGFoundationModel] | list[AI4RAGEmbeddingModel]
        Models provided by the user in the input payload.

    available_models : list[OGXFoundationModel | OGXEmbeddingModel]
        Models registered within OGX that passed validation (respond to requests).

    not_responding_models : list[OGXFoundationModel | OGXEmbeddingModel]
        Models that are registered within OGX but do not respond.

    Raises
    ------
    SearchSpaceValueError
        When some of the models provided by the user are not available for the experiment
        or some of the models do not respond.
    """

    available_model_ids = [m.model_id for m in available_models]
    not_responding_model_ids = [m.model_id for m in not_responding_models]

    user_not_responding_models = [m.model_id for m in provided_models if m.model_id in not_responding_model_ids]
    user_unavailable_models = [
        m.model_id
        for m in provided_models
        if m.model_id not in available_model_ids and m.model_id not in not_responding_model_ids
    ]

    error_messages = []
    if user_not_responding_models:
        error_messages.append(
            f"Provided models: {user_not_responding_models} are registered but do not respond. "
            "Remove these models from the experiment configuration and try again."
        )

    if user_unavailable_models:
        error_messages.append(
            f"Provided models: {user_unavailable_models} are not registered within OGX. "
            "Register these models or try a different model for the experiment."
        )

    if error_messages:
        raise SearchSpaceValueError("\n".join(error_messages))
