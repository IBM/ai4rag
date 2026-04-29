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
    not_responding_foundation_models: list[LSFoundationModel]
    not_responding_embedding_models: list[LSEmbeddingModel]


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
            "Embedding model '%s' does not respond and will be excluded from search space.",
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
            "There are no available models of type 'llm' or the ones registered are not responding. "
            "Please look at the full logs."
        )
    if not embedding_models:
        raise SearchSpaceValueError(
            "There are no available models of type 'embedding' or the ones registered are not responding. "
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
    provided_models: list,
    available_models: list[LSFoundationModel | LSEmbeddingModel],
    not_responding_models: list[LSFoundationModel | LSEmbeddingModel],
) -> None:
    """
    Check whether models provided by the user are available for the experiment.

    Parameters
    ----------
    provided_models : list
        Models provided by the user in the input payload.

    available_models : list[LSFoundationModel | LSEmbeddingModel]
        Models registered within llama-stack.

    not_responding_models : list[LSFoundationModel | LSEmbeddingModel]
        Models that are registered within llama-stack but do not respond.

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

    log = []
    if user_not_responding_models:
        log.append(
            f"Provided models: `{user_not_responding_models}` are registered but do not respond. "
            f"Remove these models from the experiment configuration and try again."
        )

    if user_unavailable_models:
        log.append(
            f"Provided models: `{user_unavailable_models}` are not registered within llama-stack. "
            f"Register these models or try different model for the experiment."
        )

    if log:
        raise SearchSpaceValueError("\n".join(log))
