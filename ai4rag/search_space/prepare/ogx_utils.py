# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import TypedDict

from ogx_client import OgxClient
from ogx_client.types import Model

from ai4rag import logger
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams
from ai4rag.rag.foundation_models.ogx import OGXFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


class _DefaultModelsResponseType(TypedDict):
    foundation_models: list[Model]
    embedding_models: list[Model]


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
        model.create_response(user_message="Hi")
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning(
            "Foundation model '%s' is registered in OGX, but does not respond.",
            model.model_id,
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
            "Embedding model '%s' is registered in OGX, but does not respond.",
            model.model_id,
        )
        return False


def _get_default_ogx_models(client: OgxClient) -> _DefaultModelsResponseType:
    """Return registered foundation and embedding model objects from OGX without validating them."""

    logger.info("Checking registered embedding and foundation models...")
    available_models = client.models.list().data

    registered_foundation_models = [
        model
        for model in available_models
        if model.custom_metadata.get("model_type") == "llm"
    ]
    registered_embedding_models = [
        model
        for model in available_models
        if model.custom_metadata.get("model_type") == "embedding"
    ]

    if not registered_foundation_models:
        raise SearchSpaceValueError(
            "There are no registered models of type 'llm' in the OGX."
        )
    if not registered_embedding_models:
        raise SearchSpaceValueError(
            "There are no registered models of type 'embedding' in the OGX."
        )

    logger.info(
        "Found registered foundation models: %s.",
        [model.id for model in registered_foundation_models],
    )
    logger.info(
        "Found registered embedding models: %s.",
        [model.id for model in registered_embedding_models],
    )

    return {
        "foundation_models": registered_foundation_models,
        "embedding_models": registered_embedding_models,
    }


def _build_valid_and_invalid_models(
    candidate_ids: list[str],
    registered_models_as_dict: dict,
    models_type: str,
    client: OgxClient,
) -> tuple[list[OGXFoundationModel | OGXEmbeddingModel], list[str]]:
    valid_model_instances: list[OGXFoundationModel | OGXEmbeddingModel] = []
    invalid_model_ids: list[str] = []
    for model_id in candidate_ids:
        if models_type == "embedding":
            custom_metadata = registered_models_as_dict[model_id].custom_metadata
            try:
                # If params is not provided model is trying to estimate parameters by sending some queries.
                # If it fails it means that model is not available
                embedding_dimension = (
                    custom_metadata.get("embedding_dimension", None)
                    if custom_metadata
                    else None
                )
                context_length = (
                    custom_metadata.get("context_length", None)
                    if custom_metadata
                    else None
                )
                _model = OGXEmbeddingModel(
                    model_id=model_id,
                    client=client,
                    params=OGXEmbeddingParams(
                        embedding_dimension=embedding_dimension,
                        context_length=context_length,
                    ),
                )
                is_valid = _validate_embedding_model(_model)
            except RuntimeError:
                logger.warning(
                    "Embedding model '%s' is registered in OGX, but does not respond.",
                    model_id,
                    exc_info=True,
                )
                invalid_model_ids.append(model_id)
                continue
        else:
            _model = OGXFoundationModel(model_id=model_id, client=client)
            is_valid = _validate_foundation_model(_model)

        if is_valid:
            valid_model_instances.append(_model)
        else:
            invalid_model_ids.append(model_id)

    return valid_model_instances, invalid_model_ids


def _validate_availability_and_create_models(
    registered_models: list[Model],
    models_type: str,
    client: OgxClient,
    provided_models_ids: list[str] | None = None,
) -> list[OGXFoundationModel | OGXEmbeddingModel]:
    """
    Validate that the requested models are registered and responding, then instantiate them.

    Parameters
    ----------
    registered_models : list[Model]
        OGX model objects returned by ``client.models.list().data``, pre-filtered by type.

    models_type : str
        ``'llm'`` or ``'embedding'``.

    client : OgxClient
        OGX client used to instantiate model objects.

    provided_models_ids : list[str] | None, default None
        Model IDs requested by the user.  When ``None``, all registered models
        are validated (the registration check is skipped since the IDs come from
        the registry itself).

    Returns
    -------
    list[OGXFoundationModel | OGXEmbeddingModel]
        Validated and instantiated models.

    Raises
    ------
    SearchSpaceValueError
        When some of the requested models are not registered in OGX or do not respond.
    """
    error_messages = []
    registered_models_as_dict = {m.id: m for m in registered_models}

    if provided_models_ids is None:
        candidate_ids = list(registered_models_as_dict)
    else:
        provided_not_registered_models_ids = [
            pm_id
            for pm_id in provided_models_ids
            if pm_id not in registered_models_as_dict
        ]
        if provided_not_registered_models_ids:
            error_messages.append(
                f"Provided models of type '{models_type}' are not registered in OGX: "
                f"'{provided_not_registered_models_ids}'."
            )
        candidate_ids = [
            pm_id for pm_id in provided_models_ids if pm_id in registered_models_as_dict
        ]

    valid_model_instances, invalid_model_ids = _build_valid_and_invalid_models(
        candidate_ids, registered_models_as_dict, models_type, client
    )

    if invalid_model_ids:
        error_messages.append(
            f"Provided models of type '{models_type}' are registered in OGX but do not respond. "
            f"Please validate these models are correctly registered and respond: '{invalid_model_ids}'."
        )

    if error_messages:
        raise SearchSpaceValueError("\n".join(error_messages))

    return valid_model_instances
