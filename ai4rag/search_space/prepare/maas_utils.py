# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from openai import OpenAI
from openai.types import Model

from ai4rag import logger
from ai4rag.components.utils.maas_client import create_maas_model_client, maas_model_base_url
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError


def _short_model_id(model_id: str) -> str:
    """Return the short, usable model id (last path segment).

    MaaS lists models with fully-qualified ids such as
    ``publishers/ai-eng-cracow/models/qwen3-8b-fp8-dynamic``; the segment used
    when calling ``chat``/``embeddings`` is the final one.
    """
    return model_id.rsplit("/", 1)[-1]


def _model_owned_by(model: Model) -> str:
    """Return the ``owned_by`` path prefix used to build the per-model endpoint.

    Falls back to deriving ``<namespace>/<short-id>`` from the model id when the
    ``owned_by`` attribute is missing, so a per-model URL can still be built.
    """
    owned_by = getattr(model, "owned_by", None)
    if owned_by:
        return owned_by
    # id shape: publishers/<namespace>/models/<short-id>
    parts = model.id.split("/")
    if len(parts) >= 4:
        return f"{parts[1]}/{parts[-1]}"
    return model.id


def _list_maas_models(client: OpenAI) -> dict[str, Model]:
    """List models available in MaaS, keyed by their short id.

    MaaS carries no metadata distinguishing foundation from embedding models, so
    this returns every listed model; the caller decides the type from the payload.

    Parameters
    ----------
    client : OpenAI
        General MaaS client pointing at the ``/maas-api/v1`` endpoint.

    Returns
    -------
    dict[str, Model]
        Mapping of short model id to its :class:`~openai.types.Model` object.
    """
    logger.info("Listing available models in MaaS...")
    available_models = client.models.list().data
    registry = {_short_model_id(model.id): model for model in available_models}
    logger.info("Found models in MaaS: %s.", list(registry))
    return registry


def _validate_foundation_model(model: OpenAIFoundationModel) -> bool:
    """
    Validate that a foundation model responds correctly with minimal tokens.

    Parameters
    ----------
    model : OpenAIFoundationModel
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
            "Foundation model '%s' is available in MaaS, but does not respond.",
            model.model_id,
        )
        return False


def _validate_embedding_model(model: OpenAIEmbeddingModel) -> bool:
    """
    Validate that an embedding model responds correctly with minimal input.

    Parameters
    ----------
    model : OpenAIEmbeddingModel
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
            "Embedding model '%s' is available in MaaS, but does not respond.",
            model.model_id,
        )
        return False


def _build_valid_and_invalid_models(
    candidate_ids: list[str],
    registered_models_as_dict: dict[str, Model],
    models_type: str,
    client: OpenAI,
) -> tuple[list[OpenAIFoundationModel | OpenAIEmbeddingModel], list[str]]:
    """Instantiate and validate models, each behind its own per-model MaaS client.

    Since MaaS exposes no model metadata, embedding parameters (dimension and
    context length) are always auto-detected by the model itself.

    Parameters
    ----------
    candidate_ids : list[str]
        Short model ids to build and validate.
    registered_models_as_dict : dict[str, Model]
        Registry of available MaaS models keyed by short id.
    models_type : str
        ``'llm'`` or ``'embedding'``.
    client : OpenAI
        General MaaS client, used only for its base URL and API key when
        deriving per-model endpoints.

    Returns
    -------
    tuple[list[OpenAIFoundationModel | OpenAIEmbeddingModel], list[str]]
        A (valid model instances, invalid model ids) pair.
    """
    valid_model_instances: list[OpenAIFoundationModel | OpenAIEmbeddingModel] = []
    invalid_model_ids: list[str] = []
    for model_id in candidate_ids:
        owned_by = _model_owned_by(registered_models_as_dict[model_id])
        per_model_client = create_maas_model_client(
            base_url=maas_model_base_url(client.base_url, owned_by),
            api_key=client.api_key,
        )
        if models_type == "embedding":
            try:
                # Params are auto-detected by the model (MaaS carries no metadata).
                # If detection fails, the model is not usable.
                _model = OpenAIEmbeddingModel(model_id=model_id, client=per_model_client)
                is_valid = _validate_embedding_model(_model)
            except RuntimeError:
                logger.warning(
                    "Embedding model '%s' is available in MaaS, but does not respond.", model_id, exc_info=True
                )
                invalid_model_ids.append(model_id)
                continue
        else:
            _model = OpenAIFoundationModel(model_id=model_id, client=per_model_client)
            is_valid = _validate_foundation_model(_model)

        if is_valid:
            valid_model_instances.append(_model)
        else:
            invalid_model_ids.append(model_id)

    return valid_model_instances, invalid_model_ids


def _validate_availability_and_create_models(
    registered_models: dict[str, Model],
    models_type: str,
    client: OpenAI,
    provided_models_ids: list[str],
) -> list[OpenAIFoundationModel | OpenAIEmbeddingModel]:
    """
    Validate that the requested models are available and responding, then instantiate them.

    Parameters
    ----------
    registered_models : dict[str, Model]
        Registry of available MaaS models keyed by short id, from
        :func:`_list_maas_models`.

    models_type : str
        ``'llm'`` or ``'embedding'``.

    client : OpenAI
        General MaaS client used to derive per-model endpoints.

    provided_models_ids : list[str]
        Short model ids requested by the user. Required: MaaS metadata cannot
        distinguish model type, so the type is inferred from which payload list
        the id appears in.

    Returns
    -------
    list[OpenAIFoundationModel | OpenAIEmbeddingModel]
        Validated and instantiated models.

    Raises
    ------
    SearchSpaceValueError
        When some of the requested models are not available in MaaS or do not respond.
    """
    error_messages = []

    provided_not_available_models_ids = [pm_id for pm_id in provided_models_ids if pm_id not in registered_models]
    if provided_not_available_models_ids:
        error_messages.append(
            f"Provided models of type '{models_type}' are not available in MaaS: "
            f"'{provided_not_available_models_ids}'."
        )
    candidate_ids = [pm_id for pm_id in provided_models_ids if pm_id in registered_models]

    valid_model_instances, invalid_model_ids = _build_valid_and_invalid_models(
        candidate_ids, registered_models, models_type, client
    )

    if invalid_model_ids:
        error_messages.append(
            f"Provided models of type '{models_type}' are available in MaaS but do not respond. "
            f"Please validate these models are correctly deployed and respond: '{invalid_model_ids}'."
        )

    if error_messages:
        raise SearchSpaceValueError("\n".join(error_messages))

    return valid_model_instances
