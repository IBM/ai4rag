# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Instantiate OpenAI-compatible RAG models from a serving client.

This module is the single place that turns *model identifiers* (or serialized
model specs) into ready-to-use :class:`OpenAIFoundationModel` /
:class:`OpenAIEmbeddingModel` instances. It supports two complementary modes,
both exposed through the same :func:`get_foundation_models` /
:func:`get_embedding_models` entry points:

* **Discovery** — given a client and plain model ids, each model's endpoint is
  discovered from the serving registry (OpenShift MaaS lists one endpoint per
  model), the model is instantiated and, by default, validated for
  responsiveness. This backs the search-space preparation step and the
  generated notebooks.
* **Restore** — given serialized specs from a search-space report (each already
  carrying its per-model ``base_url``, inference ``params``, detected
  ``language`` and prompt templates), models are rebuilt verbatim, reusing every
  stored parameter and taking only the API key/token from the client. No
  registry listing or re-detection is performed. This backs the optimization run.

Although the endpoint-discovery path is tailored to OpenShift MaaS, any
OpenAI-compatible serving stack works: pass an :class:`~openai.OpenAI` client
and the ids it serves. Callers that need bespoke behaviour can instantiate the
model wrappers directly instead of going through these helpers.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from openai import OpenAI
from openai.types import Model

from ai4rag import logger
from ai4rag.components.utils.maas_client import create_maas_model_client, maas_model_base_url
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError

__all__ = ["get_embedding_models", "get_foundation_models"]

# Discriminators for the two supported model families. Kept as plain strings so
# they can flow straight into the user-facing error messages below.
_FOUNDATION = "foundation"
_EMBEDDING = "embedding"

# A model entry is either a bare id (discovery mode) or a serialized spec
# mapping produced by the search-space report (restore mode).
ModelEntry = str | Mapping[str, Any]


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
    """List models available on the serving client, keyed by their short id.

    MaaS carries no metadata distinguishing foundation from embedding models, so
    this returns every listed model; the caller decides the type by which helper
    it invokes.

    Parameters
    ----------
    client : OpenAI
        General serving client pointing at the model-list endpoint (for MaaS,
        ``/maas-api/v1``).

    Returns
    -------
    dict[str, Model]
        Mapping of short model id to its :class:`~openai.types.Model` object.
    """
    logger.info("Listing available models on the serving endpoint...")
    available_models = client.models.list().data
    registry = {_short_model_id(model.id): model for model in available_models}
    logger.info("Found models: %s.", list(registry))
    return registry


def _validate_foundation_model(model: OpenAIFoundationModel) -> bool:
    """Validate that a foundation model responds to a minimal chat request.

    Returns
    -------
    bool
        ``True`` if the model responds successfully, ``False`` otherwise.
    """
    try:
        model.chat(messages=[{"role": "user", "content": "Hi"}])
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning("Foundation model '%s' is available but does not respond.", model.model_id)
        return False


def _validate_embedding_model(model: OpenAIEmbeddingModel) -> bool:
    """Validate that an embedding model responds to a minimal embed request.

    Returns
    -------
    bool
        ``True`` if the model responds successfully, ``False`` otherwise.
    """
    try:
        model.embed_query("test")
        return True
    except Exception:  # pylint: disable=broad-exception-caught
        logger.warning("Embedding model '%s' is available but does not respond.", model.model_id)
        return False


def _normalize_entry(entry: ModelEntry) -> dict[str, Any]:
    """Coerce a model entry (id string or spec mapping) into a spec dict.

    A bare string is treated as a model id with no other stored settings; a
    mapping is copied and required to carry a ``model_id``.
    """
    if isinstance(entry, str):
        return {"model_id": entry}
    if isinstance(entry, Mapping):
        spec = dict(entry)
        if not spec.get("model_id"):
            raise SearchSpaceValueError("Model spec must include a non-empty 'model_id'.")
        return spec
    raise SearchSpaceValueError(f"Unsupported model entry type: '{type(entry).__name__}'.")


def _resolve_base_url(spec: dict[str, Any], client: OpenAI, registry: dict[str, Model]) -> str:
    """Resolve a model's per-endpoint base URL.

    In restore mode the spec already carries a ``base_url``; in discovery mode it
    is derived from the model's ``owned_by`` prefix in the serving registry.
    """
    base_url = spec.get("base_url")
    if base_url:
        return str(base_url)
    owned_by = _model_owned_by(registry[spec["model_id"]])
    return maas_model_base_url(client.base_url, owned_by)


def _instantiate(
    spec: dict[str, Any],
    per_model_client: OpenAI,
    model_type: str,
) -> OpenAIFoundationModel | OpenAIEmbeddingModel:
    """Build a single model wrapper from its spec behind an already-built client.

    Stored ``params`` are passed through verbatim; for embedding models this
    means auto-detection (dimension/context length) is skipped when the values
    were restored from a report, and performed only when they are absent.
    """
    model_id = spec["model_id"]
    params = spec.get("params") or None

    if model_type == _EMBEDDING:
        return OpenAIEmbeddingModel(client=per_model_client, model_id=model_id, params=params)

    language = Language(**spec["language"]) if spec.get("language") else None
    return OpenAIFoundationModel(
        client=per_model_client,
        model_id=model_id,
        params=params,
        language=language,
        system_message_text=spec.get("system_message_text"),
        user_message_text=spec.get("user_message_text"),
        context_template_text=spec.get("context_template_text"),
    )


def _validate(model: OpenAIFoundationModel | OpenAIEmbeddingModel, model_type: str) -> bool:
    """Dispatch to the responsiveness check for the model's family."""
    if model_type == _EMBEDDING:
        return _validate_embedding_model(model)
    return _validate_foundation_model(model)


def _get_models(
    client: OpenAI,
    models: Sequence[ModelEntry],
    *,
    model_type: str,
    validate: bool,
) -> list[OpenAIFoundationModel | OpenAIEmbeddingModel]:
    """Instantiate a homogeneous list of models, discovering or restoring each.

    See :func:`get_foundation_models` / :func:`get_embedding_models` for the
    public contract. Availability and responsiveness failures are accumulated
    and raised together so the caller sees every problem at once.
    """
    if not isinstance(client, OpenAI):
        raise SearchSpaceValueError(f"Unrecognized client type: '{type(client).__name__}'.")

    specs = [_normalize_entry(entry) for entry in models]
    if not specs:
        return []

    # Only hit the serving registry when at least one endpoint must be discovered
    # (i.e. a spec without a stored base_url). Restore-only calls avoid the round-trip.
    needs_registry = any(not spec.get("base_url") for spec in specs)
    registry = _list_maas_models(client) if needs_registry else {}

    error_messages: list[str] = []

    unavailable = [spec["model_id"] for spec in specs if not spec.get("base_url") and spec["model_id"] not in registry]
    if unavailable:
        error_messages.append(
            f"Provided models of type '{model_type}' are not available on the serving endpoint: '{unavailable}'."
        )

    instances: list[OpenAIFoundationModel | OpenAIEmbeddingModel] = []
    not_responding: list[str] = []
    for spec in specs:
        if spec["model_id"] in unavailable:
            continue
        per_model_client = create_maas_model_client(
            base_url=_resolve_base_url(spec, client, registry),
            api_key=client.api_key,
        )
        try:
            model = _instantiate(spec, per_model_client, model_type)
        except RuntimeError:
            # Only reachable in discovery mode, where embedding params are
            # auto-detected against a live endpoint; a failure means the model
            # is deployed but not usable.
            logger.warning("Model '%s' (%s) is available but does not respond.", spec["model_id"], model_type)
            not_responding.append(spec["model_id"])
            continue

        if validate and not _validate(model, model_type):
            not_responding.append(spec["model_id"])
            continue
        instances.append(model)

    if not_responding:
        error_messages.append(
            f"Provided models of type '{model_type}' are available but do not respond. "
            f"Please validate these models are correctly deployed and respond: '{not_responding}'."
        )

    if error_messages:
        raise SearchSpaceValueError("\n".join(error_messages))

    return instances


def get_foundation_models(
    client: OpenAI,
    models: Sequence[ModelEntry],
    *,
    validate: bool = True,
) -> list[OpenAIFoundationModel]:
    """Instantiate foundation (generation) models from a serving client.

    Each entry in *models* is either:

    * a **model id** (``str``) — the model's endpoint is discovered from the
      client's serving registry, the model is instantiated, and (when *validate*
      is ``True``) checked for responsiveness; or
    * a **spec mapping** — a serialized search-space report entry carrying
      ``model_id``, ``base_url``, ``params``, ``language`` and prompt templates;
      the model is restored verbatim and only the client's API key/token is used.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI-compatible client. Its API key is reused for every
        per-model endpoint; its ``base_url`` is used only when an endpoint must
        be discovered.
    models : Sequence[str | Mapping[str, Any]]
        Model ids and/or serialized specs to instantiate.
    validate : bool, default=True
        When ``True``, each instantiated model is probed with a minimal chat
        request; non-responding models raise :class:`SearchSpaceValueError`.
        Restore calls typically pass ``False`` to trust a previously validated
        report.

    Returns
    -------
    list[OpenAIFoundationModel]
        The instantiated foundation models, in input order.

    Raises
    ------
    SearchSpaceValueError
        When *client* is not an :class:`~openai.OpenAI` instance, a requested id
        is not available on the serving endpoint, or a model does not respond.
    """
    return _get_models(client, models, model_type=_FOUNDATION, validate=validate)  # type: ignore[return-value]


def get_embedding_models(
    client: OpenAI,
    models: Sequence[ModelEntry],
    *,
    validate: bool = True,
) -> list[OpenAIEmbeddingModel]:
    """Instantiate embedding models from a serving client.

    Behaves like :func:`get_foundation_models` but for embedding models. In
    discovery mode the embedding dimension and context length are auto-detected
    against the live endpoint; in restore mode the stored ``params`` are reused,
    skipping that (network-bound) detection.

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI-compatible client (see :func:`get_foundation_models`).
    models : Sequence[str | Mapping[str, Any]]
        Model ids and/or serialized specs to instantiate.
    validate : bool, default=True
        When ``True``, each instantiated model is probed with a minimal embed
        request; non-responding models raise :class:`SearchSpaceValueError`.

    Returns
    -------
    list[OpenAIEmbeddingModel]
        The instantiated embedding models, in input order.

    Raises
    ------
    SearchSpaceValueError
        When *client* is not an :class:`~openai.OpenAI` instance, a requested id
        is not available on the serving endpoint, or a model does not respond.
    """
    return _get_models(client, models, model_type=_EMBEDDING, validate=validate)  # type: ignore[return-value]
