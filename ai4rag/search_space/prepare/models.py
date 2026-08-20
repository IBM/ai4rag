# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Instantiate OpenAI-compatible RAG models from a serving client.

This module is the single place that turns *model identifiers* (or serialized
model specs) into ready-to-use :class:`OpenAIFoundationModel` /
:class:`OpenAIEmbeddingModel` instances. A **single** :class:`~openai.OpenAI`
client serves everything — it lists models and serves ``chat.completions`` and
``embeddings`` for every model at the same endpoint — so all instantiated
wrappers share that one client. Model ids are used **verbatim**, exactly as
returned by ``models.list()`` (including any ``/`` characters).

Two complementary modes are exposed through the same
:func:`get_foundation_models` / :func:`get_embedding_models` entry points:

* **Discovery** — given plain model ids, each id is checked against the serving
  registry, the model is instantiated on the shared client and, by default,
  validated for responsiveness (embedding dimension/context length are
  auto-detected against the live endpoint). This backs the search-space
  preparation step and the generated notebooks.
* **Restore** — given serialized specs from a search-space report (each carrying
  its ``model_id``, inference ``params``, detected ``language`` and prompt
  templates), models are rebuilt verbatim, reusing every stored parameter. No
  registry listing or re-detection is performed. This backs the optimization run.

This code lives in :mod:`ai4rag.search_space.prepare` — alongside its primary
consumer, :func:`prepare_search_space_with_maas` — because building the models
that populate a search space *is* search-space preparation. The optimization
component imports the same helpers directly for its restore path.

Any OpenAI-compatible serving stack works: pass an :class:`~openai.OpenAI`
client and the ids it serves. Callers that need bespoke behaviour can
instantiate the model wrappers directly instead of going through these helpers.
"""

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from typing import Any

from openai import OpenAI

from ai4rag import logger
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel, Language
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.search_space.src.exceptions import SearchSpaceValueError

__all__ = ["get_embedding_models", "get_foundation_models", "serialize_model"]

# Discriminators for the two supported model families. Kept as plain strings so
# they can flow straight into the user-facing error messages below.
_FOUNDATION = "foundation"
_EMBEDDING = "embedding"

# A model entry is either a bare id (discovery mode) or a serialized spec
# mapping produced by the search-space report (restore mode).
ModelEntry = str | Mapping[str, Any]


def _list_maas_model_ids(client: OpenAI) -> set[str]:
    """List the ids of every model available on the serving client.

    Ids are returned **verbatim** (exactly as ``models.list()`` reports them,
    including any ``/`` characters), since those are the ids passed to
    ``chat``/``embeddings``. MaaS carries no metadata distinguishing foundation
    from embedding models, so every listed model is returned; the caller decides
    the type by which helper it invokes.

    Parameters
    ----------
    client : OpenAI
        Serving client pointing at the OpenAI-compatible model-list endpoint.

    Returns
    -------
    set[str]
        The full ids of every available model.
    """
    logger.info("Listing available models on the serving endpoint...")
    model_ids = {model.id for model in client.models.list().data}
    logger.info("Found models: %s.", sorted(model_ids))
    return model_ids


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


def _instantiate(
    spec: dict[str, Any],
    client: OpenAI,
    model_type: str,
) -> OpenAIFoundationModel | OpenAIEmbeddingModel:
    """Build a single model wrapper from its spec on the shared serving client.

    Stored ``params`` are passed through verbatim; for embedding models this
    means auto-detection (dimension/context length) is skipped when the values
    were restored from a report, and performed only when they are absent.
    """
    model_id = spec["model_id"]
    params = spec.get("params") or None

    if model_type == _EMBEDDING:
        return OpenAIEmbeddingModel(client=client, model_id=model_id, params=params)

    language = Language(**spec["language"]) if spec.get("language") else None
    return OpenAIFoundationModel(
        client=client,
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

    # Record the discovery flag *before* normalization: a bare id string is a
    # discovery request (verify availability against the registry), whereas a
    # spec mapping is a restore request (rebuilt verbatim). _normalize_entry
    # collapses both into a dict, so the distinction must be captured up front.
    parsed = [(isinstance(entry, str), _normalize_entry(entry)) for entry in models]
    if not parsed:
        return []

    # Only hit the serving registry when at least one id must be discovered.
    # Restore-only calls avoid the round-trip.
    needs_registry = any(is_discovery for is_discovery, _ in parsed)
    available_ids = _list_maas_model_ids(client) if needs_registry else set()

    error_messages: list[str] = []

    unavailable = [
        spec["model_id"] for is_discovery, spec in parsed if is_discovery and spec["model_id"] not in available_ids
    ]
    if unavailable:
        error_messages.append(
            f"Provided models of type '{model_type}' are not available on the serving endpoint: '{unavailable}'."
        )

    instances: list[OpenAIFoundationModel | OpenAIEmbeddingModel] = []
    not_responding: list[str] = []
    for _, spec in parsed:
        if spec["model_id"] in unavailable:
            continue
        try:
            model = _instantiate(spec, client, model_type)
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

    * a **model id** (``str``) — its availability is checked against the client's
      serving registry, the model is instantiated on the shared client, and (when
      *validate* is ``True``) checked for responsiveness; or
    * a **spec mapping** — a serialized search-space report entry carrying
      ``model_id``, ``params``, ``language`` and prompt templates; the model is
      restored verbatim without any registry round-trip.

    Ids are used exactly as provided (including any ``/`` characters).

    Parameters
    ----------
    client : OpenAI
        Authenticated OpenAI-compatible client shared by every instantiated
        model; it lists models and serves chat/embeddings for all of them.
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


def serialize_model(model: BaseFoundationModel | BaseEmbeddingModel) -> dict[str, Any]:
    """Serialize a model instance into a plain search-space-report spec.

    This is the inverse of the *restore* path above: the dict produced here is
    exactly what :func:`get_foundation_models` / :func:`get_embedding_models`
    consume when ``validate=False``. Keeping both directions in this module makes
    it the single home for the model↔spec round-trip that the search-space report
    relies on.

    Captures the model identifier, a type discriminator, its inference
    parameters, and — for foundation models — the detected language and prompt
    templates.

    Parameters
    ----------
    model : BaseFoundationModel | BaseEmbeddingModel
        The instantiated model to serialize.

    Returns
    -------
    dict[str, Any]
        A spec mapping carrying ``model_id``, ``type`` (``"embedding"`` or
        ``"generation"``) and ``params``; foundation models additionally carry
        ``language`` and the ``system_message_text`` / ``user_message_text`` /
        ``context_template_text`` prompt templates.
    """
    is_embedding = isinstance(model, BaseEmbeddingModel)

    params = model.params
    if is_dataclass(params):
        params_dict = {
            field.name: getattr(params, field.name)
            for field in fields(params)
            if getattr(params, field.name) is not None
        }
    elif hasattr(params, "model_dump"):
        params_dict = params.model_dump()
    elif hasattr(params, "dict"):
        params_dict = params.dict()
    else:
        params_dict = {}

    result: dict[str, Any] = {
        "model_id": model.model_id,
        "type": "embedding" if is_embedding else "generation",
        "params": params_dict,
    }

    if not is_embedding:
        if hasattr(model, "language") and model.language is not None:
            result["language"] = model.language.to_dict()
        result["system_message_text"] = model.system_message_text
        result["user_message_text"] = model.user_message_text
        result["context_template_text"] = model.context_template_text

    return result
