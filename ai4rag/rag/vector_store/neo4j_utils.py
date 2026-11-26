# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Private adapters and helpers used by `ai4rag.rag.vector_store.neo4j`."""

import asyncio
import copy
import hashlib
import json
import random
from collections.abc import Mapping
from threading import Lock
from time import perf_counter, sleep
from typing import Any

import neo4j
from neo4j_graphrag.components.types import LexicalGraphConfig, Neo4jGraph
from neo4j_graphrag.embeddings.base import Embedder as _NeoEmbedder
from neo4j_graphrag.llm.base import LLMInterfaceV2 as _NeoLLMInterface
from neo4j_graphrag.llm.types import LLMResponse
from neo4j_graphrag.types import LLMMessage
from openai import APIConnectionError

from ai4rag import logger
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.vector_store.utils import validate_search_params


class Neo4jGraphSchema:
    """Names used by ai4rag's persistent Neo4j graph schema."""

    DOCUMENT_LABEL = "Document"
    CHUNK_LABEL = "Chunk"
    ENTITY_LABEL = "__Entity__"

    CONTAINS_RELATIONSHIP = "CONTAINS"
    FROM_CHUNK_RELATIONSHIP = "FROM_CHUNK"

    ID = "id"
    TEXT = "text"
    EMBEDDING = "embedding"
    DOCUMENT_ID = "document_id"
    SEQUENCE_NUMBER = "sequence_number"
    METADATA = "metadata"
    COLLECTION = "collection"
    SOURCE = "source"
    KG_COLLECTIONS = "ai4rag_kg_collections"
    ENTITY_KEY = "ai4rag_entity_key"


_SCHEMA = Neo4jGraphSchema

# One statement per flush; rows are already deduplicated before batching.
_KG_WRITE_BATCH_SIZE = 1000


# Keep KG extraction requests short enough to avoid gateway timeouts for
# constrained graph generation.
DEFAULT_KG_EXTRACTION_MAX_COMPLETION_TOKENS = 1024
# SDK retries cover short-lived blips; these application-level retries absorb
# gateway rate-limit windows and transient 5xx responses between them.
DEFAULT_KG_LLM_MAX_RETRIES = 4
DEFAULT_KG_LLM_RETRY_BASE_SECONDS = 1.0
# KG extraction owns the retry policy end to end. Leaving the OpenAI SDK's own
# retries enabled would multiply both layers (5 requests x 3 attempts) and hide
# those attempts from the metrics and the rate-limit breaker.
DEFAULT_KG_LLM_SDK_MAX_RETRIES = 0
# Abort instead of issuing hundreds of futile requests once the endpoint clearly
# rejects everything, e.g. when the model subscription token quota is exhausted.
DEFAULT_KG_RATE_LIMIT_BREAKER_THRESHOLD = 12
KG_LLM_RETRY_MAX_BACKOFF_SECONDS = 60.0
# Small caps keep a single extraction response well inside the completion
# budget, so a dense chunk cannot spend it all on entities and truncate the JSON.
KG_MAX_ENTITIES_PER_CHUNK = 5
KG_MAX_RELATIONSHIPS_PER_CHUNK = 5
# Descriptions are optional and only used when the selected schema requires
# them; keep them to a short phrase so the JSON stays compact.
KG_MAX_DESCRIPTION_WORDS = 8
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 425, 429})
# MaaS does not call tools on our behalf, so extraction cannot rely on
# function calling. The contract is a required JSON Schema instead: every
# field is listed in ``required`` and every object forbids extra properties,
# which lets structured-output-capable endpoints enforce the payload.
_KG_NODE_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "label": {"type": "string"},
        "properties": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    "required": ["id", "label", "properties"],
    "additionalProperties": False,
}
_KG_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "start_node_id": {"type": "string"},
        "end_node_id": {"type": "string"},
        "type": {"type": "string"},
    },
    "required": ["start_node_id", "end_node_id", "type"],
    "additionalProperties": False,
}
_KG_BATCH_ARGUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "chunks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "nodes": {"type": "array", "items": _KG_NODE_SCHEMA},
                    "relationships": {"type": "array", "items": _KG_RELATIONSHIP_SCHEMA},
                },
                "required": ["chunk_id", "nodes", "relationships"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chunks"],
    "additionalProperties": False,
}
KG_BATCH_RESPONSE_FORMAT = {"type": "json_object"}
KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "kg_batch",
        "strict": True,
        "schema": _KG_BATCH_ARGUMENTS_SCHEMA,
    },
}
_KG_RESPONSE_FORMATS = frozenset({"json_object", "json_schema"})

# GraphRAG's lexical nodes are temporary: _CanonicalKGWriter discards them in
# favor of the canonical nodes already stored by add_documents. Keep their
# labels distinct from entity types chosen by free extraction (e.g. Document).
_KG_LEXICAL_GRAPH_CONFIG = LexicalGraphConfig(
    document_node_label="__AI4RAG_KG_SOURCE_DOCUMENT__",
    chunk_node_label="__AI4RAG_KG_SOURCE_CHUNK__",
)


def _collection_vector_index_name(collection_name: str) -> str:
    """Return the vector-index name dedicated to *collection_name*."""
    return f"{collection_name}__embedding"


def _http_status_code(error: BaseException) -> int | None:
    """Return the HTTP status carried by an SDK or provider exception, when present."""
    status = getattr(error, "status_code", None)
    if not isinstance(status, int) or isinstance(status, bool):
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _retry_after_seconds(error: BaseException) -> float | None:
    """Return the provider's numeric ``Retry-After`` hint, when one is present."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        seconds = float(value)
    except (AttributeError, TypeError, ValueError):
        return None
    return max(0.0, seconds)


def is_rate_limit_llm_error(error: BaseException) -> bool:
    """Return whether *error* is a provider-side rate limit (HTTP 429)."""
    return _http_status_code(error) == 429


def is_retryable_llm_error(error: BaseException) -> bool:
    """Return whether a failed LLM call is worth retrying.

    Rate limits, transient server errors, and transport failures are retryable;
    client errors such as a malformed request are not.
    """
    if isinstance(error, APIConnectionError):
        return True
    status = _http_status_code(error)
    if status is None:
        return False
    return status in _RETRYABLE_HTTP_STATUS_CODES or status >= 500


def kg_retry_delay_seconds(attempt: int, error: BaseException, base_seconds: float) -> float:
    """Return the backoff before retrying a KG request.

    A numeric ``Retry-After`` header takes precedence, capped at
    :data:`KG_LLM_RETRY_MAX_BACKOFF_SECONDS`. Otherwise the delay grows
    exponentially with jitter so concurrent workers do not retry in lockstep.
    """
    retry_after = _retry_after_seconds(error)
    if retry_after is not None:
        return min(retry_after, KG_LLM_RETRY_MAX_BACKOFF_SECONDS)
    ceiling = min(base_seconds * (2**attempt), KG_LLM_RETRY_MAX_BACKOFF_SECONDS)
    return random.uniform(base_seconds * 0.5, ceiling)


def _model_with_sdk_retries(model: BaseFoundationModel, retries: int) -> BaseFoundationModel:
    """Return a shallow model view whose OpenAI client uses *retries* SDK retries.

    The original model is left untouched because generation shares the same
    instance. Clients without ``with_options`` are returned unchanged. The
    lookup uses the class rather than the instance so duck-typed wrappers are
    handled while generic mocks keep passing through.
    """
    client = getattr(model, "client", None)
    with_options = getattr(type(client), "with_options", None)
    if not callable(with_options):
        return model
    view = copy.copy(model)
    view.client = with_options(client, max_retries=retries)
    return view


def _validate_kg_extraction_config(  # pylint: disable=too-many-branches,too-many-locals
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate KG extraction settings and apply the constrained default."""
    if config is None:
        return {
            "mode": "constrained",
            "max_completion_tokens": DEFAULT_KG_EXTRACTION_MAX_COMPLETION_TOKENS,
            "response_format": "json_schema",
            "max_skipped_chunks": 0,
            "partial_failure_policy": "degrade",
            "max_retries": DEFAULT_KG_LLM_MAX_RETRIES,
            "retry_base_seconds": DEFAULT_KG_LLM_RETRY_BASE_SECONDS,
            "sdk_max_retries": DEFAULT_KG_LLM_SDK_MAX_RETRIES,
            "rate_limit_breaker_threshold": DEFAULT_KG_RATE_LIMIT_BREAKER_THRESHOLD,
            "fail_on_empty_kg": True,
        }
    if not isinstance(config, dict):
        raise TypeError("kg_extraction_config must be a dictionary or None.")

    mode = config.get("mode", "constrained")
    if mode not in {"constrained", "free"}:
        raise ValueError("kg_extraction_config.mode must be 'constrained' or 'free'.")
    max_completion_tokens = config.get("max_completion_tokens", DEFAULT_KG_EXTRACTION_MAX_COMPLETION_TOKENS)
    if (
        isinstance(max_completion_tokens, bool)
        or not isinstance(max_completion_tokens, int)
        or max_completion_tokens < 1
    ):
        raise ValueError("kg_extraction_config.max_completion_tokens must be a positive integer.")
    response_format = config.get("response_format", "json_schema")
    if response_format not in _KG_RESPONSE_FORMATS:
        raise ValueError("kg_extraction_config.response_format must be 'json_object' or 'json_schema'.")
    max_skipped_chunks = config.get("max_skipped_chunks", 0)
    if isinstance(max_skipped_chunks, bool) or not isinstance(max_skipped_chunks, int) or max_skipped_chunks < 0:
        raise ValueError("kg_extraction_config.max_skipped_chunks must be a non-negative integer.")
    partial_failure_policy = config.get("partial_failure_policy", "degrade")
    if partial_failure_policy not in {"degrade", "raise"}:
        raise ValueError("kg_extraction_config.partial_failure_policy must be 'degrade' or 'raise'.")
    max_retries = config.get("max_retries", DEFAULT_KG_LLM_MAX_RETRIES)
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("kg_extraction_config.max_retries must be a non-negative integer.")
    retry_base_seconds = config.get("retry_base_seconds", DEFAULT_KG_LLM_RETRY_BASE_SECONDS)
    if (
        isinstance(retry_base_seconds, bool)
        or not isinstance(retry_base_seconds, (int, float))
        or retry_base_seconds <= 0
    ):
        raise ValueError("kg_extraction_config.retry_base_seconds must be a positive number.")
    sdk_max_retries = config.get("sdk_max_retries", DEFAULT_KG_LLM_SDK_MAX_RETRIES)
    if isinstance(sdk_max_retries, bool) or not isinstance(sdk_max_retries, int) or sdk_max_retries < 0:
        raise ValueError("kg_extraction_config.sdk_max_retries must be a non-negative integer.")
    rate_limit_breaker_threshold = config.get("rate_limit_breaker_threshold", DEFAULT_KG_RATE_LIMIT_BREAKER_THRESHOLD)
    if (
        isinstance(rate_limit_breaker_threshold, bool)
        or not isinstance(rate_limit_breaker_threshold, int)
        or rate_limit_breaker_threshold < 1
    ):
        raise ValueError("kg_extraction_config.rate_limit_breaker_threshold must be a positive integer.")
    fail_on_empty_kg = config.get("fail_on_empty_kg", True)
    if not isinstance(fail_on_empty_kg, bool):
        raise ValueError("kg_extraction_config.fail_on_empty_kg must be a boolean.")
    common = {
        "mode": mode,
        "max_completion_tokens": max_completion_tokens,
        "response_format": response_format,
        "max_skipped_chunks": max_skipped_chunks,
        "partial_failure_policy": partial_failure_policy,
        "max_retries": max_retries,
        "retry_base_seconds": float(retry_base_seconds),
        "sdk_max_retries": sdk_max_retries,
        "rate_limit_breaker_threshold": rate_limit_breaker_threshold,
        "fail_on_empty_kg": fail_on_empty_kg,
    }
    if mode == "constrained":
        return common

    limits = {
        "max_entities_per_chunk": config.get("max_entities_per_chunk"),
        "max_relationships_per_chunk": config.get("max_relationships_per_chunk"),
    }
    for name, value in limits.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"kg_extraction_config.{name} must be a positive integer for free extraction.")
    return {**common, **limits}


class _EmbedderAdapter(_NeoEmbedder):  # type: ignore[misc]
    """Wraps :class:`BaseEmbeddingModel` to satisfy ``neo4j_graphrag``'s embedder interface."""

    def __init__(self, model: BaseEmbeddingModel, cached_embeddings: Mapping[str, list[float]] | None = None) -> None:
        super().__init__()
        self._model = model
        self._cached_embeddings = cached_embeddings or {}
        self._model_calls = 0
        self._cache_hits = 0
        self._elapsed_seconds = 0.0
        self._metrics_lock = Lock()

    def embed_query(self, text: str) -> list[float]:
        cached = self._cached_embeddings.get(text)
        if cached is not None:
            with self._metrics_lock:
                self._cache_hits += 1
            return cached

        started_at = perf_counter()
        result = self._model.embed_query(text)
        with self._metrics_lock:
            self._model_calls += 1
            self._elapsed_seconds += perf_counter() - started_at
        return result

    @property
    def metrics(self) -> tuple[int, int, float]:
        """Return model calls, cache hits, and cumulative model-call duration."""
        with self._metrics_lock:
            return self._model_calls, self._cache_hits, self._elapsed_seconds


def _kg_response_message(response: Any) -> Any:
    """Return the assistant message carried by a chat-completion choice or message."""
    if isinstance(response, Mapping):
        message = response.get("message")
        return message if message is not None else response
    message = getattr(response, "message", None)
    return message if message is not None else response


def _kg_message_content(message: Any) -> str:
    """Return a message's textual content, treating a missing payload as empty."""
    content = message.get("content") if isinstance(message, Mapping) else getattr(message, "content", None)
    return content if isinstance(content, str) else ""


class _LLMAdapter(_NeoLLMInterface):  # type: ignore[misc]  # pylint: disable=too-many-instance-attributes
    """Wraps :class:`BaseFoundationModel` to satisfy ``neo4j_graphrag``'s LLM interface.

    Implements GraphRAG's V2 list-of-messages contract and bridges the sync
    :meth:`BaseFoundationModel.chat` via a worker thread. Response JSON is
    normalised for common, schema-compatible model variants before parsing.
    The adapter owns the retry policy: the OpenAI client is wrapped so its own
    retries do not stack on top of :meth:`_chat_with_retries`.
    """

    def __init__(
        self,
        model: BaseFoundationModel,
        max_completion_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        max_retries: int = DEFAULT_KG_LLM_MAX_RETRIES,
        retry_base_seconds: float = DEFAULT_KG_LLM_RETRY_BASE_SECONDS,
        sdk_max_retries: int = DEFAULT_KG_LLM_SDK_MAX_RETRIES,
    ) -> None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer.")
        if (
            isinstance(retry_base_seconds, bool)
            or not isinstance(retry_base_seconds, (int, float))
            or retry_base_seconds <= 0
        ):
            raise ValueError("retry_base_seconds must be a positive number.")
        if isinstance(sdk_max_retries, bool) or not isinstance(sdk_max_retries, int) or sdk_max_retries < 0:
            raise ValueError("sdk_max_retries must be a non-negative integer.")
        super().__init__(model_name=model.model_id)
        self._model = _model_with_sdk_retries(model, sdk_max_retries)
        self._max_completion_tokens = max_completion_tokens
        self._response_format = dict(response_format) if response_format is not None else None
        self._max_retries = max_retries
        self._retry_base_seconds = float(retry_base_seconds)
        self._calls = 0
        self._elapsed_seconds = 0.0
        self._format_repairs = 0
        self._retries = 0
        self._failed_calls = 0
        self._metrics_lock = Lock()

    def invoke(
        self,
        input: list[LLMMessage],  # pylint: disable=redefined-builtin
        *,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Invoke the wrapped model using GraphRAG V2's message contract."""
        return self._invoke(
            input,
            max_completion_tokens=kwargs.pop("max_completion_tokens", self._max_completion_tokens),
            response_format=response_format,
            **kwargs,
        )

    def _invoke(
        self,
        input: list[LLMMessage],  # pylint: disable=redefined-builtin
        max_completion_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Make one request using the supplied completion limit."""
        model_kwargs = dict(kwargs)
        if max_completion_tokens is not None:
            model_kwargs["max_completion_tokens"] = max_completion_tokens
        effective_response_format = response_format if response_format is not None else self._response_format
        if effective_response_format is not None:
            model_kwargs["response_format"] = effective_response_format
        started_at = perf_counter()
        response = self._chat_with_retries(input, model_kwargs)
        with self._metrics_lock:
            self._elapsed_seconds += perf_counter() - started_at
        self._log_stop_reason(response)
        content = _kg_message_content(_kg_response_message(response))
        normalised_content = _normalise_kg_response(content)
        if normalised_content != content:
            with self._metrics_lock:
                self._format_repairs += 1
        return LLMResponse(content=normalised_content)

    def _chat_with_retries(  # pylint: disable=redefined-builtin
        self, input: list[LLMMessage], model_kwargs: Mapping[str, Any]
    ) -> Any:
        """Call the model, retrying rate limits and transient failures with backoff.

        Retries happen in the calling worker thread, so blocking sleep does not
        stall the event loop that awaits :meth:`ainvoke`. Every HTTP attempt is
        counted so the metrics reconcile with the traffic the endpoint sees.
        """
        attempts = self._max_retries + 1
        for attempt in range(attempts):
            with self._metrics_lock:
                self._calls += 1
            try:
                return self._model.chat(input, **model_kwargs)[0]
            except Exception as exc:  # pylint: disable=broad-except
                if attempt + 1 >= attempts or not is_retryable_llm_error(exc):
                    with self._metrics_lock:
                        self._failed_calls += 1
                    raise
                delay = kg_retry_delay_seconds(attempt, exc, self._retry_base_seconds)
                with self._metrics_lock:
                    self._retries += 1
                logger.warning(
                    "KG extraction LLM request failed with %s (status=%s); retrying in %.1fs "
                    "(attempt %d/%d, model=%s).",
                    type(exc).__name__,
                    _http_status_code(exc),
                    delay,
                    attempt + 1,
                    attempts,
                    self._model.model_id,
                )
                sleep(delay)
        raise RuntimeError("KG extraction retry loop exited without a response.")

    def _log_stop_reason(self, response: Any) -> None:
        """Log the model's finish reason without leaking response content."""
        stop_reason = (
            response.get("finish_reason") if isinstance(response, Mapping) else getattr(response, "finish_reason", None)
        )
        logger.info(
            "KG extraction LLM response received (model=%s, stop_reason=%s).",
            self._model.model_id,
            stop_reason if isinstance(stop_reason, (str, int, float)) else "unknown",
        )

    @property
    def metrics(self) -> tuple[int, float, int]:
        """Return model request attempts, cumulative duration, and repaired response count."""
        with self._metrics_lock:
            return self._calls, self._elapsed_seconds, self._format_repairs

    @property
    def retry_metrics(self) -> tuple[int, int]:
        """Return the number of retried requests and requests that exhausted retries."""
        with self._metrics_lock:
            return self._retries, self._failed_calls

    async def ainvoke(
        self,
        input: list[LLMMessage],  # pylint: disable=redefined-builtin
        *,
        response_format: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Asynchronously invoke the wrapped model using GraphRAG V2's contract."""
        return await asyncio.to_thread(self.invoke, input, response_format=response_format, **kwargs)

    async def ainvoke_with_completion_limit(
        self, input: str, max_completion_tokens: int  # pylint: disable=redefined-builtin
    ) -> LLMResponse:
        """Invoke the model with a request-specific completion-token limit."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._invoke_with_completion_limit, input, max_completion_tokens)

    def _invoke_with_completion_limit(
        self, input: str, max_completion_tokens: int  # pylint: disable=redefined-builtin
    ) -> LLMResponse:
        """Invoke with a request-specific output limit while retaining adapter metrics."""
        return self._invoke([{"role": "user", "content": input}], max_completion_tokens=max_completion_tokens)


# Keep malformed payloads untouched so GraphRAG can apply its configured error policy.
# pylint: disable-next=too-many-return-statements,too-many-branches
def _normalise_kg_response(content: str) -> str:
    """Convert common schema-compatible KG response variants to GraphRAG's JSON object.

    The extractor expects an object containing ``nodes`` and ``relationships``.
    Some OpenAI-compatible models instead return a fenced object, a singleton
    list containing that object, or use ``entities``/``relations`` aliases.
    Keep unparseable content unchanged so GraphRAG retains its normal error
    policy, and never log model output because it contains source content.
    """
    stripped = content.strip()
    recovered_embedded_object = False
    if stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped.split("\n", maxsplit=1)[-1].rsplit("```", maxsplit=1)[0].strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        # Some MaaS deployments add a short preamble despite JSON-object mode.
        # Recover only an already-valid, balanced object; never synthesize data.
        decoder = json.JSONDecoder()
        parsed = None
        for offset, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[offset:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                recovered_embedded_object = True
                break
        if parsed is None:
            return content

    if isinstance(parsed, list):
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            parsed = parsed[0]
        else:
            parsed = {"nodes": parsed, "relationships": []}
    if not isinstance(parsed, dict):
        return content
    if "chunks" in parsed:
        if isinstance(parsed["chunks"], list):
            return (
                content
                if not recovered_embedded_object and content == stripped and set(parsed) == {"chunks"}
                else json.dumps({"chunks": parsed["chunks"]})
            )
        return content

    nodes = parsed.get("nodes", parsed.get("entities", []))
    relationships = parsed.get("relationships", parsed.get("relations", []))
    if not isinstance(nodes, list) or not isinstance(relationships, list):
        return content
    if not recovered_embedded_object and content == stripped and set(parsed) == {"nodes", "relationships"}:
        return content
    return json.dumps({"nodes": nodes, "relationships": relationships})


class _CanonicalKGWriter:
    """Write extracted entities onto chunks already stored by ``add_documents``."""

    def __init__(self, driver: neo4j.Driver, database: str, collection_name: str) -> None:
        self.driver = driver
        self.neo4j_database = database
        self.batch_size = _KG_WRITE_BATCH_SIZE
        self._collection_name = collection_name
        self._write_elapsed_seconds = 0.0
        self._write_lock = Lock()

    @property
    def write_elapsed_seconds(self) -> float:
        """Return cumulative time spent writing KG graphs."""
        with self._write_lock:
            return self._write_elapsed_seconds

    def write_graphs(self, graphs: list[tuple[str, Neo4jGraph]]) -> None:
        """Persist extracted graphs with collection-scoped entity identities.

        The upstream writer creates temporary nodes for every extraction and
        resolves duplicates in a later APOC pass.  That makes indexing more
        expensive as the graph grows.  Instead, entity nodes are merged by a
        stable key composed from collection, type, and normalized name.  Nodes
        without a name remain local to their source chunk to avoid an unsafe
        merge.

        Rows are deduplicated before writing: the same entity usually appears
        in many chunks, and re-merging it row by row only costs write work.
        Duplicate node properties are merged with last-write-wins semantics.
        Each entity carries the chunks it was extracted from so the
        ``FROM_CHUNK`` links are written in the same transaction as the node.
        """
        node_rows: list[dict[str, Any]] = []
        relationship_rows: list[dict[str, Any]] = []

        for chunk_id, graph in graphs:
            entities = [
                node for node in graph.nodes if node.label not in _KG_LEXICAL_GRAPH_CONFIG.lexical_graph_node_labels
            ]
            key_by_id = {node.id: self._entity_key(chunk_id, node.id, node.label, node.properties) for node in entities}
            for node in entities:
                node_rows.append(
                    {
                        "key": key_by_id[node.id],
                        "labels": [node.label, _SCHEMA.ENTITY_LABEL],
                        "properties": node.properties,
                        "chunk_id": chunk_id,
                    }
                )
            relationship_rows.extend(
                {
                    "start_key": key_by_id[relationship.start_node_id],
                    "end_key": key_by_id[relationship.end_node_id],
                    "type": relationship.type,
                    "properties": relationship.properties,
                }
                for relationship in graph.relationships
                if relationship.start_node_id in key_by_id and relationship.end_node_id in key_by_id
            )

        node_rows = self._dedupe_nodes(node_rows)
        relationship_rows = self._dedupe_relationships(relationship_rows)

        started_at = perf_counter()
        for offset in range(0, len(node_rows), self.batch_size):
            self._write_entity_batch(node_rows[offset : offset + self.batch_size])
        for offset in range(0, len(relationship_rows), self.batch_size):
            self._write_relationship_batch(relationship_rows[offset : offset + self.batch_size])
        with self._write_lock:
            self._write_elapsed_seconds += perf_counter() - started_at

    @staticmethod
    def _dedupe_nodes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse rows that merge the same entity, keeping the first label set.

        The chunks that mentioned the entity are collected into ``chunk_ids``
        so one row writes both the node and all of its provenance links.
        """
        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            existing = merged.get(row["key"])
            if existing is None:
                merged[row["key"]] = {
                    "key": row["key"],
                    "labels": row["labels"],
                    "properties": dict(row["properties"]),
                    "chunk_ids": [row["chunk_id"]],
                }
            else:
                existing["properties"].update(row["properties"])
                if row["chunk_id"] not in existing["chunk_ids"]:
                    existing["chunk_ids"].append(row["chunk_id"])
        return list(merged.values())

    @staticmethod
    def _dedupe_relationships(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Collapse rows that merge the same relationship between two entities."""
        merged: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in rows:
            identity = (row["start_key"], row["end_key"], row["type"])
            existing = merged.get(identity)
            if existing is None:
                merged[identity] = {**row, "properties": dict(row["properties"])}
            else:
                existing["properties"].update(row["properties"])
        return list(merged.values())

    def _entity_key(self, chunk_id: str, node_id: str, label: str, properties: dict[str, Any]) -> str:
        name = properties.get("name")
        identity = (
            f"{self._collection_name}\x00{label.casefold()}\x00{' '.join(name.split()).casefold()}"
            if isinstance(name, str) and name.strip()
            else f"{self._collection_name}\x00{chunk_id}\x00{node_id}"
        )
        return hashlib.sha256(identity.encode()).hexdigest()

    def _write_entity_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.driver.session(database=self.neo4j_database) as session:
            session.execute_write(self._write_entity_batch_tx, rows, self._collection_name)

    @staticmethod
    def _write_entity_batch_tx(tx: neo4j.Transaction, rows: list[dict[str, Any]], collection_name: str) -> None:
        # Entities and their FROM_CHUNK provenance links share one transaction:
        # the links need no other input than the freshly merged node.
        tx.run(
            "UNWIND $rows AS row "
            "CALL apoc.merge.node(row.labels, {ai4rag_entity_key: row.key}, {}) YIELD node "
            "SET node += row.properties, node.ai4rag_entity_key = row.key "
            "SET node.ai4rag_kg_collections = CASE "
            "WHEN $col IN COALESCE(node.ai4rag_kg_collections, []) THEN node.ai4rag_kg_collections "
            "ELSE COALESCE(node.ai4rag_kg_collections, []) + $col END "
            "WITH node, row "
            "UNWIND row.chunk_ids AS chunk_id "
            f"MATCH (c:`{collection_name}`:{_SCHEMA.CHUNK_LABEL} "
            f"{{{_SCHEMA.COLLECTION}: $col, {_SCHEMA.ID}: chunk_id}}) "
            f"MERGE (node)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(c)",
            rows=rows,
            col=collection_name,
        ).consume()

    def _write_relationship_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self.driver.session(database=self.neo4j_database) as session:
            session.execute_write(self._write_relationship_batch_tx, rows, self._collection_name)

    @staticmethod
    def _write_relationship_batch_tx(tx: neo4j.Transaction, rows: list[dict[str, Any]], collection_name: str) -> None:
        tx.run(
            "UNWIND $rows AS row "
            "MATCH (start:__Entity__ {ai4rag_entity_key: row.start_key}) "
            "WITH start, row MATCH (end:__Entity__ {ai4rag_entity_key: row.end_key}) "
            "CALL apoc.merge.relationship(start, row.type, {}, row.properties, end, row.properties) YIELD rel "
            "SET rel.ai4rag_kg_collections = CASE "
            "WHEN $col IN COALESCE(rel.ai4rag_kg_collections, []) THEN rel.ai4rag_kg_collections "
            "ELSE COALESCE(rel.ai4rag_kg_collections, []) + $col END",
            rows=rows,
            col=collection_name,
        ).consume()


def _build_graph_retrieval_query(
    include_entity_neighbors: bool,
    entity_neighbor_limit: int,
    entity_pivot_limit: int = 3,
    entity_relationship_hops: int = 1,
    relationship_neighbor_limit: int = 5,
) -> str:
    """Build the Cypher retrieval query for :class:`VectorCypherRetriever`.

    The query receives seed ``Chunk`` nodes from vector search, ranks their
    graph neighbors by query similarity and graph support, then returns
    distinct chunks as individual results. A seed with no graph neighbors is
    returned as a fallback. Candidate chunks are scored using their own
    embeddings rather than inheriting a seed's ANN score.

    Multi-hop expansion deliberately aggregates over *distinct related
    entities* instead of Cypher ``path`` values. Binding ``path`` materialises
    every simple path between a pivot and a related entity, which is
    combinatorially larger than the entity set and exhausts the server heap
    (``TransientError.General.OutOfMemoryError`` / ``51N36``) on dense graphs.
    Both the expanded entities and the traversed relationships are also
    restricted to the current collection, so expansion never walks through
    another collection's subgraph.
    """
    # $col is passed via query_params in _search_graph to scope graph expansion
    # to one collection. The seed index is already collection-specific.
    collection_filter = f"WHERE node.{_SCHEMA.COLLECTION} = $col "

    if include_entity_neighbors and entity_neighbor_limit:
        entity_block = (
            "CALL (node) { "
            f"OPTIONAL MATCH (entity:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(node) "
            f"OPTIONAL MATCH (entity)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(ent_nb:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE ent_nb <> node AND ent_nb.{_SCHEMA.COLLECTION} = $col "
            "WITH ent_nb, count(DISTINCT entity) AS overlap "
            f"WHERE ent_nb IS NOT NULL AND ent_nb.{_SCHEMA.EMBEDDING} IS NOT NULL "
            f"WITH ent_nb, overlap, vector.similarity.cosine(ent_nb.{_SCHEMA.EMBEDDING}, $query_vector) AS similarity "
            f"ORDER BY similarity DESC, overlap DESC, ent_nb.{_SCHEMA.ID} ASC "
            f"LIMIT {entity_neighbor_limit} "
            "RETURN collect({chunk: ent_nb, strength: toFloat(overlap)}) AS ent_hits } "
        )
    else:
        entity_block = "WITH node, [] AS ent_hits "

    if entity_pivot_limit and entity_relationship_hops and relationship_neighbor_limit:
        relationship_block = (
            "CALL (node) { "
            f"MATCH (pivot:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(node) "
            f"OPTIONAL MATCH (pivot)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(pivot_chunk:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE pivot_chunk.{_SCHEMA.COLLECTION} = $col "
            "WITH node, pivot, count(DISTINCT pivot_chunk) AS degree "
            f"ORDER BY degree DESC, pivot.name ASC, pivot.{_SCHEMA.ID} ASC "
            f"LIMIT {entity_pivot_limit} "
            f"MATCH (pivot)-[rels*1..{entity_relationship_hops}]-(related:{_SCHEMA.ENTITY_LABEL}) "
            "WHERE related <> pivot "
            f"AND $col IN COALESCE(related.{_SCHEMA.KG_COLLECTIONS}, []) "
            f"AND ALL(rel IN rels WHERE type(rel) <> '{_SCHEMA.FROM_CHUNK_RELATIONSHIP}' "
            f"AND $col IN COALESCE(rel.{_SCHEMA.KG_COLLECTIONS}, [])) "
            f"MATCH (related)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(rel_nb:{_SCHEMA.CHUNK_LABEL}) "
            f"WHERE rel_nb <> node AND rel_nb.{_SCHEMA.COLLECTION} = $col "
            f"AND rel_nb.{_SCHEMA.EMBEDDING} IS NOT NULL "
            "WITH rel_nb, count(DISTINCT related) AS entity_count, min(size(rels)) AS hops "
            "WITH rel_nb, entity_count, hops, "
            f"vector.similarity.cosine(rel_nb.{_SCHEMA.EMBEDDING}, $query_vector) AS similarity "
            f"ORDER BY similarity DESC, entity_count DESC, hops ASC, rel_nb.{_SCHEMA.ID} ASC "
            f"LIMIT {relationship_neighbor_limit} "
            "RETURN collect({chunk: rel_nb, strength: toFloat(entity_count) / toFloat(hops)}) AS rel_hits } "
        )
    else:
        relationship_block = "WITH node, ent_hits, [] AS rel_hits "

    # Give independent query similarity most of the weight; bound accumulated
    # graph support so high-degree entities cannot dominate the local route.
    return (
        collection_filter
        + entity_block
        + relationship_block
        + "WITH node, ent_hits + rel_hits AS graph_hits "
        + "WITH CASE WHEN size(graph_hits) = 0 "
        + "THEN [{chunk: node, strength: 0.0}] ELSE graph_hits END AS hits "
        + "UNWIND hits AS hit "
        + "WITH hit.chunk AS candidate, sum(hit.strength) AS graph_strength "
        + "WITH candidate, graph_strength, "
        + f"coalesce(vector.similarity.cosine(candidate.{_SCHEMA.EMBEDDING}, $query_vector), 0.0) AS semantic_score "
        + "WITH candidate, graph_strength, "
        + "0.8 * semantic_score + 0.2 * graph_strength / (1.0 + graph_strength) AS score "
        + f"ORDER BY score DESC, graph_strength DESC, candidate.{_SCHEMA.ID} ASC LIMIT $top_k "
        + f"RETURN candidate.{_SCHEMA.TEXT} AS text, score, "
        + f"candidate.{_SCHEMA.DOCUMENT_ID} AS document_id, candidate.{_SCHEMA.METADATA} AS metadata"
    )


def _validate_neo4j_search_params(
    search_mode: str,
    ranker_strategy: str | None = None,
    ranker_k: int | None = None,
    ranker_alpha: float | None = None,
    store_class: type | None = None,
    **kwargs: Any,
) -> None:
    # pylint: disable=duplicate-code
    validate_search_params(
        search_mode,
        ranker_strategy,
        ranker_k,
        ranker_alpha,
        supported_modes=("graph",),
        store_class=store_class,
    )
    # pylint: enable=duplicate-code

    if search_mode == "graph":
        graph_hops = kwargs.get("graph_hops", 0)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 3)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 1)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 5)
        if not isinstance(graph_hops, int) or graph_hops != 0:
            raise ValueError(
                "graph_hops must be 0 because Neo4j graph search expands only through extracted KG entities, "
                f"got {graph_hops!r}."
            )
        if not isinstance(entity_neighbor_limit, int) or entity_neighbor_limit < 0:
            raise ValueError(f"entity_neighbor_limit must be a non-negative integer, got {entity_neighbor_limit!r}.")
        for name, value in {
            "entity_pivot_limit": entity_pivot_limit,
            "entity_relationship_hops": entity_relationship_hops,
            "relationship_neighbor_limit": relationship_neighbor_limit,
        }.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
