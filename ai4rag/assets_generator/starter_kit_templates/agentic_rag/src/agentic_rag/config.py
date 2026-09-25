# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import base64
import json
from dataclasses import dataclass
from os import getenv
from pathlib import Path


def _decode_template(name: str) -> str:
    """Decode a legacy base64-encoded template for older starter-kits."""
    encoded = getenv(name, "")
    if not encoded:
        return ""
    try:
        return base64.b64decode(encoded).decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{name} must contain valid base64-encoded UTF-8 text") from exc


def _load_agent_config() -> dict:
    """Load generated settings from deployment environment or a local file."""
    encoded_config = getenv("AGENT_CONFIG_B64", "").strip()
    if encoded_config:
        try:
            data = json.loads(base64.b64decode(encoded_config).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("AGENT_CONFIG_B64 must contain valid base64-encoded JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("AGENT_CONFIG_B64 must decode to a JSON object")
        return data

    # Local fallback keeps ``make run-app`` convenient. OpenShell deployments
    # inject the configuration through AGENT_CONFIG_B64 instead of copying the
    # JSON file into the sandbox.
    candidates = (
        Path.cwd() / "agent_config.json",
        Path(__file__).resolve().parents[2] / "agent_config.json",
    )
    for path in candidates:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Could not read agent configuration from {path}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Agent configuration in {path} must be a JSON object")
            return data
    return {}


def get_chat_base_url() -> str | None:
    """Return the shared OpenAI-compatible MaaS endpoint.

    MaaS serves foundation and embedding models through the same ``/v1``
    endpoint. The model identifier belongs in the request body, so it must not
    be rewritten into a model-specific URL.
    """
    explicit_url = getenv("CHAT_BASE_URL", "").strip()
    if explicit_url:
        return explicit_url

    maas_url = getenv("MAAS_BASE_URL", "").strip().rstrip("/")
    if not maas_url:
        return getenv("BASE_URL") or None
    return maas_url


@dataclass(frozen=True)
class AgentConfig:
    """Runtime configuration generated from an optimized RAG pattern."""

    model_id: str
    temperature: float
    max_completion_tokens: int
    system_message: str
    user_message_template: str
    context_template: str
    language_code: str
    language_name: str
    embedding_model_id: str
    embedding_dimension: int
    provider_type: str
    collection_name: str
    retrieval_method: str
    number_of_chunks: int
    search_mode: str
    ranker_strategy: str
    ranker_alpha: float | None
    port: int

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load and validate agent configuration from environment variables."""
        file_config = _load_agent_config()
        generation = file_config.get("generation", {})
        prompts = file_config.get("prompts", {})
        embedding = file_config.get("embedding", {})
        retrieval = file_config.get("retrieval", {})
        vector_store = file_config.get("vector_store", {})
        runtime = file_config.get("runtime", {})
        try:
            temperature = float(getenv("TEMPERATURE", generation.get("temperature", 0.0)))
            max_completion_tokens = int(getenv("MAX_COMPLETION_TOKENS", generation.get("max_completion_tokens", 1024)))
        except ValueError as exc:
            raise ValueError("TEMPERATURE and MAX_COMPLETION_TOKENS must be numeric") from exc

        return cls(
            model_id=getenv("MODEL_ID", generation.get("model_id", "")),
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            system_message=prompts.get("system_message")
            or _decode_template("SYSTEM_MESSAGE_B64")
            or getenv("SYSTEM_MESSAGE", ""),
            user_message_template=prompts.get("user_message_template")
            or _decode_template("USER_MESSAGE_B64")
            or "Context:\n{reference_documents}\n\nQuestion: {question}",
            context_template=prompts.get("context_template")
            or _decode_template("CONTEXT_TEMPLATE_B64")
            or "{document}",
            language_code=getenv("LANGUAGE_CODE", generation.get("language_code", "")),
            language_name=getenv("LANGUAGE_NAME", generation.get("language_name", "auto")),
            embedding_model_id=getenv("EMBEDDING_MODEL_ID", embedding.get("model_id", "")),
            embedding_dimension=int(getenv("EMBEDDING_DIMENSION", embedding.get("dimension", 768))),
            provider_type=getenv("PROVIDER_TYPE", vector_store.get("provider_type", "milvus")),
            collection_name=(
                getenv("MILVUS_COLLECTION_NAME")
                or getenv("PGVECTOR_COLLECTION_NAME")
                or vector_store.get("collection_name", "")
            ),
            retrieval_method=getenv("RETRIEVAL_METHOD", retrieval.get("method", "simple")),
            number_of_chunks=int(getenv("NUMBER_OF_CHUNKS", retrieval.get("number_of_chunks", 5))),
            search_mode=getenv("SEARCH_MODE", retrieval.get("search_mode", "vector")),
            ranker_strategy=getenv("RANKER_STRATEGY", retrieval.get("ranker_strategy", "")),
            ranker_alpha=(
                float(getenv("RANKER_ALPHA", retrieval.get("ranker_alpha")))
                if getenv("RANKER_ALPHA", retrieval.get("ranker_alpha")) not in (None, "")
                else None
            ),
            port=int(getenv("PORT", runtime.get("port", 8000))),
        )
