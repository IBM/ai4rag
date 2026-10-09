# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import base64
import binascii
import json
from dataclasses import dataclass
from os import getenv
from pathlib import Path
from typing import TypedDict, cast


class GenerationConfig(TypedDict):
    """Generated foundation-model settings."""

    model_id: str
    temperature: float
    max_completion_tokens: int
    language_code: str
    language_name: str


class PromptConfig(TypedDict):
    """Generated prompt templates."""

    system_message: str
    user_message_template: str
    context_template: str


class EmbeddingConfig(TypedDict):
    """Generated embedding settings."""

    model_id: str
    dimension: int


class RetrievalConfig(TypedDict):
    """Generated retriever settings."""

    method: str
    number_of_chunks: int
    search_mode: str
    ranker_strategy: str
    ranker_k: int | None
    ranker_alpha: float | None


class VectorStoreConfig(TypedDict):
    """Generated vector-store settings."""

    provider_type: str
    collection_name: str


class AgentConfigData(TypedDict):
    """Expected structure of the generated agent_config.json."""

    generation: GenerationConfig
    prompts: PromptConfig
    embedding: EmbeddingConfig
    retrieval: RetrievalConfig
    vector_store: VectorStoreConfig


def _load_agent_config() -> AgentConfigData:
    """Load generated settings from deployment environment or a local file."""
    encoded_config = getenv("AGENT_CONFIG_B64", "").strip()
    if encoded_config:
        try:
            data = json.loads(base64.b64decode(encoded_config, validate=True).decode("utf-8"))
        except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("AGENT_CONFIG_B64 must contain valid base64-encoded JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("AGENT_CONFIG_B64 must decode to a JSON object")
        return cast(AgentConfigData, data)

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
            return cast(AgentConfigData, data)
    raise ValueError("AGENT_CONFIG_B64 or agent_config.json is required")


# pylint: disable=too-many-instance-attributes
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
    ranker_k: int | None
    ranker_alpha: float | None

    @classmethod
    def from_env(cls) -> "AgentConfig":
        """Load generated RAG settings."""
        file_config = _load_agent_config()
        try:
            generation = file_config["generation"]
            prompts = file_config["prompts"]
            embedding = file_config["embedding"]
            retrieval = file_config["retrieval"]
            vector_store = file_config["vector_store"]
            config = cls(
                model_id=generation["model_id"],
                temperature=float(generation["temperature"]),
                max_completion_tokens=int(generation["max_completion_tokens"]),
                system_message=prompts["system_message"],
                user_message_template=prompts["user_message_template"],
                context_template=prompts["context_template"],
                language_code=generation["language_code"],
                language_name=generation["language_name"],
                embedding_model_id=embedding["model_id"],
                embedding_dimension=int(embedding["dimension"]),
                provider_type=vector_store["provider_type"],
                collection_name=vector_store["collection_name"],
                retrieval_method=retrieval["method"],
                number_of_chunks=int(retrieval["number_of_chunks"]),
                search_mode=retrieval["search_mode"],
                ranker_strategy=retrieval["ranker_strategy"],
                ranker_k=int(retrieval["ranker_k"]) if retrieval["ranker_k"] is not None else None,
                ranker_alpha=float(retrieval["ranker_alpha"]) if retrieval["ranker_alpha"] is not None else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid agent configuration: {exc}") from exc
        for name in (
            "model_id",
            "user_message_template",
            "context_template",
            "embedding_model_id",
            "collection_name",
            "provider_type",
            "retrieval_method",
            "search_mode",
        ):
            if not isinstance(getattr(config, name), str) or not getattr(config, name).strip():
                raise ValueError(f"Agent configuration requires a non-empty {name}")
        if min(config.max_completion_tokens, config.embedding_dimension, config.number_of_chunks) < 1:
            raise ValueError("Token limit, embedding dimension, and chunk count must be positive")
        return config
