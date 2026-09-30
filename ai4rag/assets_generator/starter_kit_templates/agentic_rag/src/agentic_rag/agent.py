# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from os import getenv
from urllib.parse import urlsplit

from openai import OpenAI

from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.template.agentic_rag_template import AgenticRAG

from .config import AgentConfig
from .tools import initialize_retriever


def create_rag(
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> AgenticRAG:
    """Create the configured RAG template from environment settings."""
    config = AgentConfig.from_env()
    model_id = model_id or config.model_id
    maas_base_url = (base_url or getenv("MAAS_BASE_URL", "")).strip().rstrip("/")
    api_key = api_key or getenv("MAAS_API_KEY")

    if not model_id:
        raise ValueError("MODEL_ID is required for the chat model.")
    if not maas_base_url:
        raise ValueError("MAAS_BASE_URL is required for the MaaS models.")
    parsed_url = urlsplit(maas_base_url)
    if not parsed_url.hostname:
        raise ValueError("MaaS base URL must include a host.")
    is_local = parsed_url.hostname in {"localhost", "127.0.0.1", "::1"}
    if parsed_url.scheme != "https" and not (parsed_url.scheme == "http" and is_local):
        raise ValueError("MaaS base URL must use HTTPS unless it points to localhost.")
    if not maas_base_url.endswith("/v1"):
        maas_base_url += "/v1"

    if not is_local and not api_key:
        raise ValueError("MAAS_API_KEY is required for non-local environments.")

    client = OpenAI(api_key=api_key or "not-needed-for-local-development", base_url=maas_base_url)
    foundation_model = OpenAIFoundationModel(
        client=client,
        model_id=model_id,
        params={
            "temperature": config.temperature,
            "max_completion_tokens": config.max_completion_tokens,
        },
        system_message_text=config.system_message,
        user_message_text=config.user_message_template,
        context_template_text=config.context_template,
        language=Language(code=config.language_code, name=config.language_name),
    )

    return AgenticRAG(
        foundation_model=foundation_model,
        retriever=initialize_retriever(
            client=client,
            config=config,
        ),
    )
