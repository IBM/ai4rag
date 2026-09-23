# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from os import getenv

from openai import OpenAI

from ai4rag.rag.foundation_models.base_model import Language
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.template.agentic_rag_template import AgenticRAG

from .config import AgentConfig, get_chat_base_url
from .tools import _initialize_retriever


def create_rag(
    model_id: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> AgenticRAG:
    """Create the configured RAG template from environment settings."""
    config = AgentConfig.from_env()
    model_id = model_id or config.model_id or getenv("MODEL_ID")
    base_url = (base_url or get_chat_base_url() or "").rstrip("/")
    api_key = api_key or getenv("MAAS_API_KEY")

    if not model_id:
        raise ValueError("MODEL_ID is required for the chat model.")
    if not base_url:
        raise ValueError("CHAT_BASE_URL or BASE_URL is required for the chat model.")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    is_local = any(host in base_url for host in ["localhost", "127.0.0.1"])
    if not is_local and not api_key:
        raise ValueError("MAAS_API_KEY is required for non-local environments.")

    client = OpenAI(api_key=api_key or "not-needed-for-local-development", base_url=base_url)
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
        retriever=_initialize_retriever(
            maas_api_key=api_key,
            maas_base_url=base_url,
            embedding_model_id=config.embedding_model_id,
            embedding_dimension=config.embedding_dimension,
            collection_name=config.collection_name,
            provider_type=config.provider_type,
            retrieval_method=config.retrieval_method,
            number_of_chunks=config.number_of_chunks,
            search_mode=config.search_mode,
            ranker_strategy=config.ranker_strategy,
            ranker_alpha=config.ranker_alpha,
        ),
    )
