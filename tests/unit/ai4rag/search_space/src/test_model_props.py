# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Tests for default RAG prompt templates."""

import pytest

from ai4rag.search_space.src.model_props import (
    DOCUMENT_NUMBER_PLACEHOLDER,
    get_context_template_text,
    get_system_message_text,
    get_user_message_text,
)


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/llama-3-1-8b-instruct",
        "ibm/granite-3-8b-instruct",
        "mistralai/mistral-large",
        "openai/gpt-oss-120b",
        "unknown-model",
    ],
)
def test_user_message_includes_grounding_and_citations(model_name: str):
    user_message = get_user_message_text(model_name)
    assert "Answer ONLY" in user_message
    assert "MUST cite sources" in user_message
    assert "{reference_documents}" in user_message
    assert "{question}" in user_message


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/llama-3-1-8b-instruct",
        "ibm/granite-3-8b-instruct",
        "mistralai/mistral-large",
        "openai/gpt-oss-120b",
        "vllm-inference-gpu-llama/redhataillama-31-8b-instruct",
    ],
)
def test_system_message_includes_rag_prefix(model_name: str):
    system_message = get_system_message_text(model_name)
    assert "retrieval-augmented assistant" in system_message
    assert "ONLY the provided documents" in system_message


def test_context_template_numbers_documents():
    context_template = get_context_template_text()
    assert f"{{{DOCUMENT_NUMBER_PLACEHOLDER}}}" in context_template
    assert "{document}" in context_template


def test_language_auto_uses_autodetect_prompt():
    """Default language='auto' embeds the autodetect instruction."""
    user_message = get_user_message_text("meta-llama/llama-3-1-8b-instruct")
    assert "You MUST write your entire answer in the same language as the question" in user_message
    assert "Do NOT respond in any other language" in user_message


def test_explicit_language_embeds_instruction():
    """Passing an explicit language name produces a 'MUST respond in <lang>' instruction."""
    user_message = get_user_message_text("meta-llama/llama-3-1-8b-instruct", language="Japanese")
    assert "You MUST respond in Japanese." in user_message
    assert "same language as the question" not in user_message


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/llama-3-1-8b-instruct",
        "ibm/granite-3-8b-instruct",
        "mistralai/mistral-large",
        "openai/gpt-oss-120b",
        "unknown-model",
    ],
)
def test_user_message_includes_consistent_answer_length(model_name: str):
    user_message = get_user_message_text(model_name)
    assert "Answer (max 150 words, with citations):" in user_message
