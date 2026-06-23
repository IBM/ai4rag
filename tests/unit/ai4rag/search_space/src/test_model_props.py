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
def test_context_template_numbers_documents(model_name: str):
    context_template = get_context_template_text(model_name)
    assert f"{{{DOCUMENT_NUMBER_PLACEHOLDER}}}" in context_template
    assert "{document}" in context_template


def test_language_autodetect_defaults_to_english_only():
    user_message = get_user_message_text("meta-llama/llama-3-1-8b-instruct")
    assert "English only" in user_message


def test_language_autodetect_enabled_uses_question_language():
    user_message = get_user_message_text("meta-llama/llama-3-1-8b-instruct", language_autodetect=True)
    assert "language of the question" in user_message
