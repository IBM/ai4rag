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
def test_user_message_contains_required_placeholders(model_name: str):
    user_message = get_user_message_text(model_name)
    assert "{reference_documents}" in user_message
    assert "{question}" in user_message


@pytest.mark.parametrize(
    "model_name, expected_fragment",
    [
        ("meta-llama/llama-3-1-8b-instruct", "helpful, respectful and honest assistant"),
        ("ibm/granite-3-8b-instruct", "Granite Chat"),
        ("mistralai/mistral-large", "helpful, respectful and honest assistant"),
        ("openai/gpt-oss-120b", "Retrieval Augmented Generation (RAG) assistant"),
        ("vllm-inference-gpu-llama/redhataillama-31-8b-instruct", "helpful, respectful and honest assistant"),
    ],
)
def test_system_message_contains_family_marker(model_name: str, expected_fragment: str):
    assert expected_fragment in get_system_message_text(model_name)


@pytest.mark.parametrize(
    "model_name, expected_fragment",
    [
        ("meta-llama/llama-3-1-8b-instruct", "[conversation]:"),
        ("ibm/granite-3-8b-instruct", "Answer Length: concise"),
        ("mistralai/mistral-large", "Generate the next agent response"),
        ("openai/gpt-oss-120b", "[Document]"),
        ("unknown-model", "Context:"),
    ],
)
def test_user_message_contains_family_marker(model_name: str, expected_fragment: str):
    assert expected_fragment in get_user_message_text(model_name)


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
def test_user_message_has_no_word_count_limit(model_name: str):
    """No prompt should impose a hard word-count limit; conciseness is expressed differently."""
    user_message = get_user_message_text(model_name)
    assert "150 words" not in user_message
    assert "max 150 words" not in user_message


def test_llama_user_message_instructs_to_be_concise():
    """Llama user message instructs conciseness in place of the removed word-count limit."""
    user_message = get_user_message_text("meta-llama/llama-3-1-8b-instruct")
    assert "Your answer should be concise" in user_message
