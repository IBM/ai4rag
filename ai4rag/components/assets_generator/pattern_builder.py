# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Build Responses API pattern definitions from HPO experiment results."""

import re

from ai4rag.components.assets_generator.prompt_filters import (
    GROUNDING_PREFIXES,
    USER_GROUNDING_SKIP_PREFIXES,
    USER_RAG_GROUNDING_PREFIXES,
    is_citation_related_line,
    strip_ogx_runtime_instructions,
)

_USER_QUERY_PLACEHOLDER = "<user_query_placeholder>"
_EMPTY_SYSTEM_FALLBACK = "You are a helpful assistant."
_EXPORT_SLOT_MARKERS = ("{reference_documents}", "{question}", "{multilingual_support}")

# Suffix lines after ``{reference_documents}``: drop structural wrappers (e.g. ``[End]``).
_DOCUMENT_SLOT_MARKERS = frozenset({"[Document]", "[End]", "Documents:", "Context:"})

# Document and question slot markers
_DOCUMENT_LABELS = ("Documents:", "Context:", "[Document]")
_QUESTION_PREFIXES = ("Question:", "Q:", "[conversation]:")


def _join_answer_scaffold_blocks(lines: list[str]) -> str:
    """Group lines into paragraph blocks, starting a new block when an answer-scaffold line appears.

    Scaffold lines are specifically in the form ``Answer (...)`` — e.g.
    ``"Answer (max 150 words):"`` — as produced by HPO prompt templates.
    Other leading text such as ``"Answer:"`` or ``"Response:"`` does NOT
    trigger a new block.
    """
    if not lines:
        return ""

    blocks: list[str] = []
    current_block: list[str] = []
    for line in lines:
        if line.startswith("Answer (") and current_block:
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))
    return "\n\n".join(blocks)


def _should_skip_redundant_user_line(stripped: str, system_has_grounding: bool) -> bool:
    """Return whether a user-template line duplicates system policy for export."""
    if is_citation_related_line(stripped):
        return True
    return system_has_grounding and any(
        stripped.startswith(prefix) for prefix in GROUNDING_PREFIXES + USER_RAG_GROUNDING_PREFIXES
    )


def _should_skip_user_export_line(stripped: str) -> bool:
    """Return whether a merged user line is OGX-owned and must not be exported."""
    if any(stripped.startswith(prefix) for prefix in USER_GROUNDING_SKIP_PREFIXES):
        return True
    return is_citation_related_line(stripped)


def _strip_document_slot_prefix(prefix: str) -> str:
    """Remove structural labels that wrap the reference-documents slot."""
    for label in _DOCUMENT_LABELS:
        if prefix == label:
            return ""
        if prefix.endswith(label):
            return prefix[: -len(label)].strip()
    return prefix


def _extract_static_suffix_line(stripped: str) -> str | None:
    """Return static instruction text from one post-documents template line."""
    if not stripped or stripped == ":" or stripped in _DOCUMENT_SLOT_MARKERS:
        return None
    if "{question}" in stripped:
        without_question = stripped.replace("{question}", "").strip()
        for question_prefix in _QUESTION_PREFIXES:
            if without_question.startswith(question_prefix):
                without_question = without_question[len(question_prefix) :].strip()
        without_question = without_question.lstrip(":.").strip()
        return without_question or None
    if stripped.startswith(_QUESTION_PREFIXES):
        return None
    if "{multilingual_support}" in stripped:
        return None
    return stripped


def _extract_static_user_from_reference_slot(text: str) -> str:
    """Extract static instructions from a template that contains ``{reference_documents}``."""
    before, after = text.split("{reference_documents}", 1)
    parts: list[str] = []
    prefix = _strip_document_slot_prefix(before.strip())
    if prefix:
        parts.append(prefix)

    suffix_lines = [
        line_text
        for line_text in (_extract_static_suffix_line(line.strip()) for line in after.splitlines())
        if line_text
    ]
    if suffix_lines:
        parts.append("\n".join(suffix_lines))
    return "\n\n".join(parts).strip()


def _system_has_grounding_policy(system: str) -> bool:
    """Return whether the system prompt already states an explicit document-only grounding rule.

    Uses the same prefix list as sentence-level filtering so that adding a new
    OGX phrase to ``GROUNDING_PREFIXES`` automatically covers system detection too.
    Does NOT match descriptive personas like "retrieval-augmented assistant" without
    an explicit grounding constraint.

    Checks at sentence granularity to avoid false positives from embedded substrings.
    """
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", system)]
    return any(any(sent.lower().startswith(p.lower()) for p in GROUNDING_PREFIXES) for sent in sentences)


def _filter_static_user_for_responses(system: str, static_user: str) -> str:
    """Drop user-template lines that duplicate system policy for Responses export.

    Pass 1 of 2: compare against ``original_system`` (author intent before OGX
    stripping). Removes user lines that repeat grounding or citation policy already
    present in the HPO system prompt.
    """
    if not static_user.strip():
        return ""

    system_has_grounding = _system_has_grounding_policy(system)

    filtered_lines: list[str] = []
    for line in static_user.splitlines():
        stripped = line.strip()
        if not stripped or _should_skip_redundant_user_line(stripped, system_has_grounding):
            continue
        filtered_lines.append(stripped)

    return _join_answer_scaffold_blocks(filtered_lines)


def _adapt_system_for_responses_export(system: str) -> str:
    """Drop OGX-runtime retrieval/citation text from the HPO system prompt."""
    return strip_ogx_runtime_instructions(system)


def _adapt_static_user_for_responses_export(static_user: str) -> str:
    """Drop merged user supplements that OGX injects at file_search runtime.

    Pass 2 of 2: strip OGX-runtime phrases from user lines that survived pass 1.
    """
    if not static_user.strip():
        return ""

    adapted_lines: list[str] = []
    for line in static_user.splitlines():
        stripped = line.strip()
        if not stripped or _should_skip_user_export_line(stripped):
            continue
        cleaned = strip_ogx_runtime_instructions(stripped)
        if cleaned:
            adapted_lines.append(cleaned)

    return _join_answer_scaffold_blocks(adapted_lines)


def _extract_static_user_instructions(user_message_text: str) -> str:
    """Return static instruction text from a HPO user template.

    Strips runtime slots (retrieved documents, question) that Responses API
    supplies via ``file_search`` and the user ``input`` message respectively.

    All current templates use {reference_documents} placeholder format.
    """
    if not user_message_text:
        return ""

    text = str(user_message_text).strip()
    if "{reference_documents}" not in text:
        return ""

    return _extract_static_user_from_reference_slot(text)


def _is_placeholder_only_export(text: str) -> bool:
    """Return whether export text contains only unresolved HPO template slots."""
    cleaned = text.strip()
    if not cleaned:
        return True
    for marker in _EXPORT_SLOT_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return not cleaned.strip()


def build_responses_system_input(generation: dict) -> str:
    """Build Responses API system input aligned with HPO chat/completion prompts.

    HPO sends ``system_message_text`` plus a formatted ``user_message_text``
    (rules, documents, question). Responses uses ``file_search`` for documents
    and a separate user message for the question. Non-redundant supplements
    from the user template are merged into export; retrieval framing, chunk
    presentation, and citation instructions owned by OGX ``config.yaml`` are
    stripped rather than rephrased into the exported system input.
    """
    original_system = (generation.get("system_message_text") or "").strip()
    exported_system = _adapt_system_for_responses_export(original_system)
    user_template = generation.get("user_message_text") or ""

    # Pass 1: dedupe vs original_system; pass 2: strip OGX-owned user supplements.
    static_user = _adapt_static_user_for_responses_export(
        _filter_static_user_for_responses(
            original_system,
            _extract_static_user_instructions(user_template),
        ),
    )

    if exported_system and static_user:
        result = f"{exported_system}\n\n{static_user}"
    else:
        result = exported_system or static_user

    # Fallback for completely empty patterns (rare edge case)
    if not result or not result.strip() or _is_placeholder_only_export(result):
        return _EMPTY_SYSTEM_FALLBACK

    return result


def build_pattern_json(
    pattern: dict,
) -> dict:
    """Update pattern information with responses template.

    Parameters
    ----------
    pattern : dict
        A single evaluation result object carrying ``indexing_params``,
        ``rag_params``, ``pattern_name``, ``collection``, etc.

    Notes
    -----
    ``pattern["settings"]["generation"]`` must include ``model_id``,
    ``temperature``, ``max_completion_tokens``, ``system_message_text``, and
    ``user_message_text`` (as produced by the experiment payload).

    Returns
    -------
    dict
        Pattern definition suitable for JSON serialisation.
    """
    generation = pattern["settings"]["generation"]
    system_input = build_responses_system_input(generation)

    responses_template = {
        "model": generation["model_id"],
        "stream": False,
        "store": False,
        "input": [
            {
                "content": [{"text": system_input, "type": "input_text"}],
                "role": "system",
            },
            {"content": [{"text": _USER_QUERY_PLACEHOLDER, "type": "input_text"}], "role": "user"},
        ],
        "tool_choice": {"mode": "required", "tools": [{}], "type": "file_search"},
        "tools": [
            {
                "type": "file_search",
                "vector_store_ids": [pattern["settings"]["vector_store_binding"]["vector_store_id"]],
                "max_num_results": pattern["settings"]["retrieval"]["number_of_chunks"],
            },
        ],
        "include": ["file_search_call.results"],
    }

    # Only include temperature and max_output_tokens if they are not None
    if generation.get("temperature") is not None:
        responses_template["temperature"] = generation["temperature"]
    if generation.get("max_completion_tokens") is not None:
        responses_template["max_output_tokens"] = generation["max_completion_tokens"]

    pattern["settings"]["responses_template"] = responses_template

    retrieval_settings = pattern["settings"]["retrieval"]
    search_mode = retrieval_settings.get("search_mode")
    ranker_strategy = retrieval_settings.get("ranker_strategy")
    ranker_k = retrieval_settings.get("ranker_k")
    ranker_alpha = retrieval_settings.get("ranker_alpha")

    if search_mode == "hybrid" and ranker_strategy == "rrf" and ranker_k is not None and ranker_k > 0:
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"] = {
            "ranker": "rrf",
            "impact_factor": ranker_k,
        }
    elif search_mode == "hybrid" and ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
        # ``ranker_alpha == 1.0`` intentionally falls through to ``else`` (semantic-only default).
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"] = {
            "ranker": "weighted",
            "alpha": ranker_alpha,
        }
    else:
        pattern["settings"]["responses_template"]["tools"][0]["ranking_options"] = {
            "ranker": "weighted",
            "alpha": 1.0,
        }

    return pattern
