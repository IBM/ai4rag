# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

_USER_QUERY_PLACEHOLDER = "<user_query_placeholder>"

_HPO_GROUNDING_PREFIX = "Answer ONLY using information from the documents below"
_HPO_CITATION_INSTRUCTION = (
    "You MUST cite sources using [1], [2], etc. matching the document numbers for every factual claim."
)
_RESPONSES_GROUNDING_INSTRUCTION = (
    "Answer ONLY using information from documents retrieved via file search. "
    "Do not use outside knowledge. "
    "If the retrieved documents do not contain the answer, say you do not have enough information."
)
_SYSTEM_GROUNDING_HPO = "Answer using ONLY the provided documents."
_SYSTEM_GROUNDING_RESPONSES = "Answer using ONLY information from documents retrieved via file search."

# OGX RAG distro injects chunk presentation, context framing, and <|file-id|> citation
# instructions at file_search runtime (see benchmarking/rag/config.yaml:
# file_search_params, context_prompt_params, annotation_prompt_params).
_CITATION_LINE_PREFIXES = ("You MUST cite sources",)
_HPO_CITATION_FRAGMENTS = (
    _HPO_CITATION_INSTRUCTION,
    "You MUST cite sources using [1], [2], etc.",
    "You MUST cite sources using [1], [2].",
)


def _is_citation_line(line: str) -> bool:
    """Return whether a line is an HPO or export citation instruction."""
    stripped = line.strip()
    return any(stripped.startswith(prefix) for prefix in _CITATION_LINE_PREFIXES)


def _strip_citation_instructions(text: str) -> str:
    """Remove citation instructions from export text.

    OGX ``annotation_prompt_params`` injects ``<|file-id|>`` citation guidance
    when ``file_search`` returns chunks, so exported ``responses_template`` must
    not duplicate HPO ``[1]``, ``[2]`` rules or Responses-specific citation text.
    """
    if not text.strip():
        return ""

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _is_citation_line(stripped):
            continue
        cleaned = stripped
        for fragment in _HPO_CITATION_FRAGMENTS:
            if fragment in cleaned:
                cleaned = cleaned.replace(fragment, "").strip()
                break
        if cleaned:
            lines.append(cleaned)

    result = "\n".join(lines)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def _system_has_grounding_policy(system: str) -> bool:
    """Return whether the system prompt already states a document-only grounding policy."""
    normalized = system.lower()
    return (
        "answer using only the provided documents" in normalized
        or "retrieval-augmented assistant" in normalized
    )


def _filter_static_user_for_responses(system: str, static_user: str) -> str:
    """Drop user-template lines that duplicate system policy for Responses export.

    HPO user templates combine static rules with ``{reference_documents}`` and
    ``{question}`` slots. After stripping those slots, some rule lines overlap
    with ``system_message_text`` (PR #75 ``_RAG_SYSTEM_PREFIX`` and family
    personas). Responses receives documents via ``file_search`` and the
    question via ``input[role=user]``, so only non-redundant supplements belong
    in ``input[role=system]``.
    """
    if not static_user.strip():
        return ""

    system_has_grounding = _system_has_grounding_policy(system)
    system_has_citation = "MUST cite sources" in system

    filtered_lines: list[str] = []
    for line in static_user.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if system_has_grounding and stripped.startswith("Answer ONLY using information from the documents"):
            continue
        if system_has_grounding and stripped.startswith("Do not use outside knowledge"):
            continue
        if system_has_grounding and stripped.startswith("If the documents do not contain the answer"):
            continue
        if system_has_citation and stripped.startswith("You MUST cite sources"):
            continue
        if system_has_grounding and stripped.startswith("You are a specialized Retrieval Augmented Generation"):
            continue
        if system_has_grounding and stripped.startswith("Prioritize correctness and ensure your response is grounded"):
            continue
        filtered_lines.append(stripped)

    if not filtered_lines:
        return ""

    blocks: list[str] = []
    current_block: list[str] = []
    for line in filtered_lines:
        if line.startswith("Answer (") and current_block:
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))
    return "\n\n".join(blocks)


def _adapt_system_for_responses_export(system: str) -> str:
    """Rephrase HPO system text for tool-based retrieval in Responses API."""
    adapted = system.replace(_SYSTEM_GROUNDING_HPO, _SYSTEM_GROUNDING_RESPONSES)
    adapted = adapted.replace(
        "If the question is unanswerable from the documents,",
        "If the question is unanswerable from the retrieved documents,",
    )
    return _strip_citation_instructions(adapted)


def _adapt_static_user_for_responses_export(static_user: str) -> str:
    """Rephrase merged HPO user supplements for file_search-based Responses export."""
    if not static_user.strip():
        return ""

    adapted_lines: list[str] = []
    for line in static_user.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(_HPO_GROUNDING_PREFIX):
            adapted_lines.append(_RESPONSES_GROUNDING_INSTRUCTION)
            continue
        if stripped.startswith("Do not use outside knowledge"):
            continue
        if stripped.startswith("If the documents do not contain the answer"):
            continue
        if _is_citation_line(stripped):
            continue
        adapted_lines.append(_strip_citation_instructions(stripped))

    if not adapted_lines:
        return ""

    blocks: list[str] = []
    current_block: list[str] = []
    for line in adapted_lines:
        if line.startswith("Answer (") and current_block:
            blocks.append("\n".join(current_block))
            current_block = [line]
        else:
            current_block.append(line)
    if current_block:
        blocks.append("\n".join(current_block))
    return "\n\n".join(blocks)


def _extract_static_user_instructions(user_message_text: str) -> str:
    """Return static instruction text from a HPO user template.

    Strips runtime slots (retrieved documents, question) that Responses API
    supplies via ``file_search`` and the user ``input`` message respectively.
    """
    if not user_message_text:
        return ""

    text = str(user_message_text).strip()
    parts: list[str] = []

    if "{reference_documents}" in text:
        before, after = text.split("{reference_documents}", 1)
        prefix = before.strip()
        for label in ("Documents:", "Context:", "[Document]"):
            if prefix == label:
                prefix = ""
            elif prefix.endswith(label):
                prefix = prefix[: -len(label)].strip()
        if prefix:
            parts.append(prefix)

        suffix_lines: list[str] = []
        for line in after.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if "{question}" in stripped:
                continue
            if stripped.startswith("Question:") or stripped.startswith("Q:"):
                continue
            if "{multilingual_support}" in stripped:
                continue
            suffix_lines.append(stripped)
        if suffix_lines:
            parts.append("\n".join(suffix_lines))
    else:
        doc_idx = len(text)
        for marker in ("Documents:\n", "Context:\n", "[Document]\n"):
            idx = text.find(marker)
            if idx != -1:
                doc_idx = min(doc_idx, idx)
        prefix = text[:doc_idx].strip()
        if prefix:
            parts.append(prefix)

    return "\n\n".join(parts).strip()


def build_responses_system_input(generation: dict) -> str:
    """Build Responses API system input aligned with HPO chat/completion prompts.

    HPO sends ``system_message_text`` plus a formatted ``user_message_text``
    (rules, documents, question). Responses uses ``file_search`` for documents
    and a separate user message for the question. Non-redundant supplements
    from the user template are merged and rephrased for tool-based retrieval
    (no ``documents below`` wording, HPO ``[1]``, ``[2]`` citations, or other
    file_search runtime text that OGX injects via ``annotation_prompt_params``).
    """
    system = _adapt_system_for_responses_export((generation.get("system_message_text") or "").strip())
    user_template = generation.get("user_message_text") or ""
    raw_system = (generation.get("system_message_text") or "").strip()

    static_user = _adapt_static_user_for_responses_export(
        _filter_static_user_for_responses(
            raw_system,
            _extract_static_user_instructions(user_template),
        ),
    )
    if not static_user:
        return system
    if not system:
        return static_user
    return f"{system}\n\n{static_user}"


def build_pattern_json(
    pattern: dict,
    detected_language: dict | None = None,
) -> dict:
    """Update pattern information with responses template.

    Parameters
    ----------
    pattern : dict
        A single evaluation result object carrying ``indexing_params``,
        ``rag_params``, ``pattern_name``, ``collection``, etc.
    detected_language : dict | None, default=None
        Language detection result (``{"code": "...", "name": "..."}``).

    Returns
    -------
    dict
        Pattern definition suitable for JSON serialisation.
    """
    generation = pattern["settings"]["generation"]
    system_input = build_responses_system_input(generation)
    if detected_language:
        pattern["settings"]["generation"]["detected_language"] = detected_language

    pattern["settings"]["responses_template"] = {
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
        "max_output_tokens": generation["max_completion_tokens"],
        "temperature": generation["temperature"],
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
