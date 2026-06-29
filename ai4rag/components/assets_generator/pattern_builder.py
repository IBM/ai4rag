# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

_USER_QUERY_PLACEHOLDER = "<user_query_placeholder>"
_EMPTY_SYSTEM_FALLBACK = "You are a helpful assistant."
_EXPORT_SLOT_MARKERS = ("{reference_documents}", "{question}", "{multilingual_support}")

# Structural markers wrapping the document slot — not instruction text for export.
_DOCUMENT_SLOT_MARKERS = frozenset({"[Document]", "[End]", "Documents:", "Context:"})

_HPO_GROUNDING_PREFIX = "Answer ONLY using information from the documents below"
_HPO_CITATION_INSTRUCTION = (
    "You MUST cite sources using [1], [2], etc. matching the document numbers for every factual claim."
)

# Phrases injected by OGX at file_search runtime (benchmarking/rag/config.yaml:
# file_search_params, context_prompt_params, annotation_prompt_params). Export must
# not repeat or substitute equivalent wording in responses_template.input[system].
_OGX_DUPLICATIVE_LINE_PREFIXES = (
    "You MUST cite sources",
    "Cite sources immediately",
    "Answer ONLY using information from the documents",
    "Answer ONLY using information from documents retrieved",
    "Answer using ONLY the provided documents",
    "Answer using ONLY information from documents",
    "Do not use outside knowledge",
    "If the retrieved documents do not contain",
    "If the documents do not contain",
    "file_search tool found",
    "BEGIN of file_search tool results",
    "END of file_search tool results",
    "The above results were retrieved to help answer",
    "Use them as supporting information only",
    "Do not add extra punctuation. Use only the file IDs",
)
_OGX_DUPLICATIVE_SUBSTRINGS = (
    "[1], [2]",
    "<|file-id|>",
    "cite as <|",
    "documents below",
    "retrieved via file search",
    "file citations",
    "retrieved to help answer the user",
    "supporting information only in answering",
)
_HPO_CITATION_FRAGMENTS = (
    _HPO_CITATION_INSTRUCTION,
    "You MUST cite sources using [1], [2], etc.",
    "You MUST cite sources using [1], [2].",
)
_SYSTEM_GROUNDING_PHRASES = (
    "Answer using ONLY the provided documents.",
    "Answer using ONLY information from documents retrieved via file search.",
)
_GROUNDING_DUPLICATE_PREFIXES = (
    "Answer ONLY using information from the documents",
    "Do not use outside knowledge",
    "If the documents do not contain the answer",
    "You are a specialized Retrieval Augmented Generation",
    "Prioritize correctness and ensure your response is grounded",
)
_USER_GROUNDING_SKIP_PREFIXES = (
    _HPO_GROUNDING_PREFIX,
    "Do not use outside knowledge",
    "If the documents do not contain the answer",
)
_DOCUMENT_LABELS = ("Documents:", "Context:", "[Document]")
_QUESTION_PREFIXES = ("Question:", "Q:", "[conversation]:")
_LEGACY_DOCUMENT_MARKERS = ("Documents:\n", "Context:\n", "[Document]\n")


def _sentence_is_ogx_duplicative(sentence: str) -> bool:
    """Return whether a sentence duplicates OGX file_search runtime injection."""
    stripped = sentence.strip().rstrip(".")
    if not stripped:
        return True
    if any(stripped.startswith(prefix.rstrip(".")) for prefix in _OGX_DUPLICATIVE_LINE_PREFIXES):
        return True
    normalized = stripped.lower()
    return any(fragment.lower() in normalized for fragment in _OGX_DUPLICATIVE_SUBSTRINGS)


def _is_citation_related_line(line: str) -> bool:
    """Return whether an entire line should be dropped as citation guidance."""
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if any(stripped.startswith(prefix) for prefix in _OGX_DUPLICATIVE_LINE_PREFIXES if "cite" in prefix.lower()):
        return True
    if any(fragment.lower() in lower for fragment in _HPO_CITATION_FRAGMENTS):
        return True
    if "[1], [2]" in stripped:
        return True
    if "document numbers for every factual claim" in lower:
        return True
    return False


def _filter_ogx_duplicative_sentences(line: str) -> str:
    """Remove OGX-duplicative sentences while keeping persona or policy sentences."""
    stripped = line.strip()
    if not stripped or _is_citation_related_line(stripped):
        return ""

    # Split on ". " only — avoids breaking abbreviations such as "i.e.,"
    parts = [part.strip() for part in stripped.split(". ") if part.strip()]
    if len(parts) <= 1:
        if _sentence_is_ogx_duplicative(stripped.rstrip(".")):
            return ""
        return stripped

    kept = [part.rstrip(".") for part in parts if not _sentence_is_ogx_duplicative(part.rstrip("."))]
    if not kept:
        return ""

    result = ". ".join(kept)
    if stripped.endswith("."):
        result += "."
    return result


def _line_is_ogx_duplicative(line: str) -> bool:
    """Return whether a line duplicates OGX file_search runtime prompt injection."""
    stripped = line.strip()
    if not stripped:
        return False
    filtered = _filter_ogx_duplicative_sentences(stripped)
    return not filtered


def _normalize_answer_scaffold(line: str) -> str:
    """Drop citation hints from answer scaffolds; OGX owns citation via annotations."""
    return line.replace(", with citations", "").replace("with citations", "").replace("  ", " ").strip()


def _strip_ogx_runtime_instructions(text: str) -> str:
    """Remove text that OGX injects via file_search config at inference time."""
    if not text.strip():
        return ""

    for phrase in _SYSTEM_GROUNDING_PHRASES:
        text = text.replace(phrase, "").replace(phrase.rstrip("."), "")

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if _line_is_ogx_duplicative(stripped) or _is_citation_related_line(stripped):
            continue

        cleaned = _filter_ogx_duplicative_sentences(stripped)
        for fragment in _HPO_CITATION_FRAGMENTS:
            if fragment in cleaned:
                cleaned = cleaned.replace(fragment, "").strip()
                break
        cleaned = _normalize_answer_scaffold(cleaned)
        if cleaned and not _line_is_ogx_duplicative(cleaned):
            lines.append(cleaned)

    result = "\n".join(lines)
    while "\n\n\n" in result:
        result = result.replace("\n\n\n", "\n\n")
    return result.strip()


def _is_citation_line(line: str) -> bool:
    """Return whether a line is an HPO citation instruction."""
    return _line_is_ogx_duplicative(line) and "cite" in line.lower()


def _strip_citation_instructions(text: str) -> str:
    """Remove citation instructions from export text (OGX annotation_prompt_params)."""
    return _strip_ogx_runtime_instructions(text)


def _join_answer_scaffold_blocks(lines: list[str]) -> str:
    """Group lines into blocks separated when an answer-scaffold line starts."""
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


def _should_skip_redundant_user_line(stripped: str, system_has_grounding: bool, system_has_citation: bool) -> bool:
    """Return whether a user-template line duplicates system policy for export."""
    if _is_citation_related_line(stripped):
        return True
    if system_has_citation and stripped.startswith("You MUST cite sources"):
        return True
    return system_has_grounding and any(stripped.startswith(prefix) for prefix in _GROUNDING_DUPLICATE_PREFIXES)


def _should_skip_user_export_line(stripped: str) -> bool:
    """Return whether a merged user line is OGX-owned and must not be exported."""
    if any(stripped.startswith(prefix) for prefix in _USER_GROUNDING_SKIP_PREFIXES):
        return True
    return _is_citation_related_line(stripped)


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


def _extract_static_user_without_reference_slot(text: str) -> str:
    """Extract static instructions from legacy templates without an explicit doc slot."""
    doc_idx = len(text)
    for marker in _LEGACY_DOCUMENT_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            doc_idx = min(doc_idx, idx)
    return text[:doc_idx].strip()


def _system_has_grounding_policy(system: str) -> bool:
    """Return whether the system prompt already states a document-only grounding policy."""
    normalized = system.lower()
    return "answer using only the provided documents" in normalized or "retrieval-augmented assistant" in normalized


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
        if not stripped or _should_skip_redundant_user_line(stripped, system_has_grounding, system_has_citation):
            continue
        filtered_lines.append(stripped)

    return _join_answer_scaffold_blocks(filtered_lines)


def _adapt_system_for_responses_export(system: str) -> str:
    """Drop OGX-runtime retrieval/citation text from the HPO system prompt."""
    return _strip_ogx_runtime_instructions(system)


def _adapt_static_user_for_responses_export(static_user: str) -> str:
    """Drop merged user supplements that OGX injects at file_search runtime."""
    if not static_user.strip():
        return ""

    adapted_lines: list[str] = []
    for line in static_user.splitlines():
        stripped = line.strip()
        if not stripped or _should_skip_user_export_line(stripped):
            continue
        cleaned = _strip_ogx_runtime_instructions(stripped)
        if cleaned:
            adapted_lines.append(cleaned)

    return _join_answer_scaffold_blocks(adapted_lines)


def _extract_static_user_instructions(user_message_text: str) -> str:
    """Return static instruction text from a HPO user template.

    Strips runtime slots (retrieved documents, question) that Responses API
    supplies via ``file_search`` and the user ``input`` message respectively.
    """
    if not user_message_text:
        return ""

    text = str(user_message_text).strip()
    if "{reference_documents}" in text:
        return _extract_static_user_from_reference_slot(text)

    prefix = _extract_static_user_without_reference_slot(text)
    return prefix


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
    system = _adapt_system_for_responses_export((generation.get("system_message_text") or "").strip())
    user_template = generation.get("user_message_text") or ""
    raw_system = (generation.get("system_message_text") or "").strip()

    static_user = _adapt_static_user_for_responses_export(
        _filter_static_user_for_responses(
            raw_system,
            _extract_static_user_instructions(user_template),
        ),
    )

    # Merge system and static user, handling empty cases
    if system and static_user:
        result = f"{system}\n\n{static_user}"
    else:
        result = system or static_user

    # Fallback for completely empty patterns (rare edge case)
    if not result or not result.strip() or _is_placeholder_only_export(result):
        return _EMPTY_SYSTEM_FALLBACK

    return result


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
        Language detection result (``{"code": "...", "name": "..."}``) stored
        on ``generation`` before building the Responses export.

    Returns
    -------
    dict
        Pattern definition suitable for JSON serialisation.
    """
    generation = pattern["settings"]["generation"]
    if detected_language:
        generation["detected_language"] = detected_language

    system_input = build_responses_system_input(generation)

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
