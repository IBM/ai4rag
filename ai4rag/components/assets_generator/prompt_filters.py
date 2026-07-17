# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Filter HPO prompts to remove OGX runtime injection duplicates.

OGX (OpenSearch GenAI eXperience) injects grounding, citation, and retrieval
instructions at runtime via benchmarking/rag/config.yaml. HPO (HyperParameter
Optimization) templates sometimes include similar phrases that must be removed
during Responses API export to avoid duplication.

This module provides filtering functions to strip OGX-owned content while
preserving HPO-specific persona, policy, and answer formatting rules.

Note
----
OGX phrase lists must stay synchronized with benchmarking/rag/config.yaml.
If OGX updates their injection strings, update the constants below.
"""

import re

# ============================================================================
# OGX Runtime Injection Strings
# ============================================================================
# Source: benchmarking/rag/config.yaml
# These phrases are injected by OGX at file_search runtime.
# Export must NOT duplicate them in responses_template.input[system].
# ============================================================================

# Citation-related phrases
CITATION_PREFIXES = (
    "You MUST cite sources",
    "Cite sources immediately",
)
CITATION_SUBSTRINGS = (
    "[1], [2]",
    "<|file-id|>",
    "cite as <|",
    "file citations",
    "document numbers for every factual claim",
)
# HPO citation fragments for filtering
HPO_CITATION_FRAGMENTS = (
    "You MUST cite sources using [1], [2], etc. matching the document numbers for every factual claim.",
    "You MUST cite sources using [1], [2], etc.",
    "You MUST cite sources using [1], [2].",
)

# Grounding/retrieval-related phrases
# Used in: sentence-level filtering (sentence_is_ogx_duplicative) and
# system grounding detection (_system_has_grounding_policy in pattern_builder.py)
GROUNDING_PREFIXES = (
    "Answer ONLY using information from the documents",
    "Answer ONLY using information from documents retrieved",
    "Answer using ONLY the provided documents",
    "Answer using ONLY information from documents",
    "Do not use outside knowledge",
    "If the retrieved documents do not contain",
    "If the documents do not contain",
)
# Used in: substring matching within sentences for partial phrase detection
GROUNDING_SUBSTRINGS = (
    "documents below",
    "retrieved via file search",
    "retrieved to help answer the user",
    "supporting information only in answering",
)
# Used in: whole-phrase removal from system prompts (strip_ogx_runtime_instructions)
SYSTEM_GROUNDING_PHRASES = (
    "Answer using ONLY the provided documents.",
    "Answer using ONLY information from documents retrieved via file search.",
)

# File search tool markers
# Used in: detecting OGX tool result wrappers in sentence-level filtering
FILE_SEARCH_MARKERS = (
    "file_search tool found",
    "BEGIN of file_search tool results",
    "END of file_search tool results",
    "The above results were retrieved to help answer",
    "Use them as supporting information only",
    "Do not add extra punctuation. Use only the file IDs",
)

# User template duplicate detection
# Used in: Pass 2 filtering (_should_skip_user_export_line in pattern_builder.py)
# OGX-owned lines that must never be exported regardless of system prompt content
USER_GROUNDING_SKIP_PREFIXES = (
    "Answer ONLY using information from the documents below",
    "Do not use outside knowledge",
    "If the documents do not contain the answer",
)
# Used in: Pass 1 filtering (_should_skip_redundant_user_line in pattern_builder.py)
# Only suppressed when system prompt already has grounding policy to avoid duplication. 
# They remain as defensive filters for custom HPO configurations that may use these phrasings.
USER_RAG_GROUNDING_PREFIXES = (
    "You are a specialized Retrieval Augmented Generation",
    "Prioritize correctness and ensure your response is grounded",
)

# Combined line prefixes for sentence-level filtering
OGX_DUPLICATIVE_LINE_PREFIXES = CITATION_PREFIXES + GROUNDING_PREFIXES + FILE_SEARCH_MARKERS

# Combined substrings for partial-match filtering
OGX_DUPLICATIVE_SUBSTRINGS = CITATION_SUBSTRINGS + GROUNDING_SUBSTRINGS


def collapse_whitespace(text: str) -> str:
    """Collapse repeated interior spaces after phrase removal.

    Parameters
    ----------
    text : str
        Text potentially containing multiple consecutive spaces.

    Returns
    -------
    str
        Text with interior whitespace collapsed to single spaces, stripped.
    """
    return re.sub(r" +", " ", text).strip()


def is_sentence_ogx_duplicative(sentence: str) -> bool:
    """Return whether a sentence duplicates OGX file_search runtime injection.

    Parameters
    ----------
    sentence : str
        Single sentence to check.

    Returns
    -------
    bool
        True if sentence matches OGX injection patterns.
    """
    stripped = sentence.strip().rstrip(".")
    if not stripped:
        return True
    if any(stripped.startswith(prefix.rstrip(".")) for prefix in OGX_DUPLICATIVE_LINE_PREFIXES):
        return True
    normalized = stripped.lower()
    return any(fragment.lower() in normalized for fragment in OGX_DUPLICATIVE_SUBSTRINGS)


def is_citation_related_line(line: str) -> bool:
    """Return whether an entire line should be dropped as citation guidance.

    Parameters
    ----------
    line : str
        Line of text to check.

    Returns
    -------
    bool
        True if line contains only citation instructions owned by OGX.
    """
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if any(stripped.startswith(prefix) for prefix in CITATION_PREFIXES):
        return True
    if any(fragment.lower() in lower for fragment in HPO_CITATION_FRAGMENTS):
        return True
    return any(sub.lower() in lower for sub in CITATION_SUBSTRINGS)


def filter_ogx_duplicative_sentences(line: str) -> str:
    """Remove OGX-duplicative sentences while keeping persona or policy sentences.

    Handles multi-sentence lines by filtering at sentence granularity.

    Parameters
    ----------
    line : str
        Line potentially containing multiple sentences.

    Returns
    -------
    str
        Line with OGX-duplicative sentences removed, or empty string if all filtered.
    """
    stripped = line.strip()
    if not stripped or is_citation_related_line(stripped):
        return ""

    # Split on ". " only — avoids breaking abbreviations such as "i.e.,"
    parts = [part.strip() for part in stripped.split(". ") if part.strip()]
    if len(parts) <= 1:
        if is_sentence_ogx_duplicative(stripped.rstrip(".")):
            return ""
        return stripped

    kept = [part.rstrip(".") for part in parts if not is_sentence_ogx_duplicative(part.rstrip("."))]
    if not kept:
        return ""

    result = ". ".join(kept)
    if stripped.endswith("."):
        result += "."
    return result


def normalize_answer_scaffold(line: str) -> str:
    """Drop citation hints from answer scaffolds; OGX owns citation via annotations.

    Parameters
    ----------
    line : str
        Line potentially containing answer scaffold with citation hints.

    Returns
    -------
    str
        Line with ", with citations" and "with citations" removed, whitespace normalized.
    """
    normalized = re.sub(r",?\s*with citations,?\s*", "", line)
    return collapse_whitespace(normalized)


def strip_ogx_runtime_instructions(text: str) -> str:
    """Remove text that OGX injects via file_search config at inference time.

    This is the main filtering function that orchestrates all OGX deduplication.

    Parameters
    ----------
    text : str
        Raw HPO prompt text (system or user message).

    Returns
    -------
    str
        Filtered text with OGX-duplicative content removed.
    """
    if not text.strip():
        return ""

    for phrase in SYSTEM_GROUNDING_PHRASES:
        text = text.replace(phrase, "").replace(phrase.rstrip("."), "")
    text = collapse_whitespace(text)

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if is_citation_related_line(stripped):
            continue

        cleaned = filter_ogx_duplicative_sentences(stripped)
        for fragment in HPO_CITATION_FRAGMENTS:
            if fragment in cleaned:
                cleaned = cleaned.replace(fragment, "").strip()
                break
        cleaned = normalize_answer_scaffold(cleaned)
        if cleaned:
            lines.append(cleaned)

    result = "\n".join(lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()
