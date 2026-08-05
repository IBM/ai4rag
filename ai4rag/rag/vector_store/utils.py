# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import re
import secrets
import string
from collections import defaultdict
from collections.abc import Iterator, Sequence
from datetime import datetime, timezone

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

#: Mandatory namespace prefix for every ai4rag vector store collection.
#:
#: This prefix is the cross-backend isolation guard. Because every collection —
#: and, for pgvector, the physical table it maps to one-to-one — is required to
#: start with it, ai4rag can never create, reuse, or drop a table/collection it
#: does not own, even when pointed at a database shared with unrelated data.
COLLECTION_NAME_PREFIX = "ai4rag"

#: Maximum collection name length, bounded by the tightest identifier limit
#: across supported backends: PostgreSQL truncates identifiers at 63 bytes
#: (``NAMEDATALEN - 1``) and Chroma caps collection names at 63 characters.
#: Enforcing it up front turns a silent, collision-inducing truncation into an
#: explicit error.
_MAX_COLLECTION_NAME_LENGTH = 63

_COLLECTION_NAME_SUFFIX_ALPHABET = string.ascii_lowercase + string.digits
_COLLECTION_NAME_SUFFIX_LENGTH = 8


def sanitize_collection_name(name: str) -> str:
    """Coerce a name into a valid identifier for every supported backend.

    Replaces non-alphanumeric characters (except underscores) with underscores,
    so the result is usable verbatim as both a backend collection name and a
    physical SQL table name.

    Parameters
    ----------
    name : str
        Raw collection name to sanitize.

    Returns
    -------
    str
        The sanitized, identifier-safe collection name.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def generate_collection_name() -> str:
    """Generate a unique vector store collection name.

    Follows the convention ``<prefix>_<UTC timestamp>_<8 random chars>``,
    e.g. ``ai4rag_20260728153000_zxcvbnml``.

    Returns
    -------
    str
        A unique, convention-following collection name.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
    suffix = "".join(secrets.choice(_COLLECTION_NAME_SUFFIX_ALPHABET) for _ in range(_COLLECTION_NAME_SUFFIX_LENGTH))
    return f"{COLLECTION_NAME_PREFIX}_{timestamp}_{suffix}"


def resolve_collection_name(collection_name: str | None) -> str:
    """Resolve, validate, and sanitize a vector store collection name.

    Single entry point every backend uses to turn the optional caller-supplied
    ``collection_name`` into the concrete name it will create or reuse. It
    enforces the invariants that keep ai4rag stores safe and portable:

    * **Auto-generation** — when ``collection_name`` is ``None`` a fresh,
      convention-following name is generated (see :func:`generate_collection_name`).
    * **Namespace guard** — a caller-supplied name must start with
      :data:`COLLECTION_NAME_PREFIX`. This is the isolation boundary guaranteeing
      ai4rag only ever touches its own tables/collections; a non-compliant name
      is rejected rather than silently coerced, so mistakes surface immediately.
    * **Identifier safety** — the name is sanitized into a valid identifier
      (see :func:`sanitize_collection_name`) and bounded to
      :data:`_MAX_COLLECTION_NAME_LENGTH`, so it is usable verbatim as a backend
      collection *and* as a physical SQL table name.

    Parameters
    ----------
    collection_name : str | None
        Existing collection name to reuse, or ``None`` to generate a new one.

    Returns
    -------
    str
        The resolved, sanitized collection name.

    Raises
    ------
    ValueError
        If ``collection_name`` does not start with :data:`COLLECTION_NAME_PREFIX`,
        or if it exceeds :data:`_MAX_COLLECTION_NAME_LENGTH` characters.
    """
    if collection_name is None:
        return generate_collection_name()

    if not collection_name.startswith(COLLECTION_NAME_PREFIX):
        raise ValueError(
            f"Collection name {collection_name!r} must start with '{COLLECTION_NAME_PREFIX}'. "
            "This prefix namespaces ai4rag-managed collections so the store never "
            "reuses or drops data it does not own."
        )

    sanitized = sanitize_collection_name(collection_name)
    if len(sanitized) > _MAX_COLLECTION_NAME_LENGTH:
        raise ValueError(
            f"Collection name {sanitized!r} exceeds the maximum length of " f"{_MAX_COLLECTION_NAME_LENGTH} characters."
        )
    return sanitized


#: Supported search modes shared by every hybrid-capable backend: pure dense
#: vector search, or dense + sparse/keyword ``hybrid`` search.
_VALID_SEARCH_MODES = ("vector", "hybrid")

#: Supported hybrid reranking strategies shared by every hybrid-capable backend.
_VALID_RANKER_STRATEGIES = ("rrf", "weighted", "normalized")


def validate_search_params(
    search_mode: str,
    ranker_strategy: str | None,
    ranker_k: int | None,
    ranker_alpha: float | None,
) -> None:
    """Validate the search mode and hybrid ranker parameter combination.

    Backend-agnostic guard shared by every hybrid-capable vector store (e.g.
    Milvus, PGVector): it enforces that ranker parameters are only supplied for a
    hybrid search and that each is paired with its matching strategy, before any
    backend-specific query is issued.

    Parameters
    ----------
    search_mode : str
        How the search should be conducted. ``"vector"`` for dense embedding
        search only, or ``"hybrid"`` for both sparse & dense (hybrid) search.
    ranker_strategy : str | None
        Reranking strategy (function) used with hybrid search. One of
        ``"rrf"``, ``"weighted"``, or ``"normalized"``. Must be unset for
        non-hybrid search.
    ranker_k : int | None
        The smoothing constant in Reciprocal Rank Fusion (RRF). Valid only
        with ``ranker_strategy="rrf"``.
    ranker_alpha : float | None
        Weighting coefficient that determines how much the system trusts
        semantic (vector) search versus lexical (keyword/BM25) search. Valid
        only with ``ranker_strategy="weighted"``.

    Raises
    ------
    ValueError
        If ``search_mode`` is unknown, if any ranker parameter is supplied
        for a non-hybrid search, if ``ranker_strategy`` is missing or invalid
        for a hybrid search, or if ``ranker_k``/``ranker_alpha`` are paired
        with the wrong strategy.
    """
    if search_mode not in _VALID_SEARCH_MODES:
        raise ValueError(f"Invalid search_mode '{search_mode}'. Must be one of {_VALID_SEARCH_MODES}.")

    has_strategy = ranker_strategy is not None and ranker_strategy != ""
    has_k = ranker_k is not None and ranker_k > 0
    has_alpha = ranker_alpha is not None and ranker_alpha != 1

    if search_mode != "hybrid":
        if has_strategy:
            raise ValueError(
                f"ranker_strategy='{ranker_strategy}' is only valid when search_mode='hybrid', "
                f"but search_mode='{search_mode}'."
            )
        if has_k:
            raise ValueError(
                f"ranker_k={ranker_k} is only valid when search_mode='hybrid', but search_mode='{search_mode}'."
            )
        if has_alpha:
            raise ValueError(
                f"ranker_alpha={ranker_alpha} is only valid when search_mode='hybrid', "
                f"but search_mode='{search_mode}'."
            )
    else:
        if not has_strategy:
            raise ValueError("ranker_strategy must be set when search_mode='hybrid'.")
        if ranker_strategy not in _VALID_RANKER_STRATEGIES:
            raise ValueError(f"Invalid ranker_strategy='{ranker_strategy}'. Must be one of {_VALID_RANKER_STRATEGIES}.")
        if has_k and ranker_strategy != "rrf":
            raise ValueError(
                f"ranker_k={ranker_k} is only valid when ranker_strategy='rrf', "
                f"but ranker_strategy='{ranker_strategy}'."
            )
        if has_alpha and ranker_strategy != "weighted":
            raise ValueError(
                f"ranker_alpha={ranker_alpha} is only valid when ranker_strategy='weighted', "
                f"but ranker_strategy='{ranker_strategy}'."
            )


def resolve_embedding_dimension(embedding_model: BaseEmbeddingModel) -> int:
    """Read the embedding dimension from an embedding model's parameters.

    The model exposes its parameters either as a plain ``dict`` or as a params
    object; both shapes are handled so a backend can size its dense vector
    field/column to the model's output vectors.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model whose ``params`` carry the dense vector dimensionality.

    Returns
    -------
    int
        Dimensionality of the dense embedding vectors.
    """
    params = embedding_model.params
    if isinstance(params, dict):
        return params["embedding_dimension"]
    return params.embedding_dimension


def iter_unique_chunks(
    documents: Sequence[AI4RAGChunk],
    embeddings: Sequence[list[float]],
) -> Iterator[tuple[AI4RAGChunk, list[float]]]:
    """Yield ``(chunk, embedding)`` pairs, skipping duplicate ``chunk_id`` values.

    Shared dedup pass for every embedding-backed store (e.g. Milvus, PGVector):
    the first occurrence of a ``chunk_id`` wins and later duplicates are dropped
    with a warning, so a single row per id reaches the backend upsert.

    Parameters
    ----------
    documents : Sequence[AI4RAGChunk]
        Chunks to store, positionally aligned with ``embeddings``.
    embeddings : Sequence[list[float]]
        Dense vectors for ``documents``, in the same order.

    Yields
    ------
    tuple[AI4RAGChunk, list[float]]
        Each unique chunk paired with its embedding, in first-seen order.
    """
    seen_ids: set[str] = set()
    for doc, embedding in zip(documents, embeddings):
        if doc.chunk_id in seen_ids:
            logger.warning(
                "Skipping duplicate chunk_id: %s from document: %s", doc.chunk_id, doc.metadata.get("document_id")
            )
            continue
        seen_ids.add(doc.chunk_id)
        yield doc, embedding


def merge_window_into_a_document(window: list[AI4RAGChunk]) -> AI4RAGChunk:
    """
    Merges a list of chunks into a single chunk.
    If consecutive chunks have intersecting merged_text, the merged_text is merged to avoid duplications.

    Parameters
    ----------
    window : list[AI4RAGChunk]
        Ordered list of chunks for merging.

    Returns
    -------
    AI4RAGChunk
        Chunk that contains the merged text and metadata of the window chunks.
    """

    def merge_metadata(multiple_metadata: list[dict]) -> dict:
        """
        Merges a list of dictionaries (metadata) into one metadata.
        The keys remain the same but the values are changed into lists of values from all metadata.

        Parameters
        ----------
        multiple_metadata : list[dict]
            List of metadata dictionaries to be merged.

        Returns
        -------
        dict
            Merged metadata; each key maps to its single value, or to a sorted
            list of unique values when the sources disagree.
        """
        if len(multiple_metadata) == 1:
            return multiple_metadata[0]

        merged_metadata = defaultdict(set)
        for metadata in multiple_metadata:
            for key, value in metadata.items():
                if isinstance(value, list):
                    merged_metadata[key].update(value)
                else:
                    merged_metadata[key].add(value)

        result = {}
        for key, value_set in merged_metadata.items():
            value_list = sorted(value_set)
            if len(value_list) == 1:
                result[key] = value_list[0]
            else:
                result[key] = value_list
        return result

    def get_str2_without_intersecting_text(str1: str, str2: str) -> tuple[str, bool]:
        """
        Finds the intersecting merged_text between the suffix of str1 and the prefix of str2.

        Parameters
        ----------
        str1 : str
            The first string.

        str2 : str
            The second string.

        Returns
        -------
        tuple[str, bool]
            1. str2 without its intersection to str1
            2. whether there was an intersection or not
        """
        # Start checking from the longest possible overlap to the shortest
        for i in range(min(len(str1), len(str2)), 0, -1):
            if str1[-i:] == str2[:i]:
                return str2[i:], True
        return str2, False

    def merge_texts(texts: list[str]) -> str:
        """
        Merges a list of texts into a single text string.
        If consecutive texts have intersecting parts, the text is merged to avoid duplications.

        Parameters
        ----------
        texts : list[str]
            Ordered list of text strings to be merged.

        Returns
        -------
        str
            Single string that contains the merged text.
        """
        merged_text = ""
        for text in texts:
            text_to_add, has_intersection = get_str2_without_intersecting_text(merged_text, text)
            if merged_text and not has_intersection:
                merged_text += " "  # Add a space between non-overlapping texts (chunks)
            merged_text += text_to_add
        return merged_text

    texts = [chunk.text for chunk in window]
    merged_text = merge_texts(texts)

    metadata = [chunk.metadata for chunk in window]
    merged_metadata = merge_metadata(metadata)

    return AI4RAGChunk(text=merged_text, metadata=merged_metadata)
