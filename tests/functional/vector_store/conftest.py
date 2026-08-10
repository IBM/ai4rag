# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Shared fixtures for the vector store functional (semantic-retrieval) suite.

Each backend gets its own module (``test_chroma``, ``test_milvus``,
``test_pgvector``) rather than a single parametrized test, because the backends
diverge:

* **Chroma** runs fully in-memory and needs no server; it has no lexical search,
  so it only ever exercises dense semantic retrieval.
* **Milvus** and **pgvector** require a live database and will additionally grow
  lexical / hybrid-search tests that do not apply to Chroma.

Per-backend modules keep each backend's setup, teardown, and future
backend-specific tests isolated, while everything the backends *share* lives
here: the story data, the real MaaS embedding model, the semantic-retrieval
assertion, and a staleness-tolerant retry helper.

Why a real embedding model (and why this is a *functional*, not *integration*,
suite): asserting that a natural-language question retrieves the passage that
answers it is only meaningful with a model that captures meaning. The
``tests/integration`` suite, by contrast, uses local deterministic embeddings
and asserts the storage-and-search contract against a real database. The
dividing line is the external dependency: integration needs only a database;
functional additionally needs the MaaS model service.

The embedding model is gated on MaaS credentials — any test that requests it is
skipped when they are absent. Each backend module adds its own skip for the
database connection settings it requires. Connection settings are read from
environment variables loaded from ``.env.local`` at the repository root.
"""

import os
import time
from collections.abc import Callable
from typing import TypeVar

import pytest
from dotenv import find_dotenv, load_dotenv

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from dev_utils.utils import build_maas_model, create_dev_maas_client

# Load MaaS + backend connection settings before any skip guard is evaluated.
load_dotenv(find_dotenv(".env.local"))

#: Identifier shared by every passage: the story is one document, chunked into
#: sequential passages (mirroring how real documents are chunked for RAG).
STORY_DOCUMENT_ID = "the-lighthouse-of-vellmar"

#: A short story split into topically distinct passages. Each passage covers a
#: single, self-contained fact, so exactly one passage answers each question in
#: :data:`STORY_QUESTIONS` — making "the right passage is retrieved" an
#: unambiguous, model-agnostic assertion.
STORY_PASSAGES = (
    "Mira Halloran served as the lighthouse keeper of Vellmar, a windswept island off the northern "
    "coast, and she tended its light faithfully for twenty-three years.",
    "The tower was crowned by a slowly rotating brass lamp that burned whale oil, casting a beam that "
    "sailors could see from forty nautical miles away on a clear night.",
    "On a violent November evening the lamp's clockwork rotation mechanism seized without warning, and "
    "for the first time in ten years the great light went completely dark.",
    "Determined that no ship should wreck on the rocks, Mira climbed the spiral staircase and turned the "
    "heavy lamp by hand with an iron crank all night until dawn broke.",
    "To honour her courage, the maritime guild presented Mira with a silver compass engraved with the "
    "words 'Steady Light'.",
    "Her only companion on the lonely island was a one-eyed grey cat named Barnacle, who slept among the "
    "coal sacks and hunted mice in the rope store.",
)

#: ``(question, expected_passage_index)`` pairs. Each question is phrased in words
#: distinct from its target passage, so passing the check requires *semantic*
#: retrieval rather than lexical overlap.
STORY_QUESTIONS = (
    ("For how many years did Mira look after the Vellmar lighthouse?", 0),
    ("What kind of fuel did the lighthouse lamp burn?", 1),
    ("What failure caused the lighthouse to go dark during the storm?", 2),
    ("How did Mira keep the beam shining after the mechanism broke?", 3),
    ("What award did the maritime guild give Mira for her bravery?", 4),
    ("What was the name of the keeper's cat?", 5),
)

T = TypeVar("T")


def _maas_credentials_present() -> bool:
    """Return whether both MaaS connection variables are set."""
    return bool(os.environ.get("MAAS_BASE_URL") and os.environ.get("MAAS_API_KEY"))


@pytest.fixture(scope="session")
def embedding_model() -> OpenAIEmbeddingModel:
    """Provide the shared, real MaaS embedding model; skip if MaaS is not configured.

    ``embedding_dimension`` and ``context_length`` are supplied explicitly so the
    model performs no auto-detection API calls at construction time. The story
    passages are far shorter than any context length, so the exact value only has
    to satisfy the model's minimum; it never triggers truncation.
    """
    if not _maas_credentials_present():
        pytest.skip("MAAS_BASE_URL / MAAS_API_KEY not set; semantic retrieval needs a real embedding model.")

    client = create_dev_maas_client()
    model_id = os.environ.get("AI4RAG_TEST_EMBEDDING_MODEL", "bge-m3")
    dimension = int(os.environ.get("AI4RAG_TEST_EMBEDDING_DIMENSION", "1024"))
    context_length = int(os.environ.get("AI4RAG_TEST_EMBEDDING_CONTEXT_LENGTH", "8192"))
    return build_maas_model(
        client,
        model_id=model_id,
        model_type="embedding",
        embedding_params={"embedding_dimension": dimension, "context_length": context_length},
    )


@pytest.fixture(scope="session")
def story_chunks() -> list[AI4RAGChunk]:
    """Provide the story passages as chunks of a single document.

    Each chunk carries ``document_id`` and ``sequence_number`` metadata; the
    sequence number equals the chunk's index, so ``story_chunks[i]`` is the
    passage that :data:`STORY_QUESTIONS` refers to as index ``i``.
    """
    return [
        AI4RAGChunk(text=text, metadata={"document_id": STORY_DOCUMENT_ID, "sequence_number": i})
        for i, text in enumerate(STORY_PASSAGES)
    ]


@pytest.fixture(scope="session")
def check_retrieval(story_chunks) -> Callable[[Callable[[str], list]], None]:
    """Return an assertion helper verifying every question retrieves its passage.

    The returned ``check(search)`` calls ``search(question)`` for each story
    question — where ``search`` maps a question to the store's ranked results —
    and asserts the top-ranked result is the passage that answers it. Taking
    ``search`` (rather than the store) lets a backend wrap the call, e.g. in
    :func:`retry` for eventually-consistent reads, without duplicating the
    assertion logic.
    """

    def _check(search: Callable[[str], list]) -> None:
        for question, expected_index in STORY_QUESTIONS:
            expected = story_chunks[expected_index]
            results = search(question)
            assert results, f"no results returned for question: {question!r}"
            assert results[0].text == expected.text, (
                f"question {question!r} retrieved the wrong passage:\n"
                f"  got:      {results[0].text!r}\n"
                f"  expected: {expected.text!r}"
            )

    return _check


@pytest.fixture(scope="session")
def retry() -> Callable[..., T]:
    """Provide a helper that polls a callable until it returns a truthy value.

    Milvus serves searches under bounded-staleness consistency by default, so
    freshly upserted rows may not be immediately visible. Wrapping the read in
    this helper makes such assertions robust without penalising the
    strongly-consistent backends, for which the first attempt already succeeds.

    Returns
    -------
    Callable
        ``retry(fn, *, attempts=20, delay=0.5)`` — calls ``fn`` up to ``attempts``
        times, sleeping ``delay`` seconds between attempts, and returns the first
        truthy result (or the last result if none was truthy).
    """

    def _retry(fn: Callable[[], T], *, attempts: int = 20, delay: float = 0.5) -> T:
        result = fn()
        for _ in range(attempts - 1):
            if result:
                return result
            time.sleep(delay)
            result = fn()
        return result

    return _retry
