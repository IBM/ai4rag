# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Shared fixtures for the vector store integration suite.

Unlike the unit tests — which run against in-memory or mocked backends — the
modules in this package exercise the concrete vector stores against **real,
externally provisioned databases** (Chroma server, Milvus, PostgreSQL+pgvector).
Each backend module is skipped unless the connection settings for that backend
are present in the environment, so the suite is safe to run anywhere: it simply
skips the backends that are not reachable.

Connection settings are read from environment variables, loaded here once from a
``.env.local`` file at the repository root (see the ``*Config.from_env``
classmethods for the exact variable names each backend consumes).

Embeddings are produced by a small, fully local :class:`DeterministicEmbeddingModel`
rather than a hosted embedding service. This keeps the integration tests focused
on the database round-trip (create → index → search → drop) without depending on
model-serving credentials, while still giving reproducible, assertable search
ordering: identical text always embeds to the identical vector, so a query for a
stored chunk's exact text is guaranteed to rank that chunk first under cosine
similarity.
"""

import hashlib
import time
from collections.abc import Callable
from typing import TypeVar

import pytest
from dotenv import find_dotenv, load_dotenv

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel

# Load real-backend connection settings before any module-level ``skipif`` guard
# is evaluated. conftest.py is imported before the test modules in its directory,
# so the ``*_HOST`` / ``*_URI`` variables the guards read are populated in time.
load_dotenv(find_dotenv(".env.local"))

#: Dimensionality of the deterministic test embeddings. Kept small to keep index
#: builds cheap, and well under pgvector's 2000-dimension HNSW ceiling.
EMBEDDING_DIMENSION = 32

#: Distinct, single-topic sentences with no shared vocabulary, so cosine
#: similarity between any two is comfortably below the self-similarity of an
#: exact-text query. This makes "a query for a chunk's own text ranks it first"
#: a deterministic, backend-independent assertion.
SAMPLE_TEXTS = (
    "The mitochondria is the powerhouse of the cell.",
    "Photosynthesis converts sunlight into chemical energy.",
    "Newton's laws describe the motion of physical bodies.",
    "The French Revolution began in the year 1789.",
    "Quantum entanglement links the states of distant particles.",
)

T = TypeVar("T")


class DeterministicEmbeddingModel(BaseEmbeddingModel[None, dict[str, int]]):
    """Local, hash-based embedding model with reproducible vectors.

    Each text is mapped to a fixed-length vector derived from its SHA-256 digest,
    so identical texts always embed to identical vectors (cosine distance ``0``)
    while distinct texts embed to distinct, non-parallel vectors. Every component
    is strictly positive, so no vector is the zero vector (for which cosine
    similarity is undefined). This yields deterministic nearest-neighbour ordering
    without contacting any embedding service.

    Parameters
    ----------
    dimension : int, default=:data:`EMBEDDING_DIMENSION`
        Dimensionality of the produced dense vectors.
    """

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        super().__init__(
            client=None, model_id="integration-deterministic-embed", params={"embedding_dimension": dimension}
        )

    def _vector(self, text: str) -> list[float]:
        """Derive a deterministic, strictly-positive vector for *text*.

        The 32-byte SHA-256 digest is extended by re-hashing with an incrementing
        counter until at least ``embedding_dimension`` bytes are available, then
        each byte is mapped into ``[1.0, 2.0]``.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        list[float]
            The deterministic embedding vector.
        """
        dimension = self.params["embedding_dimension"]
        components: list[float] = []
        counter = 0
        while len(components) < dimension:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            components.extend(1.0 + byte / 255.0 for byte in digest)
            counter += 1
        return components[:dimension]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)


@pytest.fixture(scope="session")
def embedding_model() -> DeterministicEmbeddingModel:
    """Provide the shared deterministic embedding model for every backend."""
    return DeterministicEmbeddingModel()


@pytest.fixture(scope="session")
def sample_chunks() -> list[AI4RAGChunk]:
    """Provide the shared sample chunks used to populate each store.

    The chunks carry ``document_id`` and ``sequence_number`` metadata so search
    results can be asserted to round-trip their metadata across every backend.
    Session-scoped and treated as read-only: ``add_documents`` never mutates the
    chunks it is given.
    """
    return [
        AI4RAGChunk(text=text, metadata={"document_id": f"doc_{i}", "sequence_number": i})
        for i, text in enumerate(SAMPLE_TEXTS)
    ]


@pytest.fixture(scope="session")
def retry() -> Callable[..., T]:
    """Provide a helper that polls a callable until it returns a truthy value.

    Some backends (notably Milvus under bounded-staleness consistency) do not
    guarantee that freshly upserted rows are immediately visible to a search.
    Wrapping the read in this helper makes such assertions robust without
    penalising the strongly-consistent backends, for which the first attempt
    already succeeds.

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
