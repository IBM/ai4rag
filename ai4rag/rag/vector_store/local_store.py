# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Helpers for throwaway, file-backed Milvus Lite vector stores.

Milvus Lite is the embedded, zero-server Milvus engine, selected simply by
pointing a :class:`~ai4rag.rag.vector_store.config.MilvusConfig` at a local file
path. This module wraps the common "disposable local index" pattern used by
model pre-selection and judge calibration, where a vector store must live only
for the duration of one evaluation and leave nothing behind on disk. It is the
local, zero-server replacement for the previously used ephemeral in-memory
Chroma store.
"""

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ai4rag import logger
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import MilvusLiteConfig
from ai4rag.rag.vector_store.get_vector_store import get_vector_store

__all__ = ["temporary_milvus_lite_store"]

#: Database file name created inside each throwaway store's private directory.
_DB_FILENAME = "store.db"


@contextmanager
def temporary_milvus_lite_store(
    embedding_model: BaseEmbeddingModel,
    collection_name: str | None = None,
) -> Iterator[BaseVectorStore]:
    """Yield a disposable, file-backed Milvus Lite vector store.

    Creates a private temporary directory, opens an embedded Milvus Lite store
    inside it (a :class:`~ai4rag.rag.vector_store.config.MilvusLiteConfig` whose
    ``db_path`` is a file in that directory), and yields the store. On exit —
    normal or exceptional — the client is closed on a best-effort basis (a
    failure is logged, never raised, so it cannot mask an exception from the
    caller or skip cleanup) and the whole directory (database file plus any
    auxiliary files Milvus Lite created) is unconditionally removed, so nothing
    is left behind.

    A fresh store bound to its own file keeps at most one Milvus Lite database
    open per caller at a time, matching the engine's single-writer model.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    collection_name : str | None, default=None
        Existing collection to reuse; must start with the ``ai4rag`` prefix. When
        omitted, a new compliant name is generated (see
        :func:`ai4rag.rag.vector_store.utils.resolve_collection_name`).

    Yields
    ------
    BaseVectorStore
        A Milvus Lite store bound to a temporary database file.
    """
    with tempfile.TemporaryDirectory(prefix="ai4rag-milvus-lite-") as tmp_dir:
        store: BaseVectorStore | None = None
        try:
            store = get_vector_store(
                embedding_model=embedding_model,
                config=MilvusLiteConfig(db_path=str(Path(tmp_dir) / _DB_FILENAME)),
                collection_name=collection_name,
            )
            yield store
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:
                    # Never let a close() failure skip the directory removal below
                    # (handled by TemporaryDirectory's own __exit__), or mask a
                    # real exception raised from inside the `with` block above.
                    logger.warning("Failed to close temporary Milvus Lite store at '%s'.", tmp_dir, exc_info=True)
