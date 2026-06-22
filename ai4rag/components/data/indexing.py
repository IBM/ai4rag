# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import logging
from pathlib import Path

from docling_core.types.doc.document import DoclingDocument
from ogx_client import OgxClient

from ai4rag.rag.chunking import DoclingChunker, LangChainChunker
from ai4rag.rag.embedding.ogx import OGXEmbeddingModel, OGXEmbeddingParams
from ai4rag.rag.vector_store.ogx import OGXVectorStore

_logger = logging.getLogger(__name__)

SUPPORTED_DISTANCE_METRICS = ("cosine", "euclidean")
SUPPORTED_CHUNKING_METHODS = ("recursive", "hybrid")
SUPPORTED_CHUNK_SIZE_RANGE = (128, 2048)


def index_documents(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    extracted_text_dir: str | Path,
    embedding_model_id: str,
    vector_io_provider_id: str,
    ogx_client: OgxClient,
    embedding_params: dict | None = None,
    distance_metric: str = "cosine",
    chunking_method: str = "recursive",
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    batch_size: int = 20,
    collection_name: str | None = None,
) -> int:
    """Chunk, embed, and index extracted documents into a vector store.

    Reads DoclingDocument JSON files from *extracted_text_dir*, splits them
    into chunks, computes embeddings via OGX, and inserts the resulting
    vectors into the configured vector store.  Documents are processed in
    batches to bound memory consumption.

    Parameters
    ----------
    extracted_text_dir
        Directory containing DoclingDocument JSON files produced by the
        text extraction stage.
    embedding_model_id
        Identifier of the embedding model served by OGX.
    vector_io_provider_id
        OGX provider identifier for the vector database backend.
    ogx_client
        Pre-configured :class:`OgxClient` instance.
    embedding_params
        Optional dictionary forwarded to :class:`OGXEmbeddingParams`.
    distance_metric
        Vector distance metric (``"cosine"`` or ``"euclidean"``).
    chunking_method
        Chunking strategy: ``"recursive"`` (LangChain) or ``"hybrid"``
        (Docling structure-aware).
    chunk_size
        Maximum chunk size in tokens.  Must be in the range 128--2048.
    chunk_overlap
        Token overlap between consecutive chunks (only used with the
        ``"recursive"`` method).
    batch_size
        Number of documents per processing batch.  ``0`` processes all
        documents in a single batch.
    collection_name
        Name of an existing vector-store collection to reuse.  When
        ``None``, a new collection is created.

    Returns
    -------
    int
        Total number of chunks indexed.

    Raises
    ------
    ValueError
        If any of the validated parameters are out of range.
    TypeError
        If *chunk_size* or *chunk_overlap* have incorrect types.
    """
    _validate_inputs(
        embedding_model_id, vector_io_provider_id, distance_metric, chunking_method, chunk_size, chunk_overlap
    )

    params = OGXEmbeddingParams(**(embedding_params or {}))

    base = Path(extracted_text_dir)
    paths = sorted(p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".json")
    total_documents = len(paths)
    _logger.info("Found %d documents to index", total_documents)

    if total_documents == 0:
        _logger.warning("No documents found in %s", extracted_text_dir)
        return 0

    chunker = _create_chunker(chunking_method, chunk_size, chunk_overlap)
    embedding_model = OGXEmbeddingModel(client=ogx_client, model_id=embedding_model_id, params=params)

    collection_kwargs = {"reuse_collection_name": collection_name} if collection_name is not None else {}
    ogx_vectorstore = OGXVectorStore(
        embedding_model=embedding_model,
        client=ogx_client,
        provider_id=vector_io_provider_id,
        distance_metric=distance_metric,
        **collection_kwargs,
    )

    effective_batch_size = batch_size if batch_size > 0 else total_documents
    total_chunks = 0
    num_batches = (total_documents + effective_batch_size - 1) // effective_batch_size

    for start in range(0, total_documents, effective_batch_size):
        batch_paths = paths[start : start + effective_batch_size]
        batch_documents = [DoclingDocument.load_from_json(p) for p in batch_paths]
        batch_chunks = chunker.split_documents(batch_documents)
        ogx_vectorstore.add_documents(batch_chunks)
        total_chunks += len(batch_chunks)
        batch_num = start // effective_batch_size + 1
        _logger.info(
            "Batch %d/%d: indexed %d documents (%d chunks), total chunks so far: %d",
            batch_num,
            num_batches,
            len(batch_documents),
            len(batch_chunks),
            total_chunks,
        )

    _logger.info("Documents indexing finished: %d documents, %d chunks", total_documents, total_chunks)
    return total_chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_inputs(
    embedding_model_id: str,
    vector_io_provider_id: str,
    distance_metric: str,
    chunking_method: str,
    chunk_size: int,
    chunk_overlap: int | float,
) -> None:
    """Validate all user-facing parameters before processing begins."""
    if not embedding_model_id:
        raise ValueError("embedding_model_id must be a non-empty string.")

    if not vector_io_provider_id or not vector_io_provider_id.strip():
        raise ValueError("vector_io_provider_id must be a non-empty string.")

    if distance_metric not in SUPPORTED_DISTANCE_METRICS:
        raise ValueError(
            f"distance metric {distance_metric!r} is not supported, "
            f"supported types are {SUPPORTED_DISTANCE_METRICS}."
        )

    if chunking_method not in SUPPORTED_CHUNKING_METHODS:
        raise ValueError(
            f"chunking_method {chunking_method!r} is not supported, "
            f"supported methods are {SUPPORTED_CHUNKING_METHODS}."
        )

    if not isinstance(chunk_size, int):
        raise TypeError("chunk_size must be an integer.")

    lo, hi = SUPPORTED_CHUNK_SIZE_RANGE
    if not lo <= chunk_size <= hi:
        raise ValueError(f"chunk_size must be an integer in the range {lo} to {hi}.")

    if not isinstance(chunk_overlap, (int, float)):
        raise TypeError("chunk_overlap must be a numerical value.")


def _create_chunker(method: str, chunk_size: int, chunk_overlap: int) -> DoclingChunker | LangChainChunker:
    """Instantiate the appropriate chunker for the given method."""
    if method == "hybrid":
        return DoclingChunker(max_tokens=chunk_size)
    return LangChainChunker(method=method, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
