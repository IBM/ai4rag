# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import tempfile
import weakref
from pathlib import Path
from typing import Any

from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
    WeightedRanker,
)

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import MilvusConfig
from ai4rag.rag.vector_store.utils import iter_unique_chunks, resolve_embedding_dimension, validate_search_params

__all__ = ["MilvusVectorStore"]


class MilvusVectorStore(BaseVectorStore):
    """Vector store backed by a remote Milvus instance via ``pymilvus``.

    Supports both pure vector search and hybrid search (dense + BM25 sparse)
    with RRF or weighted reranking, using Milvus native server-side fusion.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    config : MilvusConfig
        Connection parameters for the Milvus server. TLS is enabled by an
        ``https://`` URI; when ``config.server_cert`` is set, its PEM text is
        written to a temporary file and passed to ``MilvusClient`` as
        ``server_pem_path`` for certificate verification.
    distance_metric : str
        Distance metric for vector similarity (default ``"cosine"``).
    collection_name : str | None
        Existing collection to reuse; must start with the ``ai4rag`` prefix. When
        omitted, a new compliant name is generated (see
        :func:`ai4rag.rag.vector_store.utils.resolve_collection_name`).
    """

    _BATCH_SIZE = 2048

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: MilvusConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
    ):
        """Initialize the store, open a client, and ensure the collection exists.

        A ``MilvusClient`` is built from *config*; when ``config.server_cert`` is
        set, its PEM text is materialized to a temporary file and passed as
        ``server_pem_path`` for TLS verification. The target collection — with
        its dense, sparse/BM25, and JSON fields — is created only when it does
        not already exist.

        Parameters
        ----------
        embedding_model : BaseEmbeddingModel
            Model used to embed documents and queries.
        config : MilvusConfig
            Connection parameters for the Milvus server.
        distance_metric : str, default="cosine"
            Distance metric used for dense vector similarity.
        collection_name : str | None, default=None
            Existing collection to reuse; must start with the ``ai4rag`` prefix.
            When omitted, a new compliant name is generated.
        """
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(self.embedding_model)

        connect_kwargs: dict[str, Any] = {"uri": config.uri}
        if config.token:
            connect_kwargs["token"] = config.token
        if config.server_cert:
            connect_kwargs["server_pem_path"] = self._write_cert_tempfile(config.server_cert)
        self._client = MilvusClient(**connect_kwargs)

        if not self._client.has_collection(self._collection_name):
            self._create_collection()

    def _write_cert_tempfile(self, cert: str) -> str:
        """Materialize a PEM certificate to a temporary file for pymilvus TLS.

        ``MilvusClient`` accepts a filesystem path (``server_pem_path``) rather
        than raw certificate bytes, so an in-memory PEM must be written to disk.
        ``NamedTemporaryFile`` creates the file with owner-only permissions, and
        a :func:`weakref.finalize` hook removes it once this store is
        garbage-collected (or at interpreter exit), so no certificate material
        lingers in the temp directory.

        Parameters
        ----------
        cert : str
            PEM-encoded server/CA certificate text.

        Returns
        -------
        str
            Path to the temporary certificate file.
        """
        # delete=False keeps the file after the context manager closes it; a
        # weakref.finalize hook removes it when this store is garbage-collected.
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="ai4rag-milvus-cert-", suffix=".pem", delete=False
        ) as cert_file:
            cert_file.write(cert)
            cert_path = Path(cert_file.name)

        weakref.finalize(self, cert_path.unlink, missing_ok=True)
        return str(cert_path)

    def _create_collection(self) -> None:
        """Create the Milvus collection with its schema, indexes, and BM25 function.

        Defines the schema (primary ``chunk_id``, analyzed ``content``, dense
        ``vector``, raw ``chunk_content`` JSON, and a ``sparse`` BM25 vector),
        attaches a FLAT/COSINE index on the dense field and a sparse inverted
        BM25 index on the sparse field, and registers the BM25 function that
        derives the sparse vector from ``content``.
        """
        schema = self._client.create_schema()
        schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535, enable_analyzer=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._embedding_dimension)
        schema.add_field(field_name="chunk_content", datatype=DataType.JSON)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["content"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)

        logger.info(
            "Creating Milvus collection: %s (dim=%d) with following schema: %s",
            self._collection_name,
            self._embedding_dimension,
            schema.to_dict(),
        )
        self._client.create_collection(
            self._collection_name,
            schema=schema,
            index_params=index_params,
        )

    def search(
        self,
        query: str,
        k: int = 5,
        include_scores: bool = False,
        search_mode: str = "vector",
        ranker_strategy: str | None = None,
        ranker_k: int | None = None,
        ranker_alpha: float | None = None,
        **kwargs,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Search for chunks relevant to *query*.

        Parameters
        ----------
        query : str
            Search query text.
        k : int, default=5
            Number of results to return.
        include_scores : bool, default=False
            Whether to include similarity scores in the results.
        search_mode : str, default="vector"
            ``"vector"`` for dense-only search or ``"hybrid"`` for dense + BM25
            sparse search.
        ranker_strategy : str | None, default=None
            Hybrid ranker: ``"rrf"``, ``"weighted"``, or ``"normalized"``.
        ranker_k : int | None, default=None
            RRF smoothing constant (``k``).
        ranker_alpha : float | None, default=None
            Weighted blend factor (``0`` = keyword, ``1`` = vector).
        **kwargs : Any
            Accepted for interface compatibility; ignored by this backend.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Matched chunks, optionally paired with their scores.
        """
        validate_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha)

        if search_mode == "hybrid":
            return self._search_hybrid(query, k, include_scores, ranker_strategy, ranker_k, ranker_alpha)
        return self._search_vector(query, k, include_scores)

    def _search_vector(
        self, query: str, k: int, include_scores: bool
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run a pure dense-vector similarity search.

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.
        include_scores : bool
            Whether to pair each returned chunk with its similarity score.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Matched chunks, optionally paired with their scores.
        """
        embedding = self.embedding_model.embed_query(query)
        search_res = self._client.search(
            collection_name=self._collection_name,
            data=[embedding],
            anns_field="vector",
            limit=k,
            output_fields=["chunk_content"],
        )

        return self._parse_milvus_results(search_res[0], include_scores)

    def _search_hybrid(
        self,
        query: str,
        k: int,
        include_scores: bool,
        ranker_strategy: str | None,
        ranker_k: int | None,
        ranker_alpha: float | None,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run a hybrid dense + sparse (BM25) search with server-side fusion.

        Issues a dense ``AnnSearchRequest`` on the ``vector`` field and a sparse
        request on the ``sparse`` field, then fuses the two result sets on the
        Milvus server with either a weighted ranker (when
        ``ranker_strategy == "weighted"``) or an RRF ranker (otherwise).

        Parameters
        ----------
        query : str
            Search query text.
        k : int
            Number of results to return.
        include_scores : bool
            Whether to pair each returned chunk with its fused score.
        ranker_strategy : str | None
            Hybrid ranker strategy. ``"weighted"`` selects a weighted ranker;
            any other value selects RRF.
        ranker_k : int | None
            RRF smoothing constant; falls back to ``60`` when unset or non-positive.
        ranker_alpha : float | None
            Weighted blend factor for the dense field; falls back to ``0.5`` when
            unset. The sparse field receives ``1 - alpha``.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Fused chunks, optionally paired with their scores.
        """
        embedding = self.embedding_model.embed_query(query)

        dense_req = AnnSearchRequest(data=[embedding], anns_field="vector", param={"nprobe": 10}, limit=k)
        sparse_req = AnnSearchRequest(data=[query], anns_field="sparse", param={"drop_ratio_search": 0.2}, limit=k)

        if ranker_strategy == "weighted":
            alpha = ranker_alpha if ranker_alpha is not None and ranker_alpha != 1 else 0.5
            ranker = WeightedRanker(alpha, 1 - alpha)
        else:
            ranker = RRFRanker(k=ranker_k if ranker_k is not None and ranker_k > 0 else 60)

        search_res = self._client.hybrid_search(
            collection_name=self._collection_name,
            reqs=[dense_req, sparse_req],
            ranker=ranker,
            limit=k,
            output_fields=["chunk_content"],
        )

        return self._parse_milvus_results(search_res[0], include_scores)

    @staticmethod
    def _parse_milvus_results(
        results: list[dict],
        include_scores: bool,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Convert raw Milvus hits into :class:`AI4RAGChunk` objects.

        Parameters
        ----------
        results : list[dict]
            Per-hit result dictionaries from a (hybrid) search, each carrying an
            ``entity.chunk_content`` payload and a ``distance`` score.
        include_scores : bool
            Whether to pair each chunk with its score.

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Parsed chunks, optionally paired with their scores.
        """
        chunks_and_scores: list[tuple[AI4RAGChunk, float]] = []
        for res in results:
            chunk_data = res["entity"]["chunk_content"]
            chunk = AI4RAGChunk(text=chunk_data["content"], metadata=chunk_data.get("metadata", {}))
            chunks_and_scores.append((chunk, res["distance"]))

        if include_scores:
            return chunks_and_scores
        return [chunk for chunk, _ in chunks_and_scores]

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
        """Embed, deduplicate, and upsert chunks into Milvus.

        Duplicate ``chunk_id`` values within *documents* are skipped (first
        occurrence wins) and logged. Rows are upserted in batches.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and stored.
        **kwargs : Any
            Optional overrides. ``batch_size`` (int) sets the upsert batch size
            (default :attr:`_BATCH_SIZE`).
        """
        if not documents:
            return

        embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])

        data: list[dict[str, Any]] = []
        for doc, embedding in iter_unique_chunks(documents, embeddings):
            data.append(
                {
                    "chunk_id": doc.chunk_id,
                    "content": doc.text,
                    "vector": embedding,
                    "chunk_content": {
                        "content": doc.text,
                        "metadata": doc.metadata,
                        "chunk_id": doc.chunk_id,
                    },
                }
            )

        batch_size = kwargs.get("batch_size", self._BATCH_SIZE)
        for idx in range(0, len(data), batch_size):
            self._client.upsert(self._collection_name, data=data[idx : idx + batch_size])

    def clean_collection(self) -> None:
        """Drop the Milvus collection."""
        self._client.drop_collection(self._collection_name)
