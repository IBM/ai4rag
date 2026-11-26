# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# pylint: disable=too-many-lines
import asyncio
import hashlib
import json
import logging
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
from threading import Lock
from time import perf_counter
from typing import Any

import neo4j
from docling_core.types.doc import DoclingDocument
from neo4j_graphrag.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.components.types import Neo4jGraph
from neo4j_graphrag.retrievers import VectorCypherRetriever, VectorRetriever
from neo4j_graphrag.types import RetrieverResultItem

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.neo4j_utils import (
    KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT,
    KG_BATCH_RESPONSE_FORMAT,
    KG_MAX_DESCRIPTION_WORDS,
    KG_MAX_ENTITIES_PER_CHUNK,
    KG_MAX_RELATIONSHIPS_PER_CHUNK,
    Neo4jGraphSchema,
    _build_graph_retrieval_query,
    _CanonicalKGWriter,
    _collection_vector_index_name,
    _EmbedderAdapter,
    _LLMAdapter,
    _validate_kg_extraction_config,
    _validate_neo4j_search_params,
    is_rate_limit_llm_error,
)
from ai4rag.rag.vector_store.utils import (
    iter_unique_chunks,
    materialize_ca_cert,
    resolve_embedding_dimension,
)

__all__ = ["Neo4jGraphRetrievalConfig", "Neo4jGraphStore", "Neo4jMemoryError"]

_SCHEMA = Neo4jGraphSchema

_NEO4J_GRAPHRAG_LOGGERS = (
    "neo4j_graphrag",
    "neo4j_graphrag.experimental.pipeline.config.runner",
)


@contextmanager
def _suppress_neo4j_graphrag_info_logs() -> Iterator[None]:
    """Prevent GraphRAG from logging document text passed to the KG pipeline.

    ``SimpleKGPipeline`` logs its complete run parameters at INFO level. Those
    parameters include every input chunk's text, so retain only warnings and
    errors while the pipeline processes user content.
    """
    configured_loggers = [logging.getLogger(name) for name in _NEO4J_GRAPHRAG_LOGGERS]
    previous_levels = [configured_logger.level for configured_logger in configured_loggers]
    try:
        for configured_logger in configured_loggers:
            configured_logger.setLevel(max(configured_logger.getEffectiveLevel(), logging.WARNING))
        yield
    finally:
        for configured_logger, previous_level in zip(configured_loggers, previous_levels):
            configured_logger.setLevel(previous_level)


_CUSTOM_CA_URI_SCHEMES = {"neo4j+s": "neo4j", "bolt+s": "bolt"}


class _KGResponseFormatError(ValueError):
    """An LLM response could not be decoded as the requested KG batch schema."""


class _KGExtractionDegradedError(RuntimeError):
    """KG extraction exceeded the caller's tolerated partial-failure budget."""


class _KGExtractionRateLimitedError(_KGExtractionDegradedError):
    """The endpoint rejected KG extraction requests until the breaker tripped."""


class Neo4jMemoryError(RuntimeError):
    """Neo4j could not complete a graph query within its configured heap.

    Raised instead of letting the driver retry a deterministic heap exhaustion
    until its retry budget expires. The message names the knobs that reduce the
    query's working set so operators do not have to read driver diagnostics.
    """


_NEO4J_MEMORY_ERROR_MARKERS = (
    "OutOfMemoryError",
    "51N36",
)


def _is_neo4j_memory_error(error: BaseException) -> bool:
    """Return whether *error* reports Neo4j running out of heap for a query."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [error]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        code = str(getattr(current, "code", "") or "")
        if any(marker in code or marker in str(current) for marker in _NEO4J_MEMORY_ERROR_MARKERS):
            return True
        pending.extend((current.__cause__, current.__context__))
    return False


def _normalize_custom_ca_uri(uri: str) -> str:
    """Return the driver-compatible URI for a connection using a custom CA.

    The Neo4j driver prohibits ``trusted_certificates`` with its ``+s`` URI
    schemes. Replacing only the scheme preserves the endpoint and whether the
    caller requested routing (``neo4j``) or a direct connection (``bolt``);
    :class:`Neo4jGraphStore` then supplies explicit TLS settings.
    """
    scheme, separator, remainder = uri.partition(":")
    return f"{_CUSTOM_CA_URI_SCHEMES.get(scheme.lower(), scheme)}{separator}{remainder}"


@dataclass(frozen=True, kw_only=True)
class Neo4jGraphRetrievalConfig:
    """Controls graph expansion during `Neo4jGraphStore` retrieval."""

    route_k: int | None = None
    include_entity_neighbors: bool = True
    entity_neighbor_limit: int = 5
    entity_pivot_limit: int = 3
    entity_relationship_hops: int = 1
    relationship_neighbor_limit: int = 5

    def __post_init__(self) -> None:
        if self.route_k is not None and (
            not isinstance(self.route_k, int) or isinstance(self.route_k, bool) or self.route_k < 1
        ):
            raise ValueError(f"route_k must be a positive integer or None, got {self.route_k!r}.")
        if not isinstance(self.include_entity_neighbors, bool):
            raise TypeError("include_entity_neighbors must be a boolean.")
        for name, value in self.to_search_kwargs().items():
            if name in {"route_k", "include_entity_neighbors"}:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "Neo4jGraphRetrievalConfig":
        """Create a configuration from graph-search keyword arguments."""
        unexpected = set(values) - set(cls.keys())
        if unexpected:
            raise ValueError(f"Unsupported Neo4j graph retrieval settings: {sorted(unexpected)}.")
        return cls(**dict(values))

    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """Return keyword names accepted by :meth:`to_search_kwargs`."""
        return tuple(item.name for item in fields(cls))

    def to_search_kwargs(self) -> dict[str, int | bool]:
        """Return the configuration in :meth:`Neo4jGraphStore.search` keyword form."""
        return {name: value for name, value in asdict(self).items() if value is not None}


class Neo4jGraphStore(BaseVectorStore):
    """Graph-only vector store backed by Neo4j.

    ``add_documents`` stores one Document and Chunk set per collection. When a
    foundation model is supplied, ``SimpleKGPipeline`` extracts entities and
    relations from those chunks, and the graph writer links them back by chunk
    ID. Graph search uses the collection-specific vector index and expands
    context through ``__Entity__`` → ``FROM_CHUNK`` links.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    config : Neo4jConfig
        Connection parameters for the Neo4j instance.
    distance_metric : str, default="cosine"
        Distance metric used for the vector index.
    collection_name : str | None, default=None
        Existing collection to reuse; must start with the ``ai4rag`` prefix.
        When omitted, a new compliant name is generated.
    """

    _BATCH_SIZE = 512
    _DEFAULT_KG_EXTRACTION_CONCURRENCY = 4
    _VECTOR_INDEX_WAIT_SECONDS = 600

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: Neo4jConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
        foundation_model: BaseFoundationModel | None = None,
        kg_extraction_config: dict[str, Any] | None = None,
    ):
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(embedding_model)
        self._foundation_model = foundation_model
        self._kg_extraction_config = _validate_kg_extraction_config(kg_extraction_config)
        driver_kwargs: dict[str, Any] = {
            "auth": (config.username, config.password),
            "liveness_check_timeout": config.liveness_check_timeout,
            "max_connection_lifetime": config.max_connection_lifetime,
            "max_transaction_retry_time": config.max_transaction_retry_time,
        }
        if config.ca_cert:
            # The Neo4j driver rejects ``trusted_certificates`` with ``+s``
            # URIs because those schemes select its built-in CA trust mode.
            # Use the corresponding base scheme below, then explicitly retain
            # TLS and verify the server against the configured private CA.
            driver_kwargs["encrypted"] = True
            driver_kwargs["trusted_certificates"] = neo4j.TrustCustomCAs(
                materialize_ca_cert(config.ca_cert, prefix="ai4rag-neo4j-cert-")
            )
        driver_uri = _normalize_custom_ca_uri(config.uri) if config.ca_cert else config.uri
        self._driver = neo4j.GraphDatabase.driver(driver_uri, **driver_kwargs)
        self._driver.verify_connectivity()
        self._vector_index_lock = Lock()
        self._vector_index_online = False
        self._canonical_schema_ready = False
        self._canonical_schema_lock = Lock()
        self._kg_extraction_degraded = False

    @property
    def is_kg_extraction_degraded(self) -> bool:
        """Whether the latest KG extraction exceeded its skipped-chunk budget."""
        return self._kg_extraction_degraded

    def _ensure_canonical_schema(self) -> None:
        """Create lookup constraints used by bounded canonical and KG writes."""
        with self._canonical_schema_lock:
            if self._canonical_schema_ready:
                return
            started_at = perf_counter()
            with self._driver.session(database=self._config.database) as session:
                # Collections written by earlier ai4rag versions did not put
                # collection on Document nodes. Backfill before adding the
                # composite constraints, so re-indexing remains idempotent.
                session.run(
                    f"MATCH (d:`{self._collection_name}`:{_SCHEMA.DOCUMENT_LABEL}) "
                    f"WHERE d.{_SCHEMA.COLLECTION} IS NULL SET d.{_SCHEMA.COLLECTION} = $col",
                    col=self._collection_name,
                ).consume()
                session.run(
                    f"MATCH (c:`{self._collection_name}`:{_SCHEMA.CHUNK_LABEL}) "
                    f"WHERE c.{_SCHEMA.COLLECTION} IS NULL SET c.{_SCHEMA.COLLECTION} = $col",
                    col=self._collection_name,
                ).consume()
                session.run(
                    "CREATE CONSTRAINT ai4rag_chunk_collection_id IF NOT EXISTS "
                    "FOR (c:Chunk) REQUIRE (c.collection, c.id) IS UNIQUE"
                ).consume()
                session.run(
                    "CREATE CONSTRAINT ai4rag_document_collection_id IF NOT EXISTS "
                    "FOR (d:Document) REQUIRE (d.collection, d.id) IS UNIQUE"
                ).consume()
                session.run(
                    "CREATE CONSTRAINT ai4rag_entity_identity IF NOT EXISTS "
                    "FOR (e:__Entity__) REQUIRE e.ai4rag_entity_key IS UNIQUE"
                ).consume()
            self._canonical_schema_ready = True
            logger.info(
                "Neo4j canonical lookup constraints ready in %.2fs (collection=%s).",
                perf_counter() - started_at,
                self._collection_name,
            )

    def _ensure_kg_schema(self, *, wait_until_online: bool = False) -> None:
        """Create this collection's vector index, optionally waiting for it to become queryable.

        Graph retrieval can call this method from every inference worker.  Once
        this store has observed the index ONLINE, neither the DDL request nor
        the wait needs to be repeated; a lock also prevents concurrent workers
        from issuing duplicate first-use requests.
        """
        with self._vector_index_lock:
            if wait_until_online and self._vector_index_online:
                return

            started_at = perf_counter()
            index_name = _collection_vector_index_name(self._collection_name)
            with self._driver.session(database=self._config.database) as session:
                session.run(
                    f"CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS "
                    # Neo4j vector indexes support one node label. The collection
                    # label scopes this index; nodes retain their :Chunk label for
                    # graph traversal queries.
                    f"FOR (n:`{self._collection_name}`) ON (n.{_SCHEMA.EMBEDDING}) "
                    f"OPTIONS {{indexConfig: {{`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}}}",
                    dim=self._embedding_dimension,
                )
                if wait_until_online:
                    session.run(
                        "CALL db.awaitIndex($index_name, $timeout_seconds)",
                        index_name=index_name,
                        timeout_seconds=self._VECTOR_INDEX_WAIT_SECONDS,
                    ).consume()
                    self._vector_index_online = True
            logger.info(
                "Neo4j vector index %s for collection=%s completed in %.2fs.",
                "online wait" if wait_until_online else "creation request",
                self._collection_name,
                perf_counter() - started_at,
            )

    # Pylint counts loop and comprehension targets below as locals. They are
    # intentionally kept together to make the write batching transaction-safe.
    # pylint: disable-next=too-many-locals
    def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
        """Embed, deduplicate, and upsert chunks into Neo4j.

        Creates ``Chunk`` and ``Document`` nodes (with the collection label)
        and ``CONTAINS`` provenance links. Also ensures a collection-specific vector index exists and sets the
        ``collection`` property on each chunk so graph traversal can scope
        related nodes to this collection without requiring
        :meth:`build_knowledge_graph_from_documents`.

        When *model* is provided via kwargs, entity extraction is performed after
        indexing: chunks are sent to the LLM, and the resulting
        ``__Entity__`` nodes are written to Neo4j with ``FROM_CHUNK``
        relationships so that ``search(mode="graph")`` can expand context via the
        knowledge graph.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and stored.
        **kwargs : Any
            Optional overrides:

            - ``batch_size`` (int) — max chunks per write transaction
              (default :attr:`_BATCH_SIZE`).
            - ``model`` — foundation model used for entity extraction. When
              omitted and no model was supplied at construction, graph search
              returns ANN seed chunks without entity-based context expansion.
            - ``kg_max_concurrent`` (int) — maximum concurrent KG extraction
              requests when a foundation model is present (default 2).
        """
        if not documents:
            return

        self._ensure_canonical_schema()
        started_at = perf_counter()
        embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])
        logger.info(
            "Canonical embedding for %d chunks completed in %.2fs (collection=%s).",
            len(documents),
            perf_counter() - started_at,
            self._collection_name,
        )
        unique_pairs = list(iter_unique_chunks(documents, embeddings))

        doc_groups: dict[str, list[tuple[AI4RAGChunk, list[float]]]] = {}
        for doc, emb in unique_pairs:
            doc_id = doc.metadata.get("document_id", doc.chunk_id)
            doc_groups.setdefault(doc_id, []).append((doc, emb))

        for pairs in doc_groups.values():
            pairs.sort(key=lambda p: p[0].metadata.get("sequence_number", 0))

        batch_size = kwargs.get("batch_size", self._BATCH_SIZE)
        pending: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]] = []
        pending_count = 0

        started_at = perf_counter()
        for doc_id, sorted_pairs in doc_groups.items():
            pending.append((doc_id, sorted_pairs))
            pending_count += len(sorted_pairs)
            if pending_count >= batch_size:
                self._upsert_doc_groups(pending)
                pending = []
                pending_count = 0

        if pending:
            self._upsert_doc_groups(pending)
        logger.info(
            "Canonical graph write for %d chunks completed in %.2fs (collection=%s).",
            len(unique_pairs),
            perf_counter() - started_at,
            self._collection_name,
        )

        # Building HNSW from the completed chunk set is substantially faster
        # than maintaining it incrementally while each chunk is written.  Do
        # not wait here: KG extraction can proceed while Neo4j populates it;
        # search waits for ONLINE before issuing a vector query.
        self._ensure_kg_schema()

        model = kwargs.get("model", self._foundation_model)
        if model is not None:
            self._run_kg_pipeline(
                chunks=[chunk for chunk, _ in unique_pairs],
                model=model,
                max_concurrent=self._get_kg_max_concurrent(kwargs),
                on_error=kwargs.get("on_error", "IGNORE"),
                perform_entity_resolution=kwargs.get("perform_entity_resolution", True),
            )

    @classmethod
    def _get_kg_max_concurrent(cls, kwargs: Mapping[str, Any]) -> int:
        """Read and validate the caller's concurrent KG extraction request limit."""
        value = kwargs.get("kg_max_concurrent", cls._DEFAULT_KG_EXTRACTION_CONCURRENCY)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError("kg_max_concurrent must be a positive integer.")
        return value

    # The adapters, writer, and pipeline are deliberately retained for phase
    # metrics after asynchronous extraction completes. The nested chunk processor
    # tracks request, retry, and degradation state.
    # pylint: disable-next=too-many-locals,too-many-statements
    def _run_kg_pipeline(
        self,
        chunks: list[AI4RAGChunk],
        model: Any,
        max_concurrent: int = _DEFAULT_KG_EXTRACTION_CONCURRENCY,
        on_error: str = "IGNORE",
        perform_entity_resolution: bool = True,
    ) -> None:
        """Extract one canonical chunk per request and write the graphs separately."""
        chunks = [chunk for chunk in chunks if chunk.text.strip()]
        if not chunks:
            return
        self._kg_extraction_degraded = False

        logger.info(
            "Starting entity extraction for %d chunks (max_concurrent=%d, collection=%s).",
            len(chunks),
            max_concurrent,
            self._collection_name,
        )
        pipeline_started_at = perf_counter()

        kg_writer = _CanonicalKGWriter(self._driver, self._config.database, self._collection_name)
        recording_mode = self._kg_extraction_config["response_format"]
        response_format = (
            KG_BATCH_JSON_SCHEMA_RESPONSE_FORMAT if recording_mode == "json_schema" else KG_BATCH_RESPONSE_FORMAT
        )
        llm = _LLMAdapter(
            model,
            max_completion_tokens=self._kg_extraction_config["max_completion_tokens"],
            response_format=response_format,
            max_retries=self._kg_extraction_config["max_retries"],
            retry_base_seconds=self._kg_extraction_config["retry_base_seconds"],
            sdk_max_retries=self._kg_extraction_config["sdk_max_retries"],
        )

        async def _run_all() -> None:  # pylint: disable=too-many-locals,too-many-statements
            sem = asyncio.Semaphore(max_concurrent)
            extracted_graphs: list[tuple[str, Neo4jGraph]] = []
            extracted_graphs_lock = asyncio.Lock()
            skipped_chunk_ids: list[str] = []
            skipped_chunks_lock = asyncio.Lock()
            consecutive_rate_limits = 0
            rate_limit_lock = asyncio.Lock()
            rate_limit_breaker = asyncio.Event()
            breaker_threshold = self._kg_extraction_config["rate_limit_breaker_threshold"]

            async def _record_rate_limit() -> None:
                """Count an exhausted rate-limited request and trip the breaker at the threshold."""
                nonlocal consecutive_rate_limits
                async with rate_limit_lock:
                    consecutive_rate_limits += 1
                    tripped = consecutive_rate_limits >= breaker_threshold
                    if tripped:
                        rate_limit_breaker.set()
                if tripped:
                    self._kg_extraction_degraded = True
                    raise _KGExtractionRateLimitedError(
                        f"The foundation-model endpoint rejected {consecutive_rate_limits} consecutive KG "
                        f"extraction requests for collection={self._collection_name}. Check the model subscription "
                        "token quota and rate limits, or lower kg_max_concurrent."
                    )

            async def _record_success() -> None:
                nonlocal consecutive_rate_limits
                async with rate_limit_lock:
                    consecutive_rate_limits = 0

            def _abort_error() -> _KGExtractionRateLimitedError:
                return _KGExtractionRateLimitedError(
                    f"KG extraction aborted after the endpoint kept rejecting requests "
                    f"(collection={self._collection_name})."
                )

            async def _request(chunk: AI4RAGChunk):
                """Issue one request without holding a slot during retry backoff."""
                if rate_limit_breaker.is_set():
                    raise _abort_error()
                async with sem:
                    if rate_limit_breaker.is_set():
                        raise _abort_error()
                    return await llm.ainvoke_with_completion_limit(
                        self._build_kg_batch_prompt([chunk]), self._kg_extraction_config["max_completion_tokens"]
                    )

            async def _process_chunk(chunk: AI4RAGChunk) -> None:
                """Extract one chunk, recording it on success or skipping it on failure."""
                try:
                    response = await _request(chunk)
                    await _record_success()
                    graphs, missing = self._parse_kg_batch_response(response.content, [chunk], on_error)
                    async with extracted_graphs_lock:
                        extracted_graphs.extend((item.chunk_id, graph) for item, graph in graphs)
                    if missing:
                        raise _KGResponseFormatError("KG LLM response omitted or invalidated the requested chunk.")
                    return
                except _KGResponseFormatError as exc:
                    if on_error == "RAISE":
                        raise
                    logger.error(
                        "Skipping KG extraction for collection=%s, chunk_id=%s after an invalid LLM response: %s",
                        self._collection_name,
                        chunk.chunk_id,
                        exc,
                    )
                except _KGExtractionRateLimitedError:
                    raise
                except Exception as exc:  # pylint: disable=broad-except
                    if is_rate_limit_llm_error(exc):
                        await _record_rate_limit()
                    if on_error == "RAISE":
                        raise
                    logger.error(
                        "Skipping KG extraction for collection=%s, chunk_id=%s after SDK retry failure: %s",
                        self._collection_name,
                        chunk.chunk_id,
                        exc,
                    )
                async with skipped_chunks_lock:
                    skipped_chunk_ids.append(chunk.chunk_id)

            tasks = [asyncio.ensure_future(_process_chunk(chunk)) for chunk in chunks]
            try:
                await asyncio.gather(*tasks)
            finally:
                # The breaker fails the first poisoned request; stop the rest instead of
                # letting them keep hammering an endpoint that already rejected us.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                # Persist what succeeded before surfacing a breaker failure, so a retry
                # does not re-pay for chunks that were already extracted.
                kg_writer.write_graphs(extracted_graphs)
            skipped_count = len(skipped_chunk_ids)
            if skipped_count >= len(chunks):
                self._kg_extraction_degraded = True
                message = (
                    f"KG extraction produced no entities for collection={self._collection_name}: "
                    f"all {len(chunks)} chunks were skipped. Check the foundation-model endpoint and its "
                    "rate limits before retrying."
                )
                if self._kg_extraction_config["fail_on_empty_kg"] or (
                    self._kg_extraction_config["partial_failure_policy"] == "raise"
                ):
                    raise _KGExtractionDegradedError(message)
                logger.error(message)
            elif skipped_count > self._kg_extraction_config["max_skipped_chunks"]:
                message = "KG extraction is degraded for collection=%s: skipped %d of %d chunks (allowed=%d)."
                self._kg_extraction_degraded = True
                if self._kg_extraction_config["partial_failure_policy"] == "raise":
                    raise _KGExtractionDegradedError(
                        message
                        % (
                            self._collection_name,
                            skipped_count,
                            len(chunks),
                            self._kg_extraction_config["max_skipped_chunks"],
                        )
                    )
                logger.error(
                    message,
                    self._collection_name,
                    skipped_count,
                    len(chunks),
                    self._kg_extraction_config["max_skipped_chunks"],
                )

        try:
            with _suppress_neo4j_graphrag_info_logs():
                asyncio.run(_run_all())
        except _KGExtractionDegradedError:
            raise
        except RuntimeError as exc:
            raise RuntimeError(
                "_run_kg_pipeline cannot be called from within a running event loop. "
                "Install 'nest_asyncio' and call nest_asyncio.apply() beforehand."
            ) from exc

        if not perform_entity_resolution:
            logger.info(
                "perform_entity_resolution=False is redundant: entity identity is resolved during KG ingestion "
                "(collection=%s).",
                self._collection_name,
            )

        llm_calls, llm_elapsed_seconds, llm_format_repairs = llm.metrics
        llm_retries, llm_failed_calls = llm.retry_metrics
        logger.info(
            "KG phases for collection=%s: LLM=%d calls/%.2fs/%d response repairs/%d retries/%d failed requests, "
            "graph writes=%.2fs.",
            self._collection_name,
            llm_calls,
            llm_elapsed_seconds,
            llm_format_repairs,
            llm_retries,
            llm_failed_calls,
            kg_writer.write_elapsed_seconds,
        )
        logger.info(
            "KG extraction and graph persistence completed in %.2fs (collection=%s).",
            perf_counter() - pipeline_started_at,
            self._collection_name,
        )

        logger.info(
            "Finished entity extraction for %d chunks (collection=%s).",
            len(chunks),
            self._collection_name,
        )

    def _build_kg_batch_prompt(self, chunks: list[AI4RAGChunk]) -> str:
        """Build the strict, provenance-preserving extraction request."""
        if self._kg_extraction_config["mode"] == "constrained":
            schema = (
                "Node labels: Person, Organization, Place, Concept, Event, Product, Technology. "
                "Relationship types: RELATED_TO, PART_OF, LOCATED_IN, BELONGS_TO, CREATED_BY, MENTIONS."
            )
            max_entities = KG_MAX_ENTITIES_PER_CHUNK
            max_relationships = KG_MAX_RELATIONSHIPS_PER_CHUNK
        else:
            schema = "Choose concise node labels and relationship types supported by the text."
            max_entities = self._kg_extraction_config["max_entities_per_chunk"]
            max_relationships = self._kg_extraction_config["max_relationships_per_chunk"]
        payload = [{"chunk_id": chunk.chunk_id, "text": chunk.text} for chunk in chunks]
        contract = (
            "Return only this JSON object, with exactly this structure:\n"
            + '{"chunks":[{"chunk_id":"input chunk ID","nodes":[{"id":"local ID","label":"type",'
            + '"properties":{"name":"entity name"}}],"relationships":[{"start_node_id":"local ID",'
            + '"end_node_id":"local ID","type":"RELATIONSHIP"}]}]}\n'
            + "Every input chunk_id must appear exactly once, including chunks with no entities (empty arrays).\n"
            + "Examples (format only; do not copy their facts or IDs):\n"
            + 'Input: [{"chunk_id":"example-relationship","text":"Ada created Atlas."}]\n'
            + 'Output: {"chunks":[{"chunk_id":"example-relationship","nodes":['
            + '{"id":"n1","label":"Person","properties":{"name":"Ada"}},'
            + '{"id":"n2","label":"Product","properties":{"name":"Atlas"}}],'
            + '"relationships":[{"start_node_id":"n2","end_node_id":"n1","type":"CREATED_BY"}]}]}\n'
            + 'Input: [{"chunk_id":"example-empty","text":"The and if or but."}]\n'
            + 'Output: {"chunks":[{"chunk_id":"example-empty","nodes":[],"relationships":[]}]}\n'
        )
        return (
            "Extract a knowledge graph independently for every input chunk. "
            "Never create relationships between different chunk_id values. "
            f"{schema}\n"
            f"Hard limits per chunk: at most {max_entities} entities and at most "
            f"{max_relationships} relationships. Never exceed these limits; when the chunk contains more "
            "candidates, keep only the most important entities and relationships and drop the rest. "
            "Node properties must contain only the entity name, written as a short name of no more than "
            f"{KG_MAX_DESCRIPTION_WORDS} words. Relationship properties must be empty. Only when the selected "
            "schema requires an entity or relationship description, generate a concise, factual description "
            f"of no more than {KG_MAX_DESCRIPTION_WORDS} words; keep it to a short phrase and never paraphrase "
            "the whole chunk.\n" + contract + f"Input chunks:\n{json.dumps(payload)}"
        )

    @staticmethod
    def _parse_kg_batch_response(
        content: str, chunks: list[AI4RAGChunk], on_error: str
    ) -> tuple[list[tuple[AI4RAGChunk, Neo4jGraph]], list[AI4RAGChunk]]:
        """Validate a batched result and return successful graphs plus chunks requiring retry."""
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        try:
            result = json.loads(content)
            entries = result["chunks"]
            if not isinstance(entries, list):
                raise ValueError("chunks must be a list")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise _KGResponseFormatError("KG batch LLM response is not valid JSON with a chunks array.") from exc

        graphs: list[tuple[AI4RAGChunk, Neo4jGraph]] = []
        seen_ids: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            chunk_id = entry.get("chunk_id")
            if not isinstance(chunk_id, str) or chunk_id not in chunk_by_id or chunk_id in seen_ids:
                continue
            try:
                graph = Neo4jGraph.model_validate(
                    {"nodes": entry.get("nodes", []), "relationships": entry.get("relationships", [])}
                )
            except ValueError:
                if on_error == "RAISE":
                    raise
                logger.error("KG batch LLM response has improper graph format for one chunk.")
                continue
            seen_ids.add(chunk_id)
            graphs.append((chunk_by_id[chunk_id], graph))
        missing = [chunk for chunk in chunks if chunk.chunk_id not in seen_ids]
        if missing:
            logger.warning("KG batch LLM response omitted or invalidated %d requested chunks.", len(missing))
        return graphs, missing

    def _upsert_doc_groups(self, doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]]) -> None:
        with self._driver.session(database=self._config.database) as session:
            session.execute_write(self._upsert_batch_tx, doc_groups, self._collection_name)

    @staticmethod
    def _upsert_batch_tx(
        tx: neo4j.Transaction,
        doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]],
        collection_name: str,
    ) -> None:
        document_rows: list[dict[str, str]] = []
        chunk_rows: list[dict[str, Any]] = []
        for doc_id, sorted_pairs in doc_groups:
            source = sorted_pairs[0][0].metadata.get("source", "")
            document_rows.append(
                {
                    "id": doc_id,
                    "source": source,
                    "metadata": json.dumps({"source": source}),
                    "collection": collection_name,
                }
            )
            for chunk, embedding in sorted_pairs:
                clean_metadata = {**chunk.metadata, "source": chunk.metadata.get("source") or doc_id}
                chunk_rows.append(
                    {
                        "id": chunk.chunk_id,
                        "text": chunk.text,
                        "embedding": embedding,
                        "document_id": doc_id,
                        "sequence_number": chunk.metadata.get("sequence_number", 0),
                        "metadata": json.dumps(clean_metadata),
                        "collection": collection_name,
                    }
                )
        tx.run(
            f"UNWIND $rows AS row MERGE (d:{collection_name}:{_SCHEMA.DOCUMENT_LABEL} "
            f"{{{_SCHEMA.COLLECTION}: row.collection, {_SCHEMA.ID}: row.id}}) "
            f"SET d.{_SCHEMA.COLLECTION} = row.collection, d.{_SCHEMA.SOURCE} = row.source, "
            f"d.{_SCHEMA.METADATA} = row.metadata",
            rows=document_rows,
        ).consume()
        tx.run(
            f"UNWIND $rows AS row MERGE (c:{collection_name}:{_SCHEMA.CHUNK_LABEL} "
            f"{{{_SCHEMA.COLLECTION}: row.collection, {_SCHEMA.ID}: row.id}}) "
            f"SET c.{_SCHEMA.TEXT} = row.text, c.{_SCHEMA.EMBEDDING} = row.embedding, "
            f"c.{_SCHEMA.DOCUMENT_ID} = row.document_id, c.{_SCHEMA.SEQUENCE_NUMBER} = row.sequence_number, "
            f"c.{_SCHEMA.METADATA} = row.metadata, c.{_SCHEMA.COLLECTION} = row.collection "
            f"WITH c, row MATCH (d:{collection_name}:{_SCHEMA.DOCUMENT_LABEL} "
            f"{{{_SCHEMA.COLLECTION}: row.collection, {_SCHEMA.ID}: row.document_id}}) "
            f"MERGE (d)-[:{_SCHEMA.CONTAINS_RELATIONSHIP}]->(c)",
            rows=chunk_rows,
        ).consume()

    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        search_mode: str = "graph",
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
        k : int
            Number of results to return.
        include_scores : bool, default=False
            Whether to include similarity scores in the return value.
        search_mode : str, default="graph"
            ``"graph"`` — ANN seed retrieval + entity-based chunk discovery via
            :class:`neo4j_graphrag.retrievers.VectorCypherRetriever` against the
            collection-specific vector index (requires extracted entities for
            graph-neighbor retrieval). Each result remains a single stored chunk.
        **kwargs : Any
            Graph-mode parameters:

            - ``include_entity_neighbors`` (bool, default True) — expand via ``__Entity__``.
            - ``entity_neighbor_limit`` (int, default 5) — max entity-linked neighbors per seed.
            - ``entity_pivot_limit`` (int, default 3) — max entity pivots for relationship traversal.
            - ``entity_relationship_hops`` (int, default 1) — relationship hops from each pivot.
            - ``relationship_neighbor_limit`` (int, default 5) — max relationship-expanded chunks per seed.
        """
        _validate_neo4j_search_params(
            search_mode,
            ranker_strategy,
            ranker_k,
            ranker_alpha,
            store_class=type(self),
            **kwargs,
        )
        self._ensure_kg_schema(wait_until_online=True)

        return self._search_graph(query, k, include_scores, **kwargs)

    def _search_graph(
        self,
        query: str,
        k: int,
        include_scores: bool,
        **kwargs,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        """Run direct-vector and local-graph retrieval concurrently.

        Both routes search the same query, so the query is embedded once and the
        resulting vector is shared instead of being recomputed per route.
        """
        route_k = kwargs.get("route_k", max(k * 2, 10))
        query_vector = self.embedding_model.embed_query(query)
        with ThreadPoolExecutor(max_workers=2) as executor:
            vector_future = executor.submit(self._search_vector_route, query_vector, route_k)
            local_future = executor.submit(self._search_local_route, query_vector, route_k, **kwargs)
            routes = {
                "vector": vector_future.result(),
                "local": local_future.result(),
            }

        pairs = self._fuse_graph_routes(routes, k)
        if include_scores:
            return pairs
        return [chunk for chunk, _ in pairs]

    def _search_local_route(
        self,
        query_vector: list[float],
        k: int,
        **kwargs,
    ) -> list[tuple[AI4RAGChunk, float]]:
        """Return individual entity/path-linked chunks using VectorCypherRetriever.

        ``query_vector`` is the already-computed embedding of the search query,
        shared with the direct-vector route.
        """
        include_entity_neighbors = kwargs.get("include_entity_neighbors", True)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)
        entity_pivot_limit = kwargs.get("entity_pivot_limit", 3)
        entity_relationship_hops = kwargs.get("entity_relationship_hops", 1)
        relationship_neighbor_limit = kwargs.get("relationship_neighbor_limit", 5)

        def _fmt(record) -> RetrieverResultItem:
            raw_meta = record.get("metadata")
            if isinstance(raw_meta, str) and raw_meta:
                try:
                    chunk_meta = json.loads(raw_meta)
                except Exception:
                    chunk_meta = {}
            else:
                chunk_meta = raw_meta or {}
            if "document_id" not in chunk_meta and record.get("document_id"):
                chunk_meta["document_id"] = record.get("document_id")
            if chunk_meta.get("source") is None:
                chunk_meta["source"] = ""
            if chunk_meta.get("document_id") is None:
                chunk_meta["document_id"] = record.get("document_id") or ""
            return RetrieverResultItem(
                content=record.get("text") or "",
                metadata={
                    "score": float(record.get("score", 0.0)),
                    "_meta": chunk_meta,
                },
            )

        retriever = VectorCypherRetriever(
            driver=self._driver,
            index_name=_collection_vector_index_name(self._collection_name),
            retrieval_query=_build_graph_retrieval_query(
                include_entity_neighbors,
                entity_neighbor_limit,
                entity_pivot_limit,
                entity_relationship_hops,
                relationship_neighbor_limit,
            ),
            embedder=_EmbedderAdapter(self.embedding_model),
            result_formatter=_fmt,
            neo4j_database=self._config.database,
        )
        try:
            result = retriever.search(query_vector=query_vector, top_k=k, query_params={"col": self._collection_name})
        except Exception as exc:
            if not _is_neo4j_memory_error(exc):
                raise
            logger.error(
                "Neo4j ran out of heap while expanding the knowledge graph for collection=%r "
                "(entity_relationship_hops=%d, entity_pivot_limit=%d, relationship_neighbor_limit=%d). "
                "Reduce graph_retrieval_config entity_relationship_hops, or raise the Neo4j heap. "
                "Failing the pattern now instead of retrying a deterministic failure.",
                self._collection_name,
                entity_relationship_hops,
                entity_pivot_limit,
                relationship_neighbor_limit,
            )
            raise Neo4jMemoryError(
                f"Neo4j heap exhausted during graph expansion for collection {self._collection_name!r}. "
                "Lower graph_retrieval_config.entity_relationship_hops or increase the Neo4j heap."
            ) from exc

        return [
            (
                AI4RAGChunk(text=item.content, metadata=item.metadata.get("_meta", {})),
                float(item.metadata.get("score", 0.0)),
            )
            for item in result.items
            if item.content
        ]

    def _search_vector_route(self, query_vector: list[float], k: int) -> list[tuple[AI4RAGChunk, float]]:
        """Return raw semantic chunk evidence without graph expansion."""
        return self._search_vector_index(
            query_vector=query_vector,
            k=k,
            index_name=_collection_vector_index_name(self._collection_name),
            text_property=_SCHEMA.TEXT,
            route="vector",
        )

    def _search_vector_index(
        self,
        query_vector: list[float],
        k: int,
        index_name: str,
        text_property: str,
        route: str,
    ) -> list[tuple[AI4RAGChunk, float]]:
        """Run a Neo4j GraphRAG vector retriever and normalize its evidence."""

        def _format(record) -> RetrieverResultItem:
            # VectorRetriever returns the indexed node under ``node``.  Retain
            # a fallback to top-level properties for compatibility with older
            # GraphRAG versions and mocked retriever records.
            node = record.get("node") or record
            raw_metadata = node.get("metadata")
            if isinstance(raw_metadata, str) and raw_metadata:
                try:
                    metadata = json.loads(raw_metadata)
                except json.JSONDecodeError:
                    metadata = {}
            else:
                metadata = dict(raw_metadata or {})
            document_id = node.get("document_id")
            if document_id is not None:
                metadata.setdefault("document_id", document_id)
            return RetrieverResultItem(
                content=node.get(text_property) or "",
                metadata={
                    "score": float(record.get("score", 0.0)),
                    "_meta": {**metadata, "route": route},
                },
            )

        retriever = VectorRetriever(
            driver=self._driver,
            index_name=index_name,
            embedder=_EmbedderAdapter(self.embedding_model),
            result_formatter=_format,
            neo4j_database=self._config.database,
        )
        result = retriever.search(query_vector=query_vector, top_k=k)
        return [
            (
                AI4RAGChunk(text=item.content, metadata=item.metadata.get("_meta", {})),
                float(item.metadata.get("score", 0.0)),
            )
            for item in result.items
            if item.content
        ]

    @staticmethod
    def _fuse_graph_routes(
        routes: dict[str, list[tuple[AI4RAGChunk, float]]], k: int
    ) -> list[tuple[AI4RAGChunk, float]]:
        """Fuse route rankings with reciprocal-rank fusion and deduplicate text."""
        fused: dict[str, tuple[AI4RAGChunk, float]] = {}
        for route, pairs in routes.items():
            for rank, (chunk, _score) in enumerate(pairs, start=1):
                key = chunk.text.strip()
                if not key:
                    continue
                contribution = 1.0 / (60 + rank)
                if key in fused:
                    previous, score = fused[key]
                    previous.metadata.setdefault("routes", []).append(route)
                    fused[key] = (previous, score + contribution)
                else:
                    chunk.metadata["routes"] = [route]
                    fused[key] = (chunk, contribution)
        return sorted(fused.values(), key=lambda pair: pair[1], reverse=True)[:k]

    def build_knowledge_graph_from_documents(
        self,
        documents: list[DoclingDocument],
        model: BaseFoundationModel,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        on_error: str = "IGNORE",
        perform_entity_resolution: bool = True,
    ) -> None:
        """Chunk Docling documents and build one collection graph.

        Uses ``FixedSizeSplitter`` to create canonical ``AI4RAGChunk`` objects,
        then delegates storage and entity extraction to :meth:`add_documents`.
        The KG writer links extracted entities to those stored chunks by ID;
        it does not persist a second set of pipeline document or chunk nodes.

        Parameters
        ----------
        documents : list[DoclingDocument]
            Parsed documents to process.
        model : BaseFoundationModel
            Foundation model used for entity and relation extraction.
        chunk_size : int, default=2000
            Target chunk size in characters (passed to ``FixedSizeSplitter``).
        chunk_overlap : int, default=200
            Overlap in characters between consecutive chunks.
        on_error : str, default="IGNORE"
            Error handling strategy passed to ``SimpleKGPipeline``
            (``"IGNORE"`` or ``"RAISE"``).
        perform_entity_resolution : bool, default=True
            Whether to merge duplicate entity nodes after extraction.
        """
        splitter = FixedSizeSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        async def _split_documents() -> list[AI4RAGChunk]:
            chunks: list[AI4RAGChunk] = []
            for doc in documents:
                text = doc.export_to_markdown()
                if not text.strip():
                    continue
                name = getattr(doc, "name", None)
                document_id = name if isinstance(name, str) and name else hashlib.sha256(text.encode()).hexdigest()
                result = await splitter.run(text)
                chunks.extend(
                    AI4RAGChunk(
                        text=part.text,
                        metadata={"document_id": document_id, "sequence_number": part.index, "source": document_id},
                    )
                    for part in result.chunks
                )
            return chunks

        try:
            chunks = asyncio.run(_split_documents())
        except RuntimeError as exc:
            raise RuntimeError(
                "build_knowledge_graph_from_documents cannot be called from within a running "
                "event loop.  Install 'nest_asyncio' and call nest_asyncio.apply() before "
                "invoking this method in a Jupyter notebook or other async context."
            ) from exc

        if not chunks:
            logger.info("No text extracted from documents; skipping KG build.")
            return

        self.add_documents(
            chunks,
            model=model,
            on_error=on_error,
            perform_entity_resolution=perform_entity_resolution,
        )

        logger.info(
            "Knowledge graph built from %d documents (collection=%s).",
            len(documents),
            self._collection_name,
        )

    def clean_collection(self) -> None:
        """Delete all nodes, KG entities, and the vector index for this collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(f"DROP INDEX `{_collection_vector_index_name(self._collection_name)}` IF EXISTS")
            session.run(
                f"MATCH (:{_SCHEMA.ENTITY_LABEL})-[r]->(:{_SCHEMA.ENTITY_LABEL}) "
                f"WHERE $col IN COALESCE(r.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"AND size(r.{_SCHEMA.KG_COLLECTIONS}) = 1 DELETE r",
                col=self._collection_name,
            )
            session.run(
                f"MATCH (:{_SCHEMA.ENTITY_LABEL})-[r]->(:{_SCHEMA.ENTITY_LABEL}) "
                f"WHERE $col IN COALESCE(r.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"SET r.{_SCHEMA.KG_COLLECTIONS} = "
                f"[owner IN r.{_SCHEMA.KG_COLLECTIONS} WHERE owner <> $col]",
                col=self._collection_name,
            )
            # Legacy entities have no ownership tag. Only delete those linked
            # exclusively to chunks from this collection, while those links
            # still exist to prove their provenance.
            session.run(
                f"MATCH (e:{_SCHEMA.ENTITY_LABEL})-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(c:{_SCHEMA.CHUNK_LABEL}) "
                f"WHERE (c.{_SCHEMA.COLLECTION} = $col OR $col IN labels(c)) "
                f"AND size(COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, [])) = 0 "
                f"AND NOT EXISTS {{ MATCH (e)-[:{_SCHEMA.FROM_CHUNK_RELATIONSHIP}]->(other:{_SCHEMA.CHUNK_LABEL}) "
                f"WHERE COALESCE(other.{_SCHEMA.COLLECTION}, '') <> $col AND NOT $col IN labels(other) }} "
                "DETACH DELETE e",
                col=self._collection_name,
            )
            # Entity nodes can be shared by several collections. Remove this
            # collection's ownership first and delete only entities no longer
            # owned by any AI4RAG collection.
            session.run(
                f"MATCH (e:{_SCHEMA.ENTITY_LABEL}) WHERE $col IN COALESCE(e.{_SCHEMA.KG_COLLECTIONS}, []) "
                f"SET e.{_SCHEMA.KG_COLLECTIONS} = "
                f"[owner IN e.{_SCHEMA.KG_COLLECTIONS} WHERE owner <> $col] "
                f"WITH e WHERE size(e.{_SCHEMA.KG_COLLECTIONS}) = 0 "
                "DETACH DELETE e",
                col=self._collection_name,
            )
            session.run(f"MATCH (n:{self._collection_name}) DETACH DELETE n")
            session.run(
                f"MATCH (c:{_SCHEMA.CHUNK_LABEL} {{{_SCHEMA.COLLECTION}: $col}}) DETACH DELETE c",
                col=self._collection_name,
            )
        logger.info("Collection %s cleaned.", self._collection_name)

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()
