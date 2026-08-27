# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
# This implementation is modelled after the neo4j-graphrag library
# (https://github.com/neo4j/neo4j-graphrag-python) but uses the raw neo4j
# driver instead, to avoid the APOC plugin dependency that some neo4j-graphrag
# internals require. All Cypher here uses only built-in Neo4j 5.x procedures.
import heapq
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from json_repair import repair_json

import neo4j
from docling_core.types.doc import DoclingDocument

from ai4rag import logger
from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.chunking.docling_chunker import DoclingChunker
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.openai_model import OpenAIFoundationModel
from ai4rag.rag.vector_store.base_vector_store import BaseVectorStore
from ai4rag.rag.vector_store.config import Neo4jConfig
from ai4rag.rag.vector_store.reranker import WeightedInMemoryAggregator
from ai4rag.rag.vector_store.utils import iter_unique_chunks, resolve_embedding_dimension

__all__ = ["Neo4jGraphStore"]


class Neo4jGraphStore(BaseVectorStore):
    """Vector store backed by Neo4j with optional graph traversal search.

    Stores chunks as graph nodes and supports two search modes:

    * ``"vector"`` — pure dense ANN retrieval via a Neo4j vector index.
    * ``"graph"`` — dense seed retrieval followed by chunk-to-chunk and
      chunk-to-entity graph traversal to widen the returned context window.

    Parameters
    ----------
    embedding_model : BaseEmbeddingModel
        Model used to embed documents and queries.
    config : Neo4jConfig
        Connection parameters for the Neo4j instance.
    distance_metric : str, default="cosine"
        Distance metric used for the vector index. Currently only ``"cosine"``
        is supported by Neo4j's native vector index.
    collection_name : str | None, default=None
        Existing collection to reuse; must start with the ``ai4rag`` prefix.
        When omitted, a new compliant name is generated.
    """

    _BATCH_SIZE = 512

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        config: Neo4jConfig,
        distance_metric: str = "cosine",
        collection_name: str | None = None,
    ):
        super().__init__(embedding_model, config, distance_metric, collection_name)
        self._embedding_dimension = resolve_embedding_dimension(embedding_model)
        self._driver = neo4j.GraphDatabase.driver(
            config.uri,
            auth=(config.username, config.password),
        )
        self._driver.verify_connectivity()
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create the vector index for this collection if absent."""
        with self._driver.session(database=self._config.database) as session:
            session.run(
                f"CREATE VECTOR INDEX `{self._collection_name}__vector` IF NOT EXISTS "
                f"FOR (n:{self._collection_name}) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: $dim, `vector.similarity_function`: 'cosine'}}}}",
                dim=self._embedding_dimension,
            )
        logger.info("Neo4j schema ready: %s (dim=%d)", self._collection_name, self._embedding_dimension)

    def add_documents(self, documents: list[AI4RAGChunk], **kwargs) -> None:
        """Embed, deduplicate, and upsert chunks into Neo4j.

        Creates ``Chunk`` and ``Document`` nodes, ``CONTAINS`` provenance links,
        and ``NEXT_CHUNK`` sequential links between consecutive chunks of the
        same document.

        Parameters
        ----------
        documents : list[AI4RAGChunk]
            Chunks to be embedded and stored.
        **kwargs : Any
            Optional overrides. ``batch_size`` (int) sets the maximum number of
            chunks per write transaction (default :attr:`_BATCH_SIZE`).
        """
        if not documents:
            return

        embeddings = self.embedding_model.embed_documents([doc.text for doc in documents])
        unique_pairs = list(iter_unique_chunks(documents, embeddings))

        # Group chunks by document_id; sort each group by sequence_number for NEXT_CHUNK links.
        doc_groups: dict[str, list[tuple[AI4RAGChunk, list[float]]]] = {}
        for doc, emb in unique_pairs:
            doc_id = doc.metadata.get("document_id", doc.chunk_id)
            doc_groups.setdefault(doc_id, []).append((doc, emb))

        for doc_id in doc_groups:
            doc_groups[doc_id].sort(key=lambda p: p[0].metadata.get("sequence_number", 0))

        batch_size = kwargs.get("batch_size", self._BATCH_SIZE)
        pending: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]] = []
        pending_count = 0

        for doc_id, sorted_pairs in doc_groups.items():
            pending.append((doc_id, sorted_pairs))
            pending_count += len(sorted_pairs)
            if pending_count >= batch_size:
                self._upsert_doc_groups(pending)
                pending = []
                pending_count = 0

        if pending:
            self._upsert_doc_groups(pending)

    def _upsert_doc_groups(
        self, doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]]
    ) -> None:
        with self._driver.session(database=self._config.database) as session:
            session.execute_write(self._upsert_batch_tx, doc_groups, self._collection_name)

    @staticmethod
    def _upsert_batch_tx(
        tx: neo4j.Transaction,
        doc_groups: list[tuple[str, list[tuple[AI4RAGChunk, list[float]]]]],
        collection_name: str,
    ) -> None:
        for doc_id, sorted_pairs in doc_groups:
            source = sorted_pairs[0][0].metadata.get("source", "")
            tx.run(
                f"MERGE (d:{collection_name}:Document {{id: $doc_id}}) "
                f"SET d.source = $source, d.metadata = $doc_metadata",
                doc_id=doc_id,
                source=source,
                doc_metadata=json.dumps({"source": source}),
            )

            for chunk, embedding in sorted_pairs:
                tx.run(
                    f"MERGE (c:{collection_name}:Chunk {{id: $id}}) "
                    f"SET c.text = $text, c.embedding = $embedding, "
                    f"c.document_id = $document_id, c.sequence_number = $sequence_number, "
                    f"c.metadata = $metadata",
                    id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=embedding,
                    document_id=doc_id,
                    sequence_number=chunk.metadata.get("sequence_number", 0),
                    metadata=json.dumps(chunk.metadata),
                )
                tx.run(
                    f"MATCH (d:{collection_name}:Document {{id: $doc_id}}), "
                    f"(c:{collection_name}:Chunk {{id: $chunk_id}}) "
                    f"MERGE (d)-[:CONTAINS]->(c)",
                    doc_id=doc_id,
                    chunk_id=chunk.chunk_id,
                )

            for i in range(len(sorted_pairs) - 1):
                chunk_a = sorted_pairs[i][0]
                chunk_b = sorted_pairs[i + 1][0]
                tx.run(
                    f"MATCH (a:{collection_name}:Chunk {{id: $id_a}}), "
                    f"(b:{collection_name}:Chunk {{id: $id_b}}) "
                    f"MERGE (a)-[:NEXT_CHUNK]->(b)",
                    id_a=chunk_a.chunk_id,
                    id_b=chunk_b.chunk_id,
                )

    def search(
        self,
        query: str,
        k: int,
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
        k : int
            Number of results to return.
        include_scores : bool, default=False
            Whether to include similarity scores.
        search_mode : str, default="vector"
            ``"vector"`` or ``"graph"``.
        ranker_strategy : str | None, default=None
            Fusion strategy for hybrid/graph: ``"rrf"``, ``"weighted"``, or
            ``"normalized"``.
        ranker_k : int | None, default=None
            RRF smoothing constant.
        ranker_alpha : float | None, default=None
            Weighted blend factor (``0`` = keyword only, ``1`` = vector only).
        **kwargs : Any
            Graph-mode parameters: ``graph_hops`` (int, default 1),
            ``include_entity_neighbors`` (bool, default True),
            ``entity_neighbor_limit`` (int, default 5).

        Returns
        -------
        list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]
            Matched chunks, optionally paired with their scores.
        """
        _validate_neo4j_search_params(search_mode, ranker_strategy, ranker_k, ranker_alpha, **kwargs)

        if search_mode == "graph":
            return self._search_graph(query, k, include_scores, ranker_strategy, ranker_k, ranker_alpha, **kwargs)
        return self._search_vector(query, k, include_scores)

    def _search_vector(
        self, query: str, k: int, include_scores: bool
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        embedding = self.embedding_model.embed_query(query)
        with self._driver.session(database=self._config.database) as session:
            result = session.run(
                f"CALL db.index.vector.queryNodes($index, $k, $embedding) "
                f"YIELD node, score "
                f"RETURN node.id AS id, node.text AS text, node.metadata AS metadata, score",
                index=f"{self._collection_name}__vector",
                k=k,
                embedding=embedding,
            )
            rows = result.data()

        results = _rows_to_chunks_with_scores(rows)
        if include_scores:
            return results
        return [chunk for chunk, _ in results]

    def _search_graph(
        self,
        query: str,
        k: int,
        include_scores: bool,
        ranker_strategy: str | None,
        ranker_k: int | None,
        ranker_alpha: float | None,
        **kwargs,
    ) -> list[AI4RAGChunk] | list[tuple[AI4RAGChunk, float]]:
        graph_hops = kwargs.get("graph_hops", 1)
        include_entity_neighbors = kwargs.get("include_entity_neighbors", True)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)

        seed_results = self._search_vector(query, k, include_scores=True)

        chunk_map: dict[str, AI4RAGChunk] = {}
        seed_scores: dict[str, float] = {}
        neighbor_scores: dict[str, float] = {}

        for seed_chunk, seed_score in seed_results:
            chunk_map[seed_chunk.chunk_id] = seed_chunk
            seed_scores[seed_chunk.chunk_id] = seed_score

        with self._driver.session(database=self._config.database) as session:
            for seed_chunk, seed_score in seed_results:
                # Sequential neighbors (forward and backward)
                rows = session.run(
                    f"MATCH (seed:{self._collection_name}:Chunk {{id: $seed_id}}) "
                    f"OPTIONAL MATCH (seed)-[:NEXT_CHUNK*1..{graph_hops}]->(fwd:{self._collection_name}:Chunk) "
                    f"OPTIONAL MATCH (bwd:{self._collection_name}:Chunk)-[:NEXT_CHUNK*1..{graph_hops}]->(seed) "
                    f"RETURN fwd, bwd",
                    seed_id=seed_chunk.chunk_id,
                ).data()
                for row in rows:
                    for node_key in ("fwd", "bwd"):
                        node = row.get(node_key)
                        if node is not None:
                            neighbor = _node_to_chunk(node)
                            if neighbor.chunk_id not in seed_scores:
                                chunk_map[neighbor.chunk_id] = neighbor
                                neighbor_scores[neighbor.chunk_id] = max(
                                    neighbor_scores.get(neighbor.chunk_id, 0.0), seed_score
                                )

                # Entity-linked neighbors
                if include_entity_neighbors:
                    rows = session.run(
                        f"MATCH (seed:{self._collection_name}:Chunk {{id: $seed_id}})"
                        f"-[:MENTIONS]->(e:{self._collection_name}:Entity)"
                        f"<-[:MENTIONS]-(neighbor:{self._collection_name}:Chunk) "
                        f"WHERE neighbor.id <> $seed_id "
                        f"RETURN DISTINCT neighbor "
                        f"LIMIT $limit",
                        seed_id=seed_chunk.chunk_id,
                        limit=entity_neighbor_limit,
                    ).data()
                    for row in rows:
                        node = row.get("neighbor")
                        if node is not None:
                            neighbor = _node_to_chunk(node)
                            if neighbor.chunk_id not in seed_scores:
                                chunk_map[neighbor.chunk_id] = neighbor
                                neighbor_scores[neighbor.chunk_id] = max(
                                    neighbor_scores.get(neighbor.chunk_id, 0.0), seed_score
                                )

        # Fuse seed and neighbor scores, then take top-k.
        if ranker_strategy and neighbor_scores:
            _, combined_scores = _fuse_results(
                [(chunk_map[cid], score) for cid, score in seed_scores.items()],
                [(chunk_map[cid], score) for cid, score in neighbor_scores.items()],
                ranker_strategy,
                ranker_k,
                ranker_alpha,
            )
        else:
            combined_scores = {**seed_scores, **neighbor_scores}

        top_k = heapq.nlargest(k, combined_scores.items(), key=lambda x: x[1])

        if include_scores:
            return [(chunk_map[cid], score) for cid, score in top_k if cid in chunk_map]
        return [chunk_map[cid] for cid, _ in top_k if cid in chunk_map]

    def build_knowledge_graph(
        self,
        model: OpenAIFoundationModel,
        entities: list[str] | None = None,
        relations: list[str] | None = None,
        chunk_batch_size: int = 16,
        max_workers: int = 4,
        max_tokens: int = 2048,
    ) -> None:
        """Extract entities and relations from indexed chunks and persist as a knowledge graph.

        Reads all ``Chunk`` nodes in the collection (paginated), sends them in
        batches to an LLM for entity/relation extraction, and writes the resulting
        ``Entity`` nodes and ``MENTIONS``/``RELATED_TO`` relationships back to
        Neo4j. All ``MERGE`` statements are idempotent; re-running on the same
        corpus only updates existing nodes.

        Extraction is parallelised across batches using a thread pool (one LLM
        call per thread), with output tokens capped at *max_tokens* per call to
        limit cost and latency.

        Parameters
        ----------
        model : OpenAIFoundationModel
            Foundation model used for chat completions.
        entities : list[str] | None, default=None
            Optional list of entity types to hint the extraction prompt.
        relations : list[str] | None, default=None
            Optional list of relation types to hint the extraction prompt.
        chunk_batch_size : int, default=16
            Number of chunks sent to the LLM per call.
        max_workers : int, default=4
            Number of parallel LLM threads.
        max_tokens : int, default=512
            Maximum output tokens per LLM call (limits cost and latency).
        """
        all_chunks = self._read_all_chunks()
        if not all_chunks:
            logger.info("No chunks found in collection %s; skipping KG build.", self._collection_name)
            return

        batches = [all_chunks[i : i + chunk_batch_size] for i in range(0, len(all_chunks), chunk_batch_size)]
        entity_hint = f"\nFocus on entity types: {', '.join(entities)}." if entities else ""
        relation_hint = f"\nFocus on relation types: {', '.join(relations)}." if relations else ""
        system_prompt = _KG_BATCH_SYSTEM_PROMPT + entity_hint + relation_hint

        def extract_batch(batch: list[dict]) -> tuple[list[dict], list[dict]]:
            texts = "\n\n".join(f"[{c['id']}] {c['text']}" for c in batch)
            choices = model.chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": texts},
                ],
                max_completion_tokens=max_tokens,
            )
            raw = choices[0].message.content or ""
            return _parse_kg_extraction(raw)

        def write_batch_result(chunk_entities: list[dict], relationships: list[dict]) -> None:
            # chunk_entities carry a "chunk_id" property set by the LLM.
            with self._driver.session(database=self._config.database) as session:
                for ent in chunk_entities:
                    chunk_id = ent.get("chunk_id", "")
                    # Reuse the per-document writer; pass single-entity list.
                    session.execute_write(
                        Neo4jGraphStore._write_kg_result_tx,
                        chunk_id,
                        [ent],
                        [],
                        self._collection_name,
                    )
                if relationships:
                    # Write relationships and cross-entity MENTIONS in a second pass
                    # once all entity nodes exist.
                    session.execute_write(
                        Neo4jGraphStore._write_kg_result_tx,
                        "",
                        [],
                        relationships,
                        self._collection_name,
                    )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                try:
                    batch_entities, batch_rels = future.result()
                    if batch_entities or batch_rels:
                        write_batch_result(batch_entities, batch_rels)
                except Exception as exc:
                    logger.warning("KG extraction batch failed: %s", exc)

        logger.info("Knowledge graph built for collection %s.", self._collection_name)

    def _read_all_chunks(self) -> list[dict]:
        """Return all Chunk nodes in this collection, paginated."""
        all_chunks: list[dict] = []
        skip = 0
        page_size = 1000
        with self._driver.session(database=self._config.database) as session:
            while True:
                page = session.run(
                    f"MATCH (c:{self._collection_name}:Chunk) "
                    f"RETURN c.id AS id, c.text AS text "
                    f"SKIP $skip LIMIT $limit",
                    skip=skip,
                    limit=page_size,
                ).data()
                all_chunks.extend(page)
                if len(page) < page_size:
                    break
                skip += page_size
        return all_chunks

    @staticmethod
    def _write_triples_tx(tx: neo4j.Transaction, triples: list[dict], collection_name: str) -> None:
        for triple in triples:
            entity = triple.get("entity", "")
            entity_type = triple.get("entity_type", "UNKNOWN")
            relation = triple.get("relation", "RELATED_TO")
            target = triple.get("target", "")
            target_type = triple.get("target_type", "UNKNOWN")
            chunk_id = triple.get("chunk_id", "")

            if not entity or not target:
                continue

            tx.run(
                f"MERGE (e1:{collection_name}:Entity {{name: $name, entity_type: $etype}}) "
                f"MERGE (e2:{collection_name}:Entity {{name: $target, entity_type: $ttype}}) "
                f"MERGE (e1)-[:RELATED_TO {{relation_type: $relation}}]->(e2)",
                name=entity,
                etype=entity_type,
                target=target,
                ttype=target_type,
                relation=relation,
            )
            if chunk_id:
                tx.run(
                    f"MATCH (c:{collection_name}:Chunk {{id: $chunk_id}}) "
                    f"MATCH (e1:{collection_name}:Entity {{name: $name, entity_type: $etype}}) "
                    f"MATCH (e2:{collection_name}:Entity {{name: $target, entity_type: $ttype}}) "
                    f"MERGE (c)-[:MENTIONS]->(e1) "
                    f"MERGE (c)-[:MENTIONS]->(e2)",
                    chunk_id=chunk_id,
                    name=entity,
                    etype=entity_type,
                    target=target,
                    ttype=target_type,
                )

    def build_knowledge_graph_from_documents(
        self,
        documents: list[DoclingDocument],
        model: OpenAIFoundationModel,
        chunker: DoclingChunker | None = None,
        max_workers: int = 4,
        max_tokens: int = 4096,
    ) -> None:
        """Chunk documents, index them, and build a knowledge graph in one step.

        Combines :meth:`add_documents` and knowledge-graph extraction into a
        single pipeline.  Entity extraction runs on the in-memory chunks
        immediately after they are stored, so no DB read-back is required.

        Extraction uses a structured JSON prompt (entities + relationships
        separate) modelled after the KGEnricher design used in ai4rag pipelines.
        Each chunk is processed by one LLM call; calls run in parallel via a
        ``ThreadPoolExecutor``.

        Parameters
        ----------
        documents : list[DoclingDocument]
            Parsed documents to chunk and index.
        model : OpenAIFoundationModel
            Foundation model used for entity extraction.
        chunker : DoclingChunker | None, default=None
            Chunker to use.  When ``None``, a :class:`DoclingChunker` with
            default settings is created automatically.
        max_workers : int, default=4
            Number of parallel LLM threads.
        max_tokens : int, default=1024
            Maximum output tokens per LLM call.
        """
        if chunker is None:
            chunker = DoclingChunker()

        chunks = chunker.split_documents(documents)
        if not chunks:
            logger.info("No chunks produced from documents; skipping KG build.")
            return

        self.add_documents(chunks)

        def extract_one(chunk: AI4RAGChunk) -> tuple[str, list[dict], list[dict]]:
            """Return (chunk_id, entities, relationships) or empty lists on failure."""
            try:
                choices = model.chat(
                    [
                        {"role": "system", "content": _KG_EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": chunk.text},
                    ],
                    max_completion_tokens=max_tokens,
                )
                raw = choices[0].message.content or ""
                entities, relationships = _parse_kg_extraction(raw)
            except Exception as exc:
                logger.warning("KG extraction failed for chunk %s: %s", chunk.chunk_id[:12], exc)
                entities, relationships = [], []
            return chunk.chunk_id, entities, relationships

        def write_kg_result(chunk_id: str, entities: list[dict], relationships: list[dict]) -> None:
            with self._driver.session(database=self._config.database) as session:
                session.execute_write(
                    Neo4jGraphStore._write_kg_result_tx,
                    chunk_id,
                    entities,
                    relationships,
                    self._collection_name,
                )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_one, c): c for c in chunks}
            for future in as_completed(futures):
                try:
                    chunk_id, entities, relationships = future.result()
                    if entities or relationships:
                        write_kg_result(chunk_id, entities, relationships)
                except Exception as exc:
                    logger.warning("KG write failed: %s", exc)

        logger.info(
            "Knowledge graph built from %d documents (%d chunks) for collection %s.",
            len(documents),
            len(chunks),
            self._collection_name,
        )

    @staticmethod
    def _write_kg_result_tx(
        tx: neo4j.Transaction,
        chunk_id: str,
        entities: list[dict],
        relationships: list[dict],
        collection_name: str,
    ) -> None:
        # MERGE each entity node (name + type as composite key; update description).
        for ent in entities:
            name = ent.get("name", "")
            if not name:
                continue
            tx.run(
                f"MERGE (e:{collection_name}:Entity {{name: $name, entity_type: $etype}}) "
                f"SET e.description = $desc",
                name=name,
                etype=ent.get("type", "Other"),
                desc=ent.get("description", ""),
            )
            if chunk_id:
                tx.run(
                    f"MATCH (c:{collection_name}:Chunk {{id: $cid}}) "
                    f"MATCH (e:{collection_name}:Entity {{name: $name, entity_type: $etype}}) "
                    f"MERGE (c)-[:MENTIONS]->(e)",
                    cid=chunk_id,
                    name=name,
                    etype=ent.get("type", "Other"),
                )

        # Build name→type lookup from extracted entities for relationship wiring.
        entity_type_map = {e["name"]: e.get("type", "Other") for e in entities if e.get("name")}

        for rel in relationships:
            src = rel.get("source", "")
            tgt = rel.get("target", "")
            if not src or not tgt:
                continue
            src_type = entity_type_map.get(src, "Other")
            tgt_type = entity_type_map.get(tgt, "Other")
            tx.run(
                f"MERGE (e1:{collection_name}:Entity {{name: $src, entity_type: $stype}}) "
                f"MERGE (e2:{collection_name}:Entity {{name: $tgt, entity_type: $ttype}}) "
                f"MERGE (e1)-[:RELATED_TO {{keywords: $kw, description: $desc}}]->(e2)",
                src=src,
                stype=src_type,
                tgt=tgt,
                ttype=tgt_type,
                kw=rel.get("keywords", ""),
                desc=rel.get("description", ""),
            )

    def resolve_entities(self) -> int:
        """Merge duplicate Entity nodes that share the same name (case-insensitive).

        After extraction, the same real-world entity can appear under slightly
        different surface forms (e.g. ``"openshift"`` vs ``"OpenShift"``).  This
        method collapses all Entity nodes whose lower-cased name matches into a
        single canonical node (keeping the capitalisation of the first node
        encountered), redirecting every ``MENTIONS`` and ``RELATED_TO``
        relationship to the survivor.

        Returns
        -------
        int
            Number of duplicate nodes removed.
        """
        with self._driver.session(database=self._config.database) as session:
            # Collect all entities grouped by lower-cased name.
            rows = session.run(
                f"MATCH (e:{self._collection_name}:Entity) "
                f"RETURN e.name AS name, elementId(e) AS eid "
                f"ORDER BY e.name"
            ).data()

        # Group by lower-cased name; first entry in each group is the canonical node.
        groups: dict[str, list[str]] = {}
        for row in rows:
            key = (row["name"] or "").lower()
            groups.setdefault(key, []).append(row["eid"])

        removed = 0
        with self._driver.session(database=self._config.database) as session:
            for eids in groups.values():
                if len(eids) < 2:
                    continue
                canonical_eid = eids[0]
                for dup_eid in eids[1:]:
                    # Re-point MENTIONS relationships from the duplicate to the canonical.
                    session.run(
                        "MATCH (c)-[:MENTIONS]->(dup) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (c)-[:MENTIONS]->(canon)",
                        dup=dup_eid, canon=canonical_eid,
                    )
                    # Re-point outgoing RELATED_TO from the duplicate.
                    session.run(
                        "MATCH (dup)-[:RELATED_TO]->(target) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (canon)-[:RELATED_TO]->(target)",
                        dup=dup_eid, canon=canonical_eid,
                    )
                    # Re-point incoming RELATED_TO to the duplicate.
                    session.run(
                        "MATCH (src)-[:RELATED_TO]->(dup) WHERE elementId(dup) = $dup "
                        "MATCH (canon) WHERE elementId(canon) = $canon "
                        "MERGE (src)-[:RELATED_TO]->(canon)",
                        dup=dup_eid, canon=canonical_eid,
                    )
                    # Delete the duplicate node and all its (now redirected) relationships.
                    session.run(
                        "MATCH (dup) WHERE elementId(dup) = $dup DETACH DELETE dup",
                        dup=dup_eid,
                    )
                    removed += 1

        logger.info("Entity resolver removed %d duplicate nodes from collection %s.", removed, self._collection_name)
        return removed

    def clean_collection(self) -> None:
        """Drop vector and fulltext indexes and delete all nodes in the collection."""
        with self._driver.session(database=self._config.database) as session:
            session.run(f"DROP INDEX `{self._collection_name}__vector` IF EXISTS")
            session.run(f"DROP INDEX `{self._collection_name}__fulltext` IF EXISTS")
            session.run(f"MATCH (n:{self._collection_name}) DETACH DELETE n")
        logger.info("Collection %s cleaned.", self._collection_name)

    def close(self) -> None:
        """Close the Neo4j driver."""
        self._driver.close()


# ---------------------------------------------------------------------------
# KG extraction prompt and parser (used by build_knowledge_graph_from_documents)
# ---------------------------------------------------------------------------

# Aligned with neo4j-graphrag's ERExtractionTemplate design:
# - No entity/relationship count limit (cap was the largest single source of recall loss)
# - Node-ID-based relationship wiring (avoids name-mismatch drops)
# - json_repair fallback before json.loads (recovers truncated/malformed responses)
# Entity-type taxonomy from LightRAG; description field retained for richer graph.
_KG_EXTRACTION_SYSTEM_PROMPT = """\
You are a top-tier algorithm designed for extracting information in structured \
formats to build a knowledge graph.

Extract ALL entities (nodes) and ALL relationships from the text. \
Do not apply any count limit — extract every entity and relationship you find.

---Entity types---
Person, Organization, Location, Concept, Method, Artifact, Event, Data, Content, Other.

---Rules---
- Retain established capitalisation (e.g. "vLLM", "OpenShift").
- Assign a unique string ID (starting from "0") to each node and reuse that ID \
in relationships to avoid name-mismatch errors.
- description: ONE sentence, max 20 words, third person.
- Do not return anything other than the JSON object below.
- Do not wrap the JSON in backticks or markdown fences.

---Output format---
{"nodes": [{"id": "0", "label": "Person", "properties": {"name": "Alice", "description": "..."}}],
 "relationships": [{"type": "WORKS_AT", "start_node_id": "0", "end_node_id": "1", "properties": {"description": "..."}}]}
"""

# System prompt for build_knowledge_graph's multi-chunk batch mode.
# Uses the same node-ID wiring; chunk attribution is via per-node "chunk_id" property.
_KG_BATCH_SYSTEM_PROMPT = """\
You are a top-tier algorithm designed for extracting information in structured \
formats to build a knowledge graph.

You will receive several text chunks, each prefixed with [chunk_id]. \
Extract ALL entities and ALL relationships from every chunk. \
Do not apply any count limit.

---Entity types---
Person, Organization, Location, Concept, Method, Artifact, Event, Data, Content, Other.

---Rules---
- Retain established capitalisation (e.g. "vLLM", "OpenShift").
- Assign a unique string ID (starting from "0") to each node and reuse that ID \
in relationships.
- Each node must carry a "chunk_id" property set to the ID of the chunk it was \
extracted from (the value inside the brackets).
- description: ONE sentence, max 20 words, third person.
- Do not return anything other than the JSON object below.
- Do not wrap the JSON in backticks or markdown fences.

---Output format---
{"nodes": [{"id": "0", "label": "Person", "properties": {"name": "Alice", "chunk_id": "c1", "description": "..."}}],
 "relationships": [{"type": "WORKS_AT", "start_node_id": "0", "end_node_id": "1", "properties": {"description": "..."}}]}
"""


def _parse_kg_extraction(raw: str) -> tuple[list[dict], list[dict]]:
    """Parse LLM JSON (node-ID format) into (entities, relationships).

    Uses ``json_repair`` as a fallback before ``json.loads`` so that truncated
    or slightly malformed responses are recovered rather than silently dropped.
    Relationships are wired via the integer node IDs assigned in the same
    response — this prevents name-mismatch drops that occur when ``source`` /
    ``target`` strings do not exactly match the entity name.
    """
    try:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        repaired = repair_json(cleaned, skip_json_loads=False, return_objects=False)
        data = json.loads(repaired) if isinstance(repaired, str) else repaired
        if not isinstance(data, dict):
            return [], []

        # Build id → entity dict from nodes.
        id_to_entity: dict[str, dict] = {}
        entities: list[dict] = []
        for node in data.get("nodes", []):
            node_id = str(node.get("id", ""))
            props = node.get("properties", {})
            name = props.get("name", "")
            if not name:
                continue
            entity = {
                "name": name,
                "type": node.get("label", "Other"),
                "description": props.get("description", ""),
                "chunk_id": props.get("chunk_id", ""),
            }
            id_to_entity[node_id] = entity
            entities.append(entity)

        # Wire relationships via node IDs — avoids name-string mismatches.
        relationships: list[dict] = []
        for rel in data.get("relationships", []):
            src = id_to_entity.get(str(rel.get("start_node_id", "")))
            tgt = id_to_entity.get(str(rel.get("end_node_id", "")))
            if not src or not tgt:
                continue
            props = rel.get("properties", {})
            relationships.append({
                "source": src["name"],
                "target": tgt["name"],
                "keywords": rel.get("type", ""),
                "description": props.get("description", ""),
            })

        return entities, relationships
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _node_to_chunk(node: dict) -> AI4RAGChunk:
    """Convert a raw Neo4j node dict to an :class:`AI4RAGChunk`."""
    raw_metadata = node.get("metadata")
    if isinstance(raw_metadata, str):
        metadata = json.loads(raw_metadata) if raw_metadata else {}
    else:
        metadata = raw_metadata or {}
    return AI4RAGChunk(text=node["text"], metadata=metadata)


def _rows_to_chunks_with_scores(rows: list[dict]) -> list[tuple[AI4RAGChunk, float]]:
    return [(_node_to_chunk(row), float(row["score"])) for row in rows]


def _fuse_results(
    vector_results: list[tuple[AI4RAGChunk, float]],
    keyword_results: list[tuple[AI4RAGChunk, float]],
    ranker_strategy: str | None,
    ranker_k: int | None,
    ranker_alpha: float | None,
) -> tuple[dict[str, AI4RAGChunk], dict[str, float]]:
    vector_scores: dict[str, float] = {}
    keyword_scores: dict[str, float] = {}
    chunk_map: dict[str, AI4RAGChunk] = {}

    for chunk, score in vector_results:
        vector_scores[chunk.chunk_id] = score
        chunk_map[chunk.chunk_id] = chunk

    for chunk, score in keyword_results:
        keyword_scores[chunk.chunk_id] = score
        chunk_map.setdefault(chunk.chunk_id, chunk)

    reranker_params: dict[str, Any] = {}
    if ranker_strategy == "rrf" and ranker_k is not None and ranker_k > 0:
        reranker_params["k"] = ranker_k
    if ranker_strategy == "weighted" and ranker_alpha is not None and ranker_alpha != 1:
        reranker_params["alpha"] = ranker_alpha

    combined_scores = WeightedInMemoryAggregator.combine_search_results(
        vector_scores, keyword_scores, ranker_strategy or "rrf", reranker_params
    )
    return chunk_map, combined_scores


def _validate_neo4j_search_params(
    search_mode: str,
    ranker_strategy: str | None,
    ranker_k: int | None,
    ranker_alpha: float | None,
    **kwargs: Any,
) -> None:
    """Validate search parameters for :class:`Neo4jGraphStore`.

    Accepts ``"vector"`` and ``"graph"`` modes only. For ``"graph"`` mode,
    validates ranker parameter mutual-exclusion rules and graph-specific
    keyword arguments.

    Raises
    ------
    ValueError
        On any invalid parameter combination.
    """
    if search_mode == "vector":
        has_strategy = ranker_strategy is not None and ranker_strategy != ""
        has_k = ranker_k is not None and ranker_k > 0
        has_alpha = ranker_alpha is not None and ranker_alpha != 1
        if has_strategy or has_k or has_alpha:
            raise ValueError("ranker parameters are only valid when search_mode='graph'.")
        return

    if search_mode == "graph":
        has_strategy = ranker_strategy is not None and ranker_strategy != ""
        has_k = ranker_k is not None and ranker_k > 0
        has_alpha = ranker_alpha is not None and ranker_alpha != 1

        if has_strategy and ranker_strategy not in ("rrf", "weighted", "normalized"):
            raise ValueError(
                f"Invalid ranker_strategy='{ranker_strategy}'. Must be one of ('rrf', 'weighted', 'normalized')."
            )
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

        graph_hops = kwargs.get("graph_hops", 1)
        entity_neighbor_limit = kwargs.get("entity_neighbor_limit", 5)

        if not isinstance(graph_hops, int) or graph_hops < 1:
            raise ValueError(f"graph_hops must be a positive integer, got {graph_hops!r}.")
        if not isinstance(entity_neighbor_limit, int) or entity_neighbor_limit < 0:
            raise ValueError(f"entity_neighbor_limit must be a non-negative integer, got {entity_neighbor_limit!r}.")
        return

    raise ValueError(f"Invalid search_mode '{search_mode}'. Must be one of ('vector', 'graph').")
