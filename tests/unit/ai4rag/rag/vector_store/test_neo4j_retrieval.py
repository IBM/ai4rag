# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import pytest

from ai4rag.rag.vector_store.neo4j import Neo4jGraphRetrievalConfig


class TestNeo4jGraphRetrievalConfig:
    """Tests for Neo4j graph-search configuration."""

    def test_defaults_serialize_to_search_kwargs(self):
        config = Neo4jGraphRetrievalConfig()

        assert config.to_search_kwargs() == {
            "include_entity_neighbors": True,
            "entity_neighbor_limit": 5,
            "entity_pivot_limit": 3,
            "entity_relationship_hops": 1,
            "relationship_neighbor_limit": 5,
        }
        assert "route_k" in Neo4jGraphRetrievalConfig.keys()

    def test_from_mapping_accepts_partial_overrides(self):
        config = Neo4jGraphRetrievalConfig.from_mapping({"entity_pivot_limit": 3, "route_k": 20})

        assert config.entity_pivot_limit == 3
        assert config.to_search_kwargs()["route_k"] == 20
        assert config.relationship_neighbor_limit == 5

    def test_from_mapping_rejects_unknown_setting(self):
        with pytest.raises(ValueError, match="Unsupported Neo4j graph retrieval settings"):
            Neo4jGraphRetrievalConfig.from_mapping({"unknown": 1})

    def test_rejects_non_positive_route_k(self):
        with pytest.raises(ValueError, match="route_k"):
            Neo4jGraphRetrievalConfig(route_k=0)
