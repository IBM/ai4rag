# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Neo4j graph-retrieval configuration."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields

__all__ = ["Neo4jGraphRetrievalConfig"]


@dataclass(frozen=True, kw_only=True)
class Neo4jGraphRetrievalConfig:
    """Controls Neo4j graph expansion during retrieval."""

    route_k: int | None = None
    include_entity_neighbors: bool = True
    entity_neighbor_limit: int = 5
    entity_pivot_limit: int = 1
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
