# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated

from annotated_types import Ge, Le, MinLen
from pydantic import BaseModel, ConfigDict, field_validator

from ai4rag.rag.template.registry import get_template_spec
from ai4rag.utils.constants import ChunkingConstraints

CONFIG = ConfigDict(extra="forbid", hide_input_in_errors=True)


class AI4RAGFoundationModel(BaseModel):
    """Attributes to be included in the generation.foundation_models payload."""

    model_config = CONFIG

    model_id: Annotated[str, MinLen(1)]


class AI4RAGEmbeddingModel(BaseModel):
    """Attributes to be included in the generation.embedding_models payload."""

    model_config = CONFIG

    model_id: Annotated[str, MinLen(1)]


class SearchSpaceConstraints(BaseModel):
    """Template and chunking constraints shared by search-space preparation paths."""

    model_config = CONFIG

    template_ids: Annotated[list[str], MinLen(1)] | None = None
    chunking_methods: Annotated[list[Annotated[str, MinLen(1)]], MinLen(1)] | None = None
    chunk_sizes: (
        Annotated[
            list[Annotated[int, Ge(ChunkingConstraints.MIN_CHUNK_SIZE), Le(ChunkingConstraints.MAX_CHUNK_SIZE)]],
            MinLen(1),
        ]
        | None
    ) = None
    chunk_overlaps: (
        Annotated[
            list[Annotated[int, Ge(ChunkingConstraints.MIN_CHUNK_OVERLAP), Le(ChunkingConstraints.MAX_CHUNK_OVERLAP)]],
            MinLen(1),
        ]
        | None
    ) = None

    @field_validator("template_ids", mode="after")
    @classmethod
    def _validate_template_ids(cls, values: list[str] | None) -> list[str] | None:
        if values is not None:
            for template_id in values:
                try:
                    get_template_spec(template_id)
                except ValueError:
                    raise ValueError("Unsupported template_id. Choose a registered RAG template.") from None
        return values

    @field_validator("chunking_methods", mode="after")
    @classmethod
    def _validate_chunking_methods(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(method not in ChunkingConstraints.METHODS for method in values):
            raise ValueError(f"Unsupported chunking methods. Supported methods: {ChunkingConstraints.METHODS!r}.")
        return values


class AI4RAGConstraints(SearchSpaceConstraints):
    """Search-space constraints and model identifiers for MaaS discovery."""

    embedding_models: Annotated[list[AI4RAGEmbeddingModel], MinLen(1)] | None = None
    foundation_models: Annotated[list[AI4RAGFoundationModel], MinLen(1)] | None = None
