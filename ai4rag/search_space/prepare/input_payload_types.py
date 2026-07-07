# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Optional

from annotated_types import Ge, Le, MinLen
from pydantic import BaseModel, ConfigDict, field_validator

from ai4rag.utils.constants import ChunkingConstraints

CONFIG = ConfigDict(extra="forbid")


class AI4RAGFoundationModel(BaseModel):
    """Attributes to be included in the generation.foundation_models payload."""

    model_config = CONFIG

    model_id: Annotated[str, MinLen(1)]


class AI4RAGEmbeddingModel(BaseModel):
    """Attributes to be included in the generation.embedding_models payload."""

    model_config = CONFIG

    model_id: Annotated[str, MinLen(1)]


class AI4RAGConstraints(BaseModel):
    """Attributes to be included in constraints payload."""

    model_config = CONFIG

    embedding_models: Optional[Annotated[list[AI4RAGEmbeddingModel], MinLen(1)]] = None
    foundation_models: Optional[Annotated[list[AI4RAGFoundationModel], MinLen(1)]] = None
    chunking_methods: Optional[Annotated[list[Annotated[str, MinLen(1)]], MinLen(1)]] = None
    chunk_sizes: Optional[
        Annotated[
            list[Annotated[int, Ge(ChunkingConstraints.MIN_CHUNK_SIZE), Le(ChunkingConstraints.MAX_CHUNK_SIZE)]],
            MinLen(1),
        ]
    ] = None
    chunk_overlaps: Optional[
        Annotated[
            list[Annotated[int, Ge(ChunkingConstraints.MIN_CHUNK_OVERLAP), Le(ChunkingConstraints.MAX_CHUNK_OVERLAP)]],
            MinLen(1),
        ]
    ] = None

    @field_validator("chunking_methods", mode="after")
    @classmethod
    def _validate_chunking_methods(cls, v):
        if v is not None:
            unsupported = [m for m in v if m not in ChunkingConstraints.METHODS]
            if unsupported:
                raise ValueError(
                    f"Unsupported chunking methods: {unsupported!r}. "
                    f"Supported methods: {ChunkingConstraints.METHODS!r}."
                )
        return v
