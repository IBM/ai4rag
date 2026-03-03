# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Optional

from annotated_types import MinLen
from pydantic import (
    BaseModel,
    ConfigDict,
)

config = ConfigDict(extra="forbid")


class AI4RAGFoundationModel(BaseModel):
    """Attributes to be included in the generation.foundation_models payload."""

    model_config = config

    model_id: Annotated[str, MinLen(1)]


class AI4RAGEmbeddingModel(BaseModel):
    """Attributes to be included in the generation.embedding_models payload."""

    model_config = config

    model_id: Annotated[str, MinLen(1)]


class AI4RAGConstraints(BaseModel):
    """Attributes to be included in constraints payload."""

    model_config = config

    embedding_models: Optional[Annotated[list[AI4RAGFoundationModel], MinLen(1)]] = None
    foundation_models: Optional[Annotated[list[AI4RAGEmbeddingModel], MinLen(1)]] = None
