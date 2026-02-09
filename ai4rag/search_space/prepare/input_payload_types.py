# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
from typing import Annotated, Optional

from annotated_types import Ge, Gt, Le, MinLen
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from ai4rag.utils.constants import (
    ChatGenerationConstants,
)

config = ConfigDict(extra="forbid")


class AI4RAGFoundationModelParams(BaseModel):
    """Attributes to be included in the generation.foundation_models payload."""
    model_config = config

    max_completion_tokens: Annotated[int, Gt(0)] = ChatGenerationConstants.MAX_COMPLETION_TOKENS
    temperature: Annotated[float, Ge(0), Le(1)] = ChatGenerationConstants.TEMPERATURE


class AI4RAGFoundationModel(BaseModel):
    """Attributes to be included in the generation.foundation_models payload."""
    model_config = config

    model_id: Annotated[str, MinLen(1)]
    parameters: Optional[AI4RAGFoundationModelParams] = Field(default_factory=AI4RAGFoundationModelParams)


class AI4RAGConstraints:
    """Attributes to be included in constraints payload."""
    model_config = config

    embedding_models: Optional[Annotated[list[Annotated[str, MinLen(1)]], MinLen(1)]] = None
    foundation_models: Optional[Annotated[list[Annotated[str, MinLen(1)]], MinLen(1)]] = None
