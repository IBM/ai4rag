# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import TypedDict, Optional

from httpx import Timeout


class LSEmbeddingParams(TypedDict, total=False):
    """LLamaStack parameters to be used to create embeddings."""

    embedding_dimension: int
    context_length: int
    timeout: Optional[float | Timeout]
    model_type: Optional[str]
    provider_id: Optional[str]
    provider_resource_id: Optional[str]
