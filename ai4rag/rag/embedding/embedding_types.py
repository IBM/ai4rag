# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import TypedDict, Optional, Mapping

from llama_stack_client import Omit
from httpx import Timeout


class LLamaStackEmbeddingParams(TypedDict, total=False):
    """LLamaStack parameters to be used to create embeddings."""

    dimensions: int | Omit

    encoding_format: str | Omit

    user: str | Omit

    extra_headers: Optional[Mapping[str, str]]

    extra_query: Optional[Mapping[str, object]]

    extra_body: Optional[object]

    timeout: Optional[float | Timeout]
