# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import math
from collections.abc import Callable

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from ai4rag.utils.constants import TokenEstimation

__all__ = ["CharApproxTokenizer"]


class CharApproxTokenizer(BaseTokenizer):
    """
    Lightweight tokenizer that approximates token count as ``ceil(len(text) / 4)``.

    It provides a consistent, model-agnostic token estimate suitable for
    chunk-size budgeting.
    """

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        """Approximate token count: 4 characters per token, rounded up."""
        return math.ceil(len(text) / TokenEstimation.CHARS_PER_TOKEN)

    def get_max_tokens(self) -> int:
        """Return the configured maximum token budget."""
        return self.max_tokens

    def get_tokenizer(self) -> Callable[[str], int]:
        """Return a token-counting callable for use by downstream splitters."""
        return self.count_tokens
