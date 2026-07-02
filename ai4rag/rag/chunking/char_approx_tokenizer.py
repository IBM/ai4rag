# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import math

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

__all__ = ["CharApproxTokenizer"]

_CHARS_PER_TOKEN = 4


class CharApproxTokenizer(BaseTokenizer):
    """
    Lightweight tokenizer that approximates token count as ``ceil(len(text) / 4)``.

    Providing a consistent, model-agnostic token estimate suitable for
    chunk-size budgeting.
    """

    max_tokens: int

    def count_tokens(self, text: str) -> int:
        """Approximate token count: 4 characters per token, rounded up."""
        return math.ceil(len(text) / _CHARS_PER_TOKEN)

    def get_max_tokens(self) -> int:
        """Return the configured maximum token budget."""
        return self.max_tokens

    def get_tokenizer(self) -> None:
        """No underlying tokenizer object — returns ``None``."""
        return None
