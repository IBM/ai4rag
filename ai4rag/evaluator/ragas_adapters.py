# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
"""Adapters exposing ai4rag foundation and embedding models to RAGAS.

RAGAS drives evaluation through its own ``BaseRagasLLM`` / ``BaseRagasEmbeddings``
interfaces.  These thin wrappers delegate to the ai4rag model abstractions
(:class:`BaseFoundationModel.chat` and :class:`BaseEmbeddingModel`), so RAGAS
reuses whatever endpoint the rest of the pipeline is already configured with
instead of opening its own OpenAI/LangChain client.
"""

import asyncio
from typing import Any

from langchain_core.outputs import Generation, LLMResult
from langchain_core.prompt_values import PromptValue
from ragas.embeddings.base import BaseRagasEmbeddings
from ragas.llms.base import BaseRagasLLM

from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

# RAGAS metric prompts emit structured JSON (statement lists, verdicts).  The
# ai4rag model's default ``max_completion_tokens`` can be small enough to
# truncate that JSON, which RAGAS then fails to parse -> NaN scores.  Cap the
# completion generously so evaluation prompts are not cut off.
DEFAULT_MAX_COMPLETION_TOKENS = 1024


class AI4RAGRagasLLM(BaseRagasLLM):
    """RAGAS LLM wrapper delegating to :meth:`BaseFoundationModel.chat`.

    Parameters
    ----------
    foundation_model : BaseFoundationModel
        The ai4rag foundation model that backs every RAGAS completion.
    max_completion_tokens : int, default=``DEFAULT_MAX_COMPLETION_TOKENS``
        Upper bound on tokens per completion. Kept generous so the structured
        JSON that RAGAS metrics emit is not truncated into unparseable output.
    """

    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
    ) -> None:
        super().__init__()
        self._model = foundation_model
        self._max_completion_tokens = max_completion_tokens

    def _generate(self, prompt: PromptValue, n: int, temperature: float, stop: list[str] | None) -> LLMResult:
        """Generate ``n`` completions for ``prompt`` via the foundation model.

        Parameters
        ----------
        prompt : PromptValue
            The RAGAS prompt to complete.
        n : int
            Number of completions to generate (at least one).
        temperature : float
            Sampling temperature forwarded to the foundation model.
        stop : list[str] | None
            Optional stop sequences; omitted from the call when falsy.

        Returns
        -------
        LLMResult
            A single-prompt result holding the generated completions.
        """
        content = prompt.to_string()
        generations = []
        for _ in range(max(1, n)):
            kwargs: dict[str, Any] = {
                "temperature": temperature,
                "max_completion_tokens": self._max_completion_tokens,
            }
            if stop:
                kwargs["stop"] = stop
            choices = self._model.chat([{"role": "user", "content": content}], **kwargs)
            generations.append(Generation(text=choices[0].message.content or ""))
        return LLMResult(generations=[generations])

    def generate_text(  # type: ignore[override]  # pylint: disable=unused-argument
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float = 1e-2,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Synchronously generate ``n`` completions for ``prompt``.

        Parameters
        ----------
        prompt : PromptValue
            The RAGAS prompt to complete.
        n : int, default=1
            Number of completions to generate.
        temperature : float, default=1e-2
            Sampling temperature forwarded to the foundation model.
        stop : list[str] | None, default=None
            Optional stop sequences.
        callbacks : Any, default=None
            RAGAS callback handles; unused by this adapter.

        Returns
        -------
        LLMResult
            The generated completions.
        """
        return self._generate(prompt, n, temperature, stop)

    async def agenerate_text(  # type: ignore[override]  # pylint: disable=unused-argument
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 1e-2,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Asynchronously generate completions by offloading to a thread.

        Parameters
        ----------
        prompt : PromptValue
            The RAGAS prompt to complete.
        n : int, default=1
            Number of completions to generate.
        temperature : float | None, default=1e-2
            Sampling temperature; ``None`` falls back to ``1e-2``.
        stop : list[str] | None, default=None
            Optional stop sequences.
        callbacks : Any, default=None
            RAGAS callback handles; unused by this adapter.

        Returns
        -------
        LLMResult
            The generated completions.
        """
        temp = 1e-2 if temperature is None else temperature
        return await asyncio.to_thread(self._generate, prompt, n, temp, stop)

    def is_finished(self, response: LLMResult) -> bool:  # type: ignore[override]  # pylint: disable=unused-argument
        """Report whether generation is complete.

        Parameters
        ----------
        response : LLMResult
            The result RAGAS is inspecting; unused because ai4rag responses
            are always returned fully formed.

        Returns
        -------
        bool
            Always ``True``.
        """
        return True


class AI4RAGRagasEmbeddings(BaseRagasEmbeddings):
    """RAGAS embeddings wrapper delegating to an ai4rag embedding model.

    Parameters
    ----------
    model : BaseEmbeddingModel
        The ai4rag embedding model that backs every RAGAS embedding call.
    """

    def __init__(self, model: BaseEmbeddingModel) -> None:
        super().__init__()
        self._model = model

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Parameters
        ----------
        text : str
            The query to embed.

        Returns
        -------
        list[float]
            The query embedding.
        """
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Parameters
        ----------
        texts : list[str]
            The documents to embed.

        Returns
        -------
        list[list[float]]
            One embedding per input document.
        """
        return self._model.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Async wrapper around :meth:`embed_query`.

        Parameters
        ----------
        text : str
            The query to embed.

        Returns
        -------
        list[float]
            The query embedding.
        """
        return await asyncio.to_thread(self._model.embed_query, text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper around :meth:`embed_documents`.

        Parameters
        ----------
        texts : list[str]
            The documents to embed.

        Returns
        -------
        list[list[float]]
            One embedding per input document.
        """
        return await asyncio.to_thread(self._model.embed_documents, texts)
