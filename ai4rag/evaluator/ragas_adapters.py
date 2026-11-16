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
from ragas.exceptions import LLMDidNotFinishException
from ragas.llms.base import BaseRagasLLM
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_random_exponential

from ai4rag.evaluator.ragas_diagnostics import failure_reason
from ai4rag.rag.embedding.base_model import BaseEmbeddingModel
from ai4rag.rag.embedding.openai_model import OpenAIEmbeddingModel
from ai4rag.rag.embedding.request_executor import QUERY_POLICY
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

# RAGAS faithfulness NLI emits one {statement, reason, verdict} object per
# claim. Long answers (10–16+ statements) routinely need >1k completion tokens;
# truncating mid-JSON surfaces as OutputParserException / NaN faithfulness.
# Keep this above that envelope while staying well under typical 32k windows.
DEFAULT_MAX_COMPLETION_TOKENS = 2048


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
        """Bind the foundation model and per-completion token limit."""
        super().__init__()
        self._model = foundation_model
        self._max_completion_tokens = max_completion_tokens

    async def generate(  # type: ignore[override]
        self,
        prompt: PromptValue,
        n: int = 1,
        temperature: float | None = 1e-2,
        stop: list[str] | None = None,
        callbacks: Any = None,
    ) -> LLMResult:
        """Apply RAGAS retries except for confirmed context overflow.

        Context-overflow failures are not retried with the same prompt. Other
        configured exception types retain the RAGAS attempt and wait limits.

        Parameters
        ----------
        prompt : PromptValue
            The RAGAS prompt to complete.
        n : int, default=1
            Number of completions requested; values below one still generate one.
        temperature : float | None, default=1e-2
            Sampling temperature; ``None`` uses RAGAS's default for ``n``.
        stop : list[str] | None, default=None
            Optional stop sequences.
        callbacks : Any, default=None
            Callback handles forwarded to the asynchronous adapter.

        Returns
        -------
        LLMResult
            The generated completions.
        """
        retrying = AsyncRetrying(
            wait=wait_random_exponential(multiplier=1, max=self.run_config.max_wait),
            stop=stop_after_attempt(self.run_config.max_retries),
            retry=retry_if_exception(
                lambda error: isinstance(error, self.run_config.exception_types)
                and failure_reason(error) != "context_overflow"
            ),
            reraise=True,
        )
        result = await retrying(
            self.agenerate_text,
            prompt=prompt,
            n=n,
            temperature=self.get_temperature(n) if temperature is None else temperature,
            stop=stop,
            callbacks=callbacks,
        )
        if not self.is_finished(result):
            raise LLMDidNotFinishException()
        return result

    def _generate(self, prompt: PromptValue, n: int, temperature: float, stop: list[str] | None) -> LLMResult:
        """Generate ``n`` completions for ``prompt`` via the foundation model.

        Parameters
        ----------
        prompt : PromptValue
            The RAGAS prompt to complete.
        n : int
            Number of completions requested; values below one still generate one.
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
            Number of completions requested; values below one still generate one.
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
            Number of completions requested; values below one still generate one.
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
        """Accept the generation result without inspecting completion status.

        Parameters
        ----------
        response : LLMResult
            The result RAGAS is inspecting; unused by this adapter.

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

    Notes
    -----
    OpenAI document calls select QUERY_POLICY and delegate recovery to the
    embedding executor. Other providers retain their own model behavior and
    the RAGAS retry wrapper used by embed_texts. Async entry points offload
    synchronous model calls to worker threads.
    """

    def __init__(self, model: BaseEmbeddingModel) -> None:
        """Bind the embedding model without creating a separate endpoint client."""
        super().__init__()
        self._model = model

    async def embed_texts(self, texts: list[str], is_async: bool = True) -> list[list[float]]:
        """Embed texts using the appropriate model or RAGAS recovery owner.

        OpenAIEmbeddingModel calls run in a worker thread with QUERY_POLICY,
        bypassing the additional RAGAS retry wrapper. Other providers delegate
        to the base implementation and its configured RAGAS retries.

        Parameters
        ----------
        texts : list[str]
            Texts to embed.
        is_async : bool, default=True
            Select the RAGAS async or executor path for other providers.
            Ignored for OpenAIEmbeddingModel, which always uses a worker thread.

        Returns
        -------
        list[list[float]]
            The wrapped model's vectors in input order.
        """
        if isinstance(self._model, OpenAIEmbeddingModel):
            return await self.aembed_documents(texts)
        return await super().embed_texts(texts, is_async=is_async)

    def embed_query(self, text: str) -> list[float]:
        """Delegate one query to the wrapped model without adding adapter retries.

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
        """Embed documents with short recovery waits for OpenAI models.

        OpenAIEmbeddingModel receives QUERY_POLICY, including its timeout and
        input-repair settings. Other providers receive an unchanged delegation
        call. This method adds no RAGAS retry wrapper.

        Parameters
        ----------
        texts : list[str]
            Texts to embed; the wrapped model determines batch sizes.

        Returns
        -------
        list[list[float]]
            The wrapped model's vectors in input order.
        """
        if isinstance(self._model, OpenAIEmbeddingModel):
            return self._model.embed_documents(texts, policy=QUERY_POLICY)
        return self._model.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Run the wrapped model's query embedding in a worker thread.

        Model exceptions propagate without an additional adapter retry loop.

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
        """Run this adapter's document embedding method in a worker thread.

        Delegation through embed_documents preserves its OpenAI QUERY_POLICY
        selection. Model exceptions propagate without another retry wrapper.

        Parameters
        ----------
        texts : list[str]
            The documents to embed.

        Returns
        -------
        list[list[float]]
            The wrapped model's vectors in input order.
        """
        return await asyncio.to_thread(self.embed_documents, texts)
