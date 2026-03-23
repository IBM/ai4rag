# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Sequence

from langchain_core.documents import Document

from ai4rag import logger
from ai4rag.rag.chunking.base_chunker import BaseChunker
from ai4rag.rag.foundation_models.base_model import BaseFoundationModel

__all__ = [
    "ContextualChunker",
]

DOCUMENT_SYSTEM_PROMPT = (
    "You are a helpful assistant that provides short contextual descriptions for document chunks. "
    "The following is the full source document:\n\n"
    "<document>\n{document}\n</document>"
)

SINGLE_CHUNK_PROMPT = (
    "Here is a chunk from the document provided in the system message:\n"
    "<chunk>\n{chunk}\n</chunk>\n\n"
    "Please give a short succinct context to situate this chunk within the overall "
    "document for the purposes of improving search retrieval of the chunk. "
    "Answer only with the succinct context and nothing else."
)

BATCH_CHUNK_PROMPT = (
    "For each of the following chunks from the document provided in the system message, "
    "provide a short succinct context to situate it within the overall document "
    "for improving search retrieval.\n\n"
    "{chunks_section}\n"
    'Respond with a JSON object containing a "contexts" key whose value is an array '
    "of objects, one per chunk in the same order. "
    'Each object must have "id" (integer) and "context" (string) keys.\n'
    "Example: "
    '{{"contexts": [{{"id": 1, "context": "..."}}, {{"id": 2, "context": "..."}}]}}'
)


def _build_chunks_section(chunks: list[Document]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(f'<chunk id="{i}">\n{chunk.page_content}\n</chunk>')
    return "\n".join(parts)


def _parse_batch_response(response_text: str, expected_count: int) -> list[str | None]:
    """Parse a JSON array batch response into a list of context strings.

    Returns a list of length ``expected_count``. Entries that could not be
    parsed are set to ``None``.
    """
    results: list[str | None] = [None] * expected_count

    # Extract JSON array from response (model may wrap it in markdown fences)
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return results

    # json_object mode returns a dict; extract the first list value from it
    if isinstance(parsed, dict):
        items = None
        for value in parsed.values():
            if isinstance(value, list):
                items = value
                break
        if items is None:
            return results
        parsed = items

    if not isinstance(parsed, list):
        return results

    for item in parsed:
        if isinstance(item, dict) and "id" in item and "context" in item:
            idx = int(item["id"]) - 1
            if 0 <= idx < expected_count:
                results[idx] = str(item["context"]).strip()

    return results


class ContextualChunker(BaseChunker[Document]):
    """Chunker that wraps a base chunker and prepends LLM-generated contextual
    descriptions to each chunk for improved retrieval quality.

    Parameters
    ----------
    base_chunker : BaseChunker[Document]
        The underlying chunker used for splitting documents into chunks.

    context_model : BaseFoundationModel
        Foundation model used to generate contextual descriptions.

    max_context_tokens : int, default=100
        Maximum number of tokens for the generated context description.
        Used as a budget hint; the actual output may be shorter.

    batch_size : int, default=10
        Number of chunks to process in a single LLM call.
        Set to 1 to disable batching.

    max_workers : int, default=4
        Maximum number of parallel threads for processing documents concurrently.

    max_document_size : int, default=100000
        Maximum document size in characters. Documents exceeding this limit
        are skipped for context enrichment — their chunks are kept as-is
        with ``contextualized=False``.

    system_prompt_template : str or None
        Custom system prompt template containing the document.
        Must contain a ``{document}`` placeholder.

    prompt_template : str or None
        Custom user prompt template for single-chunk context generation.
        Must contain a ``{chunk}`` placeholder.

    batch_prompt_template : str or None
        Custom user prompt template for batched context generation.
        Must contain a ``{chunks_section}`` placeholder.
    """

    def __init__(
        self,
        base_chunker: BaseChunker[Document],
        context_model: BaseFoundationModel,
        max_context_tokens: int = 100,
        batch_size: int = 10,
        max_workers: int = 4,
        max_document_size: int = 100_000,
        system_prompt_template: str | None = None,
        prompt_template: str | None = None,
        batch_prompt_template: str | None = None,
    ) -> None:
        self.base_chunker = base_chunker
        self.context_model = context_model
        self.max_context_tokens = max_context_tokens
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_document_size = max_document_size
        self.system_prompt_template = system_prompt_template or DOCUMENT_SYSTEM_PROMPT
        self.prompt_template = prompt_template or SINGLE_CHUNK_PROMPT
        self.batch_prompt_template = batch_prompt_template or BATCH_CHUNK_PROMPT

    def split_documents(self, documents: Sequence[Document]) -> list[Document]:
        """Split documents using the base chunker and prepend contextual descriptions.

        Parameters
        ----------
        documents : Sequence[Document]
            Source documents to chunk and contextualise.

        Returns
        -------
        list[Document]
            Chunks with contextual descriptions prepended to ``page_content``.
            Each chunk's metadata includes ``contextualized`` (bool) and
            ``original_page_content`` (str).
        """
        doc_content_map = {
            doc.metadata.get("document_id", str(hash(doc.page_content))): doc.page_content for doc in documents
        }

        chunks = self.base_chunker.split_documents(documents)

        grouped: dict[str, list[int]] = defaultdict(list)
        for idx, chunk in enumerate(chunks):
            doc_id = chunk.metadata.get("document_id", "")
            grouped[doc_id].append(idx)

        if self.max_workers > 1 and len(grouped) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._contextualize_document_chunks, doc_id, chunk_indices, chunks, doc_content_map
                    ): doc_id
                    for doc_id, chunk_indices in grouped.items()
                }
                for future in as_completed(futures):
                    future.result()
        else:
            for doc_id, chunk_indices in grouped.items():
                self._contextualize_document_chunks(doc_id, chunk_indices, chunks, doc_content_map)

        return chunks

    def _contextualize_document_chunks(
        self,
        doc_id: str,
        chunk_indices: list[int],
        chunks: list[Document],
        doc_content_map: dict[str, str],
    ) -> None:
        """Generate and prepend context for all chunks belonging to one document."""
        document_content = doc_content_map.get(doc_id, "")

        if len(document_content) > self.max_document_size:
            logger.warning(
                "Document '%s' exceeds max_document_size (%d > %d chars). "
                "Skipping contextual enrichment for its %d chunks.",
                doc_id,
                len(document_content),
                self.max_document_size,
                len(chunk_indices),
            )
            for idx in chunk_indices:
                chunk = chunks[idx]
                chunk.metadata["original_page_content"] = chunk.page_content
                chunk.metadata["contextualized"] = False
            return

        for batch_start in range(0, len(chunk_indices), self.batch_size):
            batch_indices = chunk_indices[batch_start : batch_start + self.batch_size]
            batch_chunks = [chunks[i] for i in batch_indices]

            contexts = self._generate_contexts_batch(document_content, batch_chunks, doc_id)

            for i, idx in enumerate(batch_indices):
                chunk = chunks[idx]
                original_content = chunk.page_content
                chunk.metadata["original_page_content"] = original_content

                if contexts[i] is not None:
                    chunk.page_content = f"[Context: {contexts[i]}]\n{original_content}"
                    chunk.metadata["contextualized"] = True
                else:
                    chunk.metadata["contextualized"] = False

    def _generate_contexts_batch(
        self, document_content: str, batch_chunks: list[Document], doc_id: str
    ) -> list[str | None]:
        """Generate contexts for a batch of chunks. Falls back to single-chunk on failure."""
        if len(batch_chunks) == 1:
            return [self._generate_single_context(document_content, batch_chunks[0], doc_id)]

        try:
            contexts = self._call_batch(document_content, batch_chunks, doc_id)
            failed_indices = [i for i, ctx in enumerate(contexts) if ctx is None]

            if failed_indices:
                logger.warning(
                    "Batch context parsing incomplete for document '%s': %d/%d chunks missing. "
                    "Retrying individually.",
                    doc_id,
                    len(failed_indices),
                    len(batch_chunks),
                )
                for i in failed_indices:
                    contexts[i] = self._generate_single_context(document_content, batch_chunks[i], doc_id)

            return contexts

        except Exception:  # pylint: disable=broad-exception-caught
            logger.warning(
                "Batch context generation failed for document '%s'. Falling back to single-chunk mode.",
                doc_id,
                exc_info=True,
            )
            return [self._generate_single_context(document_content, chunk, doc_id) for chunk in batch_chunks]

    def _build_system_message(self, document_content: str) -> dict[str, str]:
        """Build the system message containing the full document."""
        return {"role": "system", "content": self.system_prompt_template.format(document=document_content)}

    def _call_batch(self, document_content: str, batch_chunks: list[Document], doc_id: str) -> list[str | None]:
        """Make a single LLM call for a batch of chunks and parse the response."""
        chunks_section = _build_chunks_section(batch_chunks)

        user_prompt = self.batch_prompt_template.format(chunks_section=chunks_section)

        messages = [
            self._build_system_message(document_content),
            {"role": "user", "content": user_prompt},
        ]
        response = self.context_model.chat(
            messages,
            response_format={"type": "json_object"},
            max_completion_tokens=len(batch_chunks) * self.max_context_tokens,
            extra_body={"prompt_cache_key": doc_id},
        )
        response_text = response[0].message.content

        return _parse_batch_response(response_text, len(batch_chunks))

    def _generate_single_context(self, document_content: str, chunk: Document, doc_id: str) -> str | None:
        """Generate context for a single chunk. Returns None on failure."""
        try:
            user_prompt = self.prompt_template.format(chunk=chunk.page_content)

            messages = [
                self._build_system_message(document_content),
                {"role": "user", "content": user_prompt},
            ]
            response = self.context_model.chat(
                messages, prompt_cache_key=doc_id, extra_body={"prompt_cache_key": doc_id}
            )
            return response[0].message.content.strip()

        except Exception:  # pylint: disable=broad-exception-caught
            seq = chunk.metadata.get("sequence_number", "?")
            logger.warning(
                "Contextual retrieval failed for chunk %s of document '%s', using original content.",
                seq,
                doc_id,
                exc_info=True,
            )
            return None

    def to_dict(self) -> dict[str, Any]:
        """Return dictionary that can be used to recreate an instance of the ContextualChunker."""
        return {
            "base_chunker": self.base_chunker.to_dict(),
            "context_model_id": self.context_model.model_id,
            "max_context_tokens": self.max_context_tokens,
            "batch_size": self.batch_size,
            "max_workers": self.max_workers,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ContextualChunker":
        """Create an instance from the dictionary.

        Note: This method cannot fully reconstruct the chunker because it requires
        a live ``context_model`` instance. Use this only for serialization of config
        metadata. For full reconstruction, instantiate manually.
        """
        raise NotImplementedError(
            "ContextualChunker.from_dict() is not supported because it requires "
            "a live foundation model instance. Reconstruct manually."
        )
