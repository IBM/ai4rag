# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from typing import Any

from ai4rag.rag.chunking.chunk import AI4RAGChunk

from ..foundation_models.base_model import BaseFoundationModel, MessageTyped
from ..retrieval.retriever import Retriever
from .base_template import BaseRAGTemplate


class AgenticRAG(BaseRAGTemplate):
    """Iterative RAG template with bounded query refinement.

    The agent retrieves the initial question, asks the foundation model for a
    focused follow-up query, and retrieves again until no new chunks are found
    or the step limit is reached. This keeps the change local to the agentic
    template while preserving the existing RAG template interface.
    """

    def __init__(
        self,
        foundation_model: BaseFoundationModel,
        retriever: Retriever,
        max_retrieval_steps: int = 2,
    ):
        super().__init__(foundation_model=foundation_model, retriever=retriever)
        if max_retrieval_steps < 1:
            raise ValueError("max_retrieval_steps must be at least 1")
        self.max_retrieval_steps = max_retrieval_steps

    @staticmethod
    def _chunk_key(chunk: AI4RAGChunk) -> tuple[str, str]:
        """Return a stable key used to remove duplicate retrieved chunks."""
        return chunk.text, repr(chunk.metadata)

    def _follow_up_query(self, question: str, documents: list[AI4RAGChunk]) -> str:
        """Ask the model for a focused query that fills context gaps."""
        context = "\n\n".join(document.text[:2000] for document in documents)
        response = self.foundation_model.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Create one concise follow-up search query for the original question. "
                        "Use the retrieved context to target information that is still missing. "
                        "Return only the query, without explanations."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Original question: {question}\nRetrieved context:\n{context}",
                },
            ],
            temperature=0,
            max_completion_tokens=128,
        )
        query = response[0].message.content if response else ""
        return query.strip() if query else question

    def _retrieve_until_sufficient(self, question: str, **kwargs) -> list[AI4RAGChunk]:
        """Retrieve multiple query views until no new context is found."""
        documents: list[AI4RAGChunk] = []
        seen: set[tuple[str, str]] = set()
        query = question

        for step in range(self.max_retrieval_steps):
            retrieved = self.retriever.retrieve(query, **kwargs)
            new_documents = [document for document in retrieved if self._chunk_key(document) not in seen]
            documents.extend(new_documents)
            seen.update(self._chunk_key(document) for document in new_documents)

            if step == self.max_retrieval_steps - 1 or not new_documents:
                break
            query = self._follow_up_query(question, documents)

        return documents

    def _build_enriched_user_message(self, question: str, **kwargs) -> tuple[list[AI4RAGChunk], str]:
        """Retrieve iteratively and render the final enriched user message."""
        reference_documents = self._retrieve_until_sufficient(question, **kwargs)
        return self._render_enriched_user_message(question, reference_documents)

    def generate(self, question: str, **kwargs) -> dict[str, Any]:
        """Generate an answer after bounded iterative retrieval."""
        reference_documents, user_message = self._build_enriched_user_message(question, **kwargs)
        response = self.foundation_model.chat(
            messages=[
                {"role": "system", "content": self.foundation_model.system_message_text},
                {"role": "user", "content": user_message},
            ]
        )
        return {
            "answer": response[0].message.content,
            "reference_documents": reference_documents,
            "question": question,
        }

    def generate_stream(self, question: str, **kwargs):
        """Yield the complete answer as one chunk until streaming is supported."""
        yield self.generate(question, **kwargs)["answer"]

    def respond(self, messages: list[MessageTyped]):
        """Retrieve context and return a native Responses API response."""
        rag_messages = self._build_rag_messages(messages)
        return self.foundation_model.responses(messages=rag_messages)

    def chat(self, messages: list[MessageTyped], **kwargs) -> list[Any]:
        """Run bounded iterative retrieval for the last user message."""
        rag_messages = self._build_rag_messages(messages, **kwargs)
        return self.foundation_model.chat(messages=rag_messages, **kwargs)
