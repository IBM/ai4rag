# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from ai4rag.rag.template.agentic_rag_template import AgenticRAG
from ai4rag.rag.template.base_template import BaseRAGTemplate
from ai4rag.rag.chunking.chunk import AI4RAGChunk


def test_agentic_rag_is_a_base_template(mocker):
    """AgenticRAG exposes the common RAG template contract."""
    foundation_model = mocker.MagicMock()
    retriever = mocker.MagicMock()

    rag = AgenticRAG(foundation_model=foundation_model, retriever=retriever)

    assert isinstance(rag, BaseRAGTemplate)
    assert rag.foundation_model is foundation_model
    assert rag.retriever is retriever


def test_agentic_rag_retrieves_again_with_a_follow_up_query(mocker):
    """AgenticRAG performs a second retrieval when new context is available."""
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer the question."
    foundation_model.user_message_text = "Context: {reference_documents}\nQuestion: {question}"
    foundation_model.context_template_text = "{document}"

    query_response = mocker.MagicMock()
    query_response.message.content = "follow-up search query"
    answer_response = mocker.MagicMock()
    answer_response.message.content = "answer"
    foundation_model.chat.side_effect = [[query_response], [answer_response]]

    first_chunk = AI4RAGChunk(text="first context", metadata={"id": "first"})
    second_chunk = AI4RAGChunk(text="second context", metadata={"id": "second"})
    retriever = mocker.MagicMock()
    retriever.retrieve.side_effect = [[first_chunk], [second_chunk]]

    result = AgenticRAG(foundation_model=foundation_model, retriever=retriever).generate("question")

    assert result["reference_documents"] == [first_chunk, second_chunk]
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "follow-up search query"]
    assert foundation_model.chat.call_count == 2
