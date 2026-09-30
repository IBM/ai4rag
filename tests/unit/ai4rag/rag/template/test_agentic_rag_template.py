# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.template.agentic_rag_template import AgenticRAG
from ai4rag.rag.template.base_template import BaseRAGTemplate


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


def test_follow_up_query_uses_full_retrieved_chunk(mocker):
    """The query model sees the complete chunk, including text after 2000 characters."""
    foundation_model = mocker.MagicMock()
    foundation_model.chat.return_value = [mocker.MagicMock(message=mocker.MagicMock(content="next query"))]
    rag = AgenticRAG(foundation_model=foundation_model, retriever=mocker.MagicMock())
    chunk = AI4RAGChunk(text="x" * 2000 + "important ending", metadata={})

    rag._follow_up_query("question", [chunk])

    assert "important ending" in foundation_model.chat.call_args.kwargs["messages"][1]["content"]


def test_agentic_rag_rephrases_question_after_empty_first_retrieval(mocker):
    """An empty first retrieval triggers a new search using the rephrased question."""
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer the question."
    foundation_model.user_message_text = "Context: {reference_documents}\nQuestion: {question}"
    foundation_model.context_template_text = "{document}"
    foundation_model.chat.side_effect = [
        [mocker.MagicMock(message=mocker.MagicMock(content="rephrased search query"))],
        [mocker.MagicMock(message=mocker.MagicMock(content="answer"))],
    ]
    chunk = AI4RAGChunk(text="found context", metadata={"id": "found"})
    retriever = mocker.MagicMock()
    retriever.retrieve.side_effect = [[], [chunk]]

    result = AgenticRAG(foundation_model=foundation_model, retriever=retriever).generate("question")

    assert result["reference_documents"] == [chunk]
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "rephrased search query"]
    assert "Retrieved context" not in foundation_model.chat.call_args_list[0].kwargs["messages"][1]["content"]
    assert foundation_model.chat.call_count == 2


def test_agentic_rag_answers_with_empty_context_after_two_empty_retrievals(mocker):
    """The retrieval limit still applies when no documents are found."""
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer the question."
    foundation_model.user_message_text = "Context: {reference_documents}\nQuestion: {question}"
    foundation_model.chat.side_effect = [
        [mocker.MagicMock(message=mocker.MagicMock(content="rephrased search query"))],
        [mocker.MagicMock(message=mocker.MagicMock(content="answer"))],
    ]
    retriever = mocker.MagicMock()
    retriever.retrieve.side_effect = [[], []]

    result = AgenticRAG(foundation_model=foundation_model, retriever=retriever).generate("question")

    assert result == {"answer": "answer", "reference_documents": [], "question": "question"}
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "rephrased search query"]
    assert foundation_model.chat.call_count == 2


def test_agentic_rag_skips_retrieval_when_rephrased_query_is_unchanged(mocker):
    """Avoid searching again when the model repeats the original question."""
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer the question."
    foundation_model.user_message_text = "Context: {reference_documents}\nQuestion: {question}"
    foundation_model.chat.side_effect = [
        [mocker.MagicMock(message=mocker.MagicMock(content=" QUESTION "))],
        [mocker.MagicMock(message=mocker.MagicMock(content="answer"))],
    ]
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = []

    result = AgenticRAG(foundation_model=foundation_model, retriever=retriever).generate("question")

    assert result["reference_documents"] == []
    retriever.retrieve.assert_called_once_with("question")
    assert foundation_model.chat.call_count == 2
