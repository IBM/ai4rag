# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import json
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.template.agentic_rag_template import AgenticRAG
from ai4rag.rag.template.base_template import BaseRAGTemplate


class ToolCallingFake(FakeMessagesListChatModel):
    """A local model that follows one LangChain tool call with an answer."""

    def bind_tools(self, _tools, **_kwargs):
        return self


@pytest.fixture
def rag(mocker):
    create_agent = mocker.patch("ai4rag.rag.template.agentic_rag_template.create_agent")
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer from context."
    foundation_model.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation_model.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = []
    rag_template = AgenticRAG(foundation_model=foundation_model, retriever=retriever, chat_model=mocker.MagicMock())
    rag_template._create_agent_mock = create_agent
    return rag_template


def _tool(create_agent, name):
    return next(item for item in create_agent.call_args.kwargs["tools"] if item.name == name)


def _call_tool(create_agent, name, arguments, context):
    tool = _tool(create_agent, name)
    return tool.func(**arguments, runtime=SimpleNamespace(context=context))


def test_real_agent_graph_rewrites_then_retrieves(mocker):
    """LangChain executes both tools and returns their observations to the model."""
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer from context."
    foundation_model.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation_model.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    initial_chunk = AI4RAGChunk(text="initial evidence", metadata={})
    follow_up_chunk = AI4RAGChunk(text="retrieved evidence", metadata={})
    retriever.retrieve.side_effect = [[initial_chunk], [follow_up_chunk]]
    model = ToolCallingFake(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "rewrite_query", "args": {"missing_information": "the answer"}, "id": "call_1"}],
            ),
            AIMessage(content="focused search"),
            AIMessage(
                content="", tool_calls=[{"name": "retriever", "args": {"query": "focused search"}, "id": "call_2"}]
            ),
            AIMessage(content="grounded answer"),
        ]
    )
    agent = AgenticRAG(foundation_model=foundation_model, retriever=retriever, chat_model=model)

    result = agent.generate("question")

    assert result == {
        "answer": "grounded answer",
        "reference_documents": [initial_chunk, follow_up_chunk],
        "question": "question",
    }
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "focused search"]


def test_agentic_rag_uses_langchain_agent(rag, mocker):
    """The model may answer directly after the required initial retrieval."""
    chunk = AI4RAGChunk(text="initial evidence", metadata={"document_id": "doc1"})
    rag.retriever.retrieve.return_value = [chunk]
    rag.foundation_model.user_message_text = "Context: {reference_documents}\nQuestion: {question}\nAnswer in English."
    graph = rag._create_agent_mock.return_value
    graph.invoke.return_value = {"messages": [AIMessage(content="answer")]}

    result = rag.generate("question")

    assert isinstance(rag, BaseRAGTemplate)
    assert result == {"answer": "answer", "reference_documents": [chunk], "question": "question"}
    rag.retriever.retrieve.assert_called_once_with("question")
    assert graph.invoke.call_args.args[0] == {
        "messages": [{"role": "user", "content": "Context: initial evidence\nQuestion: question\nAnswer in English."}]
    }


def test_agent_collects_tool_results_for_evaluation(rag, mocker):
    """Agent tool calls return context and preserve retrieved chunks for scoring."""
    first = AI4RAGChunk(text="first context", metadata={"source": "first"})
    second = AI4RAGChunk(text="second context", metadata={"source": "second"})
    rag.retriever.retrieve.side_effect = [[first], [first, second]]
    create_agent = rag._create_agent_mock

    def run_graph(*_args, **kwargs):
        context = kwargs["context"]
        assert "second context" in _call_tool(create_agent, "retriever", {"query": "second query"}, context)
        assert "limit reached" in _call_tool(create_agent, "retriever", {"query": "third query"}, context).lower()
        return {"messages": [AIMessage(content="grounded answer")]}

    create_agent.return_value.invoke.side_effect = run_graph

    result = rag.generate("question")

    assert result["reference_documents"] == [first, second]
    assert [call.args[0] for call in rag.retriever.retrieve.call_args_list] == ["question", "second query"]
    assert create_agent.call_args.kwargs["model"] is rag.chat_model
    assert "Answer from context." in create_agent.call_args.kwargs["system_prompt"]


def test_empty_tool_result_can_be_followed_by_another_search(rag, mocker):
    """The agent can search again when the initial retrieval finds no documents."""
    rag.retriever.retrieve.side_effect = [[], [AI4RAGChunk(text="found", metadata={})]]
    create_agent = rag._create_agent_mock

    def run_graph(*_args, **kwargs):
        context = kwargs["context"]
        assert "found" in _call_tool(create_agent, "retriever", {"query": "second"}, context)
        assert "limit reached" in _call_tool(create_agent, "retriever", {"query": "third"}, context).lower()
        return {"messages": [AIMessage(content="answer")]}

    create_agent.return_value.invoke.side_effect = run_graph

    assert rag.generate("question")["reference_documents"][0].text == "found"


def test_rewrite_tool_uses_missing_information_and_retrieved_context(rag, mocker):
    """A separate bounded tool asks the model for a focused search query."""
    rag.chat_model.invoke.return_value = AIMessage(content="focused missing fact")
    rag.retriever.retrieve.return_value = [AI4RAGChunk(text="known fact", metadata={})]
    create_agent = rag._create_agent_mock

    def run_graph(*_args, **kwargs):
        context = kwargs["context"]
        args = {"missing_information": "missing date"}
        assert _call_tool(create_agent, "rewrite_query", args, context) == "focused missing fact"
        assert _call_tool(create_agent, "rewrite_query", args, context) == "focused missing fact"
        assert (
            "limit reached"
            in _call_tool(create_agent, "rewrite_query", {"missing_information": "another fact"}, context).lower()
        )
        return {"messages": [AIMessage(content="answer")]}

    create_agent.return_value.invoke.side_effect = run_graph

    rag.generate("question")

    prompt = rag.chat_model.invoke.call_args.args[0][1].content
    assert "missing date" in prompt
    assert "known fact" in prompt
    assert rag.chat_model.invoke.call_count == 2


def test_chat_returns_choice_with_agent_answer(rag, mocker):
    """The existing chat interface can consume the agent's final answer."""
    mocker.patch.object(rag, "_run_agent", return_value=("answer", []))

    choices = rag.chat([{"role": "user", "content": "question"}])

    assert choices[0].message.content == "answer"


def test_chat_completions_executes_retriever_tool(mocker):
    """The LangChain agent sends tools to Chat Completions and executes the returned call."""
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        if len(requests) == 1:
            assert {item["function"]["name"] for item in requests[0]["tools"]} == {"retriever", "rewrite_query"}
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "retriever", "arguments": '{"query":"focused search"}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            assert any(message["role"] == "tool" for message in requests[1]["messages"])
            message = {"role": "assistant", "content": "grounded answer"}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            },
        )

    chat_model = ChatOpenAI(
        model="model",
        api_key="test",
        base_url="https://maas.example/v1",
        use_responses_api=False,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer from context."
    foundation_model.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation_model.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    initial = AI4RAGChunk(text="initial evidence", metadata={})
    follow_up = AI4RAGChunk(text="follow-up evidence", metadata={})
    retriever.retrieve.side_effect = [[initial], [follow_up]]

    result = AgenticRAG(foundation_model, retriever, chat_model=chat_model).generate("question")

    assert result["answer"] == "grounded answer"
    assert result["reference_documents"] == [initial, follow_up]
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "focused search"]
    assert len(requests) == 2


def test_retrieval_limit_is_validated_on_assignment(rag):
    """The retrieval budget remains valid when changed after construction."""
    with pytest.raises(ValueError, match="positive integer"):
        rag.max_retrieval_steps = 0
    with pytest.raises(ValueError, match="positive integer"):
        rag.max_retrieval_steps = True

    rag.max_retrieval_steps = 3
    assert rag.max_retrieval_steps == 3


def test_initial_retrieval_counts_toward_limit(rag, mocker):
    """A one-step budget cannot issue another retrieval from the agent."""
    rag.max_retrieval_steps = 1
    create_agent = rag._create_agent_mock

    def run_graph(*_args, **kwargs):
        context = kwargs["context"]
        assert "limit reached" in _call_tool(create_agent, "retriever", {"query": "another query"}, context).lower()
        return {"messages": [AIMessage(content="answer")]}

    create_agent.return_value.invoke.side_effect = run_graph

    rag.generate("question")

    rag.retriever.retrieve.assert_called_once_with("question")
