# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from openai import BadRequestError

from ai4rag.rag.chunking.chunk import AI4RAGChunk
from ai4rag.rag.retrieval.document_compressor import LLMDocumentCompressor
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
    rag_template = AgenticRAG(
        foundation_model=foundation_model, retriever=retriever, max_retrieval_steps=2, chat_model=mocker.MagicMock()
    )
    rag_template._create_agent_mock = create_agent
    return rag_template


def _tool(create_agent, name):
    return next(item for item in create_agent.call_args.kwargs["tools"] if item.name == name)


def _call_tool(create_agent, name, arguments, context):
    tool = _tool(create_agent, name)
    return tool.func(**arguments, runtime=SimpleNamespace(context=context))


def test_real_agent_graph_rewrites_then_retrieves(mocker, caplog):
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

    assert result["answer"] == "grounded answer"
    assert result["reference_documents"] == [initial_chunk, follow_up_chunk]
    assert result["question"] == "question"
    assert result["conversation"] == {
        "question": "question",
        "initial_retrieval": {"query": "question", "documents": [{"text": "initial evidence", "metadata": {}}]},
        "context_compression_fallback": False,
        "compression": None,
        "messages": [
            {"type": "human", "content": "Question: question\nContext: initial evidence"},
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {
                        "name": "rewrite_query",
                        "args": {"missing_information": "the answer"},
                        "id": "call_1",
                        "type": "tool_call",
                    }
                ],
            },
            {"type": "tool", "tool_call_id": "call_1", "name": "rewrite_query", "output": "focused search"},
            {
                "type": "ai",
                "content": "",
                "tool_calls": [
                    {"name": "retriever", "args": {"query": "focused search"}, "id": "call_2", "type": "tool_call"}
                ],
            },
            {
                "type": "tool",
                "tool_call_id": "call_2",
                "name": "retriever",
                "output": "Question: question\nContext: retrieved evidence",
            },
            {"type": "ai", "content": "grounded answer", "tool_calls": []},
        ],
        "tool_execution": {
            "rewrite_query": [{"input": {"missing_information": "the answer"}, "output": "focused search"}],
            "retriever": [
                {
                    "input": {"query": "focused search"},
                    "output": "Question: question\nContext: retrieved evidence",
                    "documents": [{"text": "retrieved evidence", "metadata": {}}],
                }
            ],
        },
    }
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "focused search"]
    assert "RAG_FAILURE" not in caplog.text


def test_default_budget_allows_two_rewrite_and_retrieval_cycles(mocker):
    """Both follow-up searches complete before the default budget forces an answer."""
    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    chunks = [AI4RAGChunk(text=text, metadata={}) for text in ("initial evidence", "first fact", "second fact")]
    retriever.retrieve.side_effect = [[chunk] for chunk in chunks]
    responses = []
    for index, query in enumerate(("first fact", "second fact")):
        responses.extend(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "rewrite_query", "args": {"missing_information": query}, "id": f"rewrite_{index}"}
                    ],
                ),
                AIMessage(content=query),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "retriever", "args": {"query": query}, "id": f"search_{index}"}],
                ),
            ]
        )
    responses.append(AIMessage(content="grounded answer"))
    agent = AgenticRAG(foundation, retriever, chat_model=ToolCallingFake(responses=responses))

    result = agent.generate("question")

    assert result["answer"] == "grounded answer"
    assert result["reference_documents"] == chunks
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "first fact", "second fact"]
    assert len(result["conversation"]["tool_execution"]["rewrite_query"]) == 2
    assert len(result["conversation"]["tool_execution"]["retriever"]) == 2


def test_agent_initialization_does_not_emit_lock_schema_warning(mocker):
    """The invocation lock remains opaque while LangChain creates tool schemas."""
    create_agent = mocker.patch("ai4rag.rag.template.agentic_rag_template.create_agent")
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer from context."
    retriever = mocker.MagicMock()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        AgenticRAG(foundation_model=foundation_model, retriever=retriever, chat_model=mocker.MagicMock())

    assert create_agent.called
    assert not any("allocate_lock" in str(warning.message) for warning in caught_warnings)


def test_agentic_rag_uses_langchain_agent(rag, mocker):
    """The model may answer directly after the required initial retrieval."""
    chunk = AI4RAGChunk(text="initial evidence", metadata={"document_id": "doc1"})
    rag.retriever.retrieve.return_value = [chunk]
    rag.foundation_model.user_message_text = "Context: {reference_documents}\nQuestion: {question}\nAnswer in English."
    graph = rag._create_agent_mock.return_value
    graph.invoke.return_value = {"messages": [AIMessage(content="answer")]}

    result = rag.generate("question")

    assert isinstance(rag, BaseRAGTemplate)
    assert result["answer"] == "answer"
    assert result["reference_documents"] == [chunk]
    assert result["question"] == "question"
    assert result["conversation"]["initial_retrieval"] == {
        "query": "question",
        "documents": [{"text": "initial evidence", "metadata": {"document_id": "doc1"}}],
    }
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
            assert "tools" not in requests[1]
            assert "tool budget is exhausted" in requests[1]["messages"][0]["content"]
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

    result = AgenticRAG(foundation_model, retriever, max_retrieval_steps=2, chat_model=chat_model).generate("question")

    assert result["answer"] == "grounded answer"
    assert result["reference_documents"] == [initial, follow_up]
    assert [call.args[0] for call in retriever.retrieve.call_args_list] == ["question", "focused search"]
    assert len(requests) == 2


@pytest.mark.parametrize("budget", [1, 2, 3])
@pytest.mark.parametrize("empty_context", [False, True])
@pytest.mark.parametrize(
    "language,answer", [("English", "Evidence is insufficient."), ("Polish", "Brakuje wystarczających informacji.")]
)
def test_real_graph_finalizes_at_retrieval_budget(mocker, budget, empty_context, language, answer):
    """Repeated search requests terminate with an answer even when retrieval is empty."""
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if payload.get("tools"):
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"search_{len(requests)}",
                        "type": "function",
                        "function": {"name": "retriever", "arguments": '{"query":"more evidence"}'},
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            instructions = payload["messages"][0]["content"]
            assert "tool budget is exhausted" in instructions
            assert "say that you could not produce an answer from the retrieved evidence" in instructions
            assert "Follow the response-language instructions already provided" in instructions
            assert any(f"Respond exclusively in {language}." in message["content"] for message in payload["messages"])
            message = {"role": "assistant", "content": answer}
            finish_reason = "stop"
        return httpx.Response(
            200,
            json={
                "id": f"response_{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "model",
                "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
            },
        )

    model = ChatOpenAI(
        model="model",
        api_key="test",
        base_url="https://maas.example/v1",
        use_responses_api=False,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    foundation.user_message_text += f"\nRespond exclusively in {language}."
    chunks = [] if empty_context else [AI4RAGChunk(text="indexed evidence", metadata={})]
    retriever.retrieve.return_value = chunks
    agent = AgenticRAG(foundation, retriever, budget, model)

    # Reuse the graph to ensure turn counts and finalization remain request-scoped.
    for _ in range(2):
        requests.clear()
        result = agent.generate("question")
        assert result["answer"] == answer
        assert result["reference_documents"] == chunks
        assert len(requests) == budget
        assert result["conversation"]["messages"][-1]["tool_calls"] == []
    assert retriever.retrieve.call_count == 2 * budget


@pytest.mark.parametrize("tool_name", ["rewrite_query", "unknown_tool"])
def test_real_graph_bounds_calls_without_retrieval_progress(mocker, tool_name):
    """Rewrite-only and invalid-tool loops cannot consume the graph recursion limit."""
    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = []
    arguments = {"missing_information": "date"} if tool_name == "rewrite_query" else {}
    model = ToolCallingFake(
        responses=[
            AIMessage(content="", tool_calls=[{"name": tool_name, "args": arguments, "id": f"call_{index}"}])
            for index in range(20)
        ]
    )
    generate = mocker.spy(model, "_generate")
    agent = AgenticRAG(foundation, retriever, max_retrieval_steps=2, chat_model=model)

    with pytest.raises(RuntimeError, match="Model returned tool calls after the tool budget was exhausted"):
        agent.generate("question")

    assert generate.call_count == (7 if tool_name == "rewrite_query" else 5)
    retriever.retrieve.assert_called_once_with("question")


def test_real_graph_keeps_concurrent_budgets_independent(mocker, caplog):
    """Questions sharing an agent each retain their own retrieval and turn budgets."""
    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = []
    # Every response requests another search, including the final tool-free
    # response. Each invocation must stop with its own failure trace.
    model = ToolCallingFake(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "retriever", "args": {"query": "again"}, "id": "search"}])
        ]
    )
    agent = AgenticRAG(foundation, retriever, max_retrieval_steps=2, chat_model=model)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(agent.generate, f"question {index}") for index in range(4)]
        for future in futures:
            with pytest.raises(RuntimeError, match="Model returned tool calls after the tool budget was exhausted"):
                future.result()

    assert retriever.retrieve.call_count == 8
    traces = _failure_traces(caplog)
    assert len(traces) == 4
    assert len({trace["diagnostic_id"] for trace in traces}) == 4
    assert all(trace["tool_calls"]["retriever"] == 1 for trace in traces)
    assert all(trace["model_calls"] == 2 for trace in traces)


def test_real_graph_rejects_tool_calls_returned_after_finalization(mocker, caplog):
    """A provider ignoring the tool-free request cannot restart retrieval."""
    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    retriever.retrieve.return_value = []
    model = ToolCallingFake(
        responses=[
            AIMessage(content="", tool_calls=[{"name": "retriever", "args": {"query": "again"}, "id": "repeat"}])
        ]
    )

    with pytest.raises(RuntimeError, match="Model returned tool calls after the tool budget was exhausted"):
        AgenticRAG(foundation, retriever, 1, model).generate("question")

    retriever.retrieve.assert_called_once_with("question")
    traces = _failure_traces(caplog)
    assert len(traces) == 1
    assert traces[0]["error_type"] == "RuntimeError"
    assert traces[0]["model_calls"] == 1
    assert traces[0]["tool_calls"]["retriever"] == 0


def test_retrieval_limit_is_validated_on_assignment(rag):
    """The retrieval budget remains valid when changed after construction."""
    with pytest.raises(ValueError, match="positive integer"):
        rag.max_retrieval_steps = 0
    with pytest.raises(ValueError, match="positive integer"):
        rag.max_retrieval_steps = True

    rag.max_retrieval_steps = 3
    assert rag.max_retrieval_steps == 3


def test_chat_model_remains_the_fourth_positional_constructor_argument(mocker):
    """Adding compression configuration does not break existing callers."""
    foundation_model = mocker.MagicMock()
    foundation_model.system_message_text = "Answer from context."
    foundation_model.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation_model.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    chat_model = mocker.MagicMock()

    rag_template = AgenticRAG(foundation_model, retriever, 2, chat_model)

    assert rag_template.chat_model is chat_model
    assert rag_template.enable_context_compression is True


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


def test_context_overflow_retries_with_compressed_retriever_tool_output(rag, mocker, caplog):
    """A context-window 400 replays the agent with excerpt-only tool responses."""
    initial = AI4RAGChunk(text="initial evidence", metadata={})
    initial_excerpt = AI4RAGChunk(text="initial excerpt", metadata={})
    large_result = AI4RAGChunk(text="very long retrieved evidence", metadata={})
    additional_result = AI4RAGChunk(text="additional retrieved evidence", metadata={})
    excerpt = AI4RAGChunk(text="relevant excerpt", metadata={})
    rag.retriever.retrieve.side_effect = [[initial], [large_result], [large_result, additional_result]]
    compress = mocker.patch(
        "ai4rag.rag.template.agentic_rag_template.LLMDocumentCompressor.compress",
        side_effect=[[initial_excerpt], [initial_excerpt, excerpt]],
    )
    create_agent = rag._create_agent_mock
    context_error = BadRequestError(
        (
            "This model's maximum context length is 32000 tokens. However, you requested 1024 output tokens "
            "and your prompt contains at least 30977 input tokens, for a total of at least 32001 tokens. "
            "Please reduce the length of the input prompt or the number of requested output tokens. "
            "(parameter=input_tokens, value=30977)"
        ),
        response=httpx.Response(400, request=httpx.Request("POST", "https://maas.example/v1/chat/completions")),
        body=None,
    )

    def run_graph(*_args, **kwargs):
        context = kwargs["context"]
        if not context.use_compressed_retriever_tool:
            _call_tool(create_agent, "retriever", {"query": "follow-up"}, context)
            raise context_error
        assert _call_tool(create_agent, "retriever", {"query": "follow-up"}, context) == "relevant excerpt"
        return {"messages": [AIMessage(content="grounded answer")]}

    create_agent.return_value.invoke.side_effect = run_graph

    result = rag.generate("question")

    assert result["answer"] == "grounded answer"
    assert result["reference_documents"] == [initial_excerpt, excerpt]
    assert result["conversation"]["context_compression_fallback"] is True
    assert result["conversation"]["compression"] == {
        "initial": {
            "input_chunks": 1,
            "retained_chunks": 1,
            "dropped_chunks": 0,
            "original_tokens": 4,
            "compressed_tokens": 4,
            "max_completion_tokens": 512,
        },
        "retriever_tool_calls": [
            {
                "input_chunks": 2,
                "retained_chunks": 2,
                "dropped_chunks": 0,
                "original_tokens": 15,
                "compressed_tokens": 8,
                "max_completion_tokens": 512,
            }
        ],
    }
    assert result["conversation"]["tool_execution"]["retriever"][0]["compressed"] is True
    assert not _failure_traces(caplog)
    assert compress.call_args_list == [
        mocker.call("question", [initial], config={"callbacks": [mocker.ANY]}),
        mocker.call("follow-up", [large_result, additional_result]),
    ]


def test_context_overflow_is_propagated_when_compression_is_disabled(rag, mocker):
    """Disabling context compression prevents the overflow retry."""
    rag.enable_context_compression = False
    context_error = BadRequestError(
        "This model's maximum context length is 32000 tokens.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://maas.example/v1/chat/completions")),
        body=None,
    )
    rag._create_agent_mock.return_value.invoke.side_effect = context_error
    compress = mocker.patch("ai4rag.rag.template.agentic_rag_template.LLMDocumentCompressor.compress")

    with pytest.raises(BadRequestError):
        rag.generate("question")

    compress.assert_not_called()


def test_context_compressor_extracts_excerpts_with_the_agent_model(rag):
    """The local extractor preserves metadata and drops chunks without evidence."""
    relevant_text = "relevant excerpt " * 20
    relevant = AI4RAGChunk(text=f"start {relevant_text}end", metadata={"source": "relevant"})
    irrelevant = AI4RAGChunk(text="unrelated document", metadata={"source": "irrelevant"})
    rag.chat_model.invoke.side_effect = [
        AIMessage(content=json.dumps({"excerpt": relevant_text})),
        AIMessage(content='{"excerpt":""}'),
    ]

    excerpts = LLMDocumentCompressor(rag.chat_model).compress("question", [relevant, irrelevant])

    assert excerpts == [AI4RAGChunk(text=relevant_text, metadata={"source": "relevant"})]
    assert rag.chat_model.invoke.call_count == 2
    assert (
        rag.chat_model.invoke.call_args_list[0].args[0][1].content
        == f"User question:\nquestion\n\nSource document:\nstart {relevant_text}end"
    )
    assert rag.chat_model.invoke.call_args_list[0].kwargs == {"config": None, "max_completion_tokens": 512}


def test_context_compressor_accepts_short_verbatim_excerpt(rag):
    """A concise exact answer is useful evidence and must not fall back to the full chunk."""
    excerpt = "Meyerson (1970–1981)"
    source = AI4RAGChunk(text=f"Presidents included {excerpt} and others.", metadata={})
    rag.chat_model.invoke.return_value = AIMessage(content=json.dumps({"excerpt": excerpt}))

    assert LLMDocumentCompressor(rag.chat_model).compress("question", [source]) == [
        AI4RAGChunk(text=excerpt, metadata={})
    ]


def test_context_compressor_retains_source_for_invalid_model_output(rag):
    """Invalid model output preserves source evidence instead of dropping it."""
    source = AI4RAGChunk(text="source " * 300, metadata={"source": "source"})
    short_source = AI4RAGChunk(text="short source", metadata={"source": "short"})
    rag.chat_model.invoke.side_effect = [
        AIMessage(content="excerpt " * 300),
        AIMessage(content="this response is longer than the short source"),
    ]

    excerpts = LLMDocumentCompressor(rag.chat_model).compress("question", [source, short_source])

    assert excerpts == [source, short_source]


def test_context_compressor_retains_source_for_non_verbatim_quote(rag):
    """A quote that differs from the source is never returned as model text."""
    source = AI4RAGChunk(text="source " * 300, metadata={})
    rag.chat_model.invoke.return_value = AIMessage(content='{"excerpt":"invented text"}')

    assert LLMDocumentCompressor(rag.chat_model).compress("question", [source]) == [source]


def test_context_compressor_retains_source_for_legacy_span_response(rag):
    """A response in the old span format remains source-grounded."""
    source = AI4RAGChunk(text="source " * 600, metadata={})
    rag.chat_model.invoke.return_value = AIMessage(
        content='{"spans":[{"start_char":0,"end_char":256},{"start_char":256,"end_char":512}]}'
    )

    assert LLMDocumentCompressor(rag.chat_model).compress("question", [source]) == [source]


def test_context_compressor_caps_total_excerpt_tokens(rag):
    """Compressed tool output remains bounded across multiple chunks."""
    chunks = [AI4RAGChunk(text=character * 2000, metadata={}) for character in "abcdefghij"]
    rag.chat_model.invoke.side_effect = [AIMessage(content=json.dumps({"excerpt": chunk.text})) for chunk in chunks]

    excerpts = LLMDocumentCompressor(rag.chat_model).compress("question", chunks)

    assert excerpts == chunks[:8]


def test_context_compressor_stops_at_first_ranked_chunk_that_exceeds_budget(rag):
    """A lower-ranked small excerpt must not displace an earlier chunk that did not fit."""
    chunks = [AI4RAGChunk(text=character * 2000, metadata={}) for character in "abcdefgh"]
    oversized = AI4RAGChunk(text="e" * 1000, metadata={})
    lower_ranked = AI4RAGChunk(text="f" * 256, metadata={})
    rag.chat_model.invoke.side_effect = [
        *(AIMessage(content=json.dumps({"excerpt": chunk.text})) for chunk in chunks),
        AIMessage(content=json.dumps({"excerpt": oversized.text})),
    ]

    excerpts = LLMDocumentCompressor(rag.chat_model).compress("question", [*chunks, oversized, lower_ranked])

    assert excerpts == chunks
    assert rag.chat_model.invoke.call_count == 9


def test_non_context_bad_request_is_not_retried(rag, mocker):
    """Other 400 responses retain their original failure semantics."""
    error = BadRequestError(
        "invalid request parameter",
        response=httpx.Response(400, request=httpx.Request("POST", "https://maas.example/v1/chat/completions")),
        body=None,
    )
    rag._create_agent_mock.return_value.invoke.side_effect = error

    with pytest.raises(BadRequestError, match="invalid request parameter"):
        rag.generate("question")

    assert rag._create_agent_mock.return_value.invoke.call_count == 1


def test_token_parameter_error_is_not_treated_as_context_overflow(rag, mocker):
    """A token-related request validation error must not trigger compression."""
    error = BadRequestError(
        "Unsupported max_completion_tokens parameter; use max_tokens instead.",
        response=httpx.Response(400, request=httpx.Request("POST", "https://maas.example/v1/chat/completions")),
        body=None,
    )
    rag._create_agent_mock.return_value.invoke.side_effect = error
    compress = mocker.patch("ai4rag.rag.template.agentic_rag_template.LLMDocumentCompressor.compress")

    with pytest.raises(BadRequestError, match="max_completion_tokens"):
        rag.generate("question")

    compress.assert_not_called()
    assert rag._create_agent_mock.return_value.invoke.call_count == 1


@pytest.mark.parametrize("failure_location", ["model", "tool", "recursion", "rewrite"])
def test_failed_graph_logs_safe_model_and_tool_events(mocker, caplog, failure_location):
    """Real graph failures log call order and observations without content."""
    from langchain.agents.middleware import wrap_model_call
    from langgraph.errors import GraphRecursionError

    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    chunk = AI4RAGChunk(text="initial evidence", metadata={"document_id": "source"})
    retriever.retrieve.return_value = [chunk]
    tool_name = "rewrite_query" if failure_location == "rewrite" else "retriever"
    arguments = {"missing_information": "missing fact"} if failure_location == "rewrite" else {"query": "missing fact"}
    model = ToolCallingFake(
        responses=[
            AIMessage(content="", tool_calls=[{"name": tool_name, "args": arguments, "id": f"c{i}"}]) for i in range(20)
        ]
    )
    if failure_location == "recursion":

        @wrap_model_call
        def without_budget(request, handler):
            return handler(request)

        # Simulate the original failure even though normal runs now finalize.
        mocker.patch("ai4rag.rag.template.agentic_rag_template._enforce_agent_budget", without_budget)
        expected_error = GraphRecursionError
    elif failure_location == "tool":
        retriever.retrieve.side_effect = [[chunk], RuntimeError("retrieval failed")]
        expected_error = RuntimeError
    else:
        generate = model._generate
        calls = []

        def fail_after_tool(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2:
                raise ConnectionError("model unavailable")
            return generate(*args, **kwargs)

        mocker.patch.object(model, "_generate", side_effect=fail_after_tool)
        expected_error = ConnectionError
    agent = AgenticRAG(foundation, retriever, chat_model=model)

    with pytest.raises(expected_error) as error:
        agent.generate("original question")

    assert not hasattr(error.value, "conversation")
    traces = _failure_traces(caplog)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["initial_document_count"] == 1
    events = trace["events"]
    assert any(event["event"] == "tool_start" and event["tool"] == tool_name for event in events)
    if failure_location == "recursion":
        assert trace["error_type"] == "GraphRecursionError"
        assert any(event.get("outcome") == "retrieval_limit_reached" for event in events)
    elif failure_location == "rewrite":
        assert any(event["event"] == "tool_error" for event in events)
        assert any(event["event"] == "model_error" for event in events)
        assert trace["model_calls"] == 2
    else:
        assert any(event["event"] == f"{failure_location}_error" for event in events)
    serialized = json.dumps(trace)
    assert all(value not in serialized for value in ("original question", "initial evidence", "missing fact", "source"))


@pytest.mark.parametrize("failure_location", ["initial", "retriever"])
def test_compression_model_failures_are_included_in_agent_trace(mocker, caplog, failure_location):
    """Compression model callbacks share the agent trace without duplicate events."""
    foundation = mocker.MagicMock()
    foundation.system_message_text = "Answer from context."
    foundation.user_message_text = "Question: {question}\nContext: {reference_documents}"
    foundation.context_template_text = "{document}"
    retriever = mocker.MagicMock()
    chunk = AI4RAGChunk(text="private evidence", metadata={"source": "private source"})
    retriever.retrieve.return_value = [chunk]
    model = ToolCallingFake(
        responses=[
            AIMessage(content=json.dumps({"excerpt": chunk.text})),
            AIMessage(
                content="", tool_calls=[{"name": "retriever", "args": {"query": "private query"}, "id": "search"}]
            ),
        ]
    )
    overflow = BadRequestError(
        "maximum context length exceeded",
        response=httpx.Response(400, request=httpx.Request("POST", "https://maas.example/v1/chat/completions")),
        body=None,
    )
    compression_error = RuntimeError("private compression error")
    generate = model._generate
    calls = []

    def fail_compression(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise overflow
        if len(calls) == (2 if failure_location == "initial" else 4):
            raise compression_error
        return generate(*args, **kwargs)

    mocker.patch.object(model, "_generate", side_effect=fail_compression)
    agent = AgenticRAG(foundation, retriever, chat_model=model)

    with pytest.raises(RuntimeError) as error:
        agent.generate("private question")

    assert error.value is compression_error
    traces = _failure_traces(caplog)
    assert len(traces) == 1
    trace = traces[0]
    assert trace["context_compression_retry"] is True
    assert trace["model_calls"] == len(calls) == (2 if failure_location == "initial" else 4)
    events = [event["event"] for event in trace["events"]]
    assert events.count("model_error") == 2
    assert events.count("model_start") == len(calls)
    assert events.count("tool_error") == int(failure_location == "retriever")
    assert trace["error_type"] == "RuntimeError"
    assert "private" not in json.dumps(trace)


def test_failed_compression_retry_logs_once_and_preserves_error(rag, mocker, caplog):
    """Failure while preparing a retry still emits safe diagnostics."""
    overflow = BadRequestError(
        "maximum context length exceeded",
        response=httpx.Response(400, request=httpx.Request("POST", "https://maas.example/v1/chat/completions")),
        body=None,
    )
    rag._create_agent_mock.return_value.invoke.side_effect = overflow
    mocker.patch.object(LLMDocumentCompressor, "compress", side_effect=RuntimeError("compression failed"))

    with pytest.raises(RuntimeError, match="compression failed") as error:
        rag.generate("original question")

    assert not hasattr(error.value, "conversation")
    traces = _failure_traces(caplog)
    assert len(traces) == 1
    assert traces[0]["context_compression_retry"] is True
    assert traces[0]["error_type"] == "RuntimeError"


@pytest.mark.parametrize(
    "result_kind,expected_retrieved,expected_new", [("empty", 0, 0), ("duplicate", 1, 0), ("mixed", 3, 2)]
)
def test_failure_log_retains_retrieval_chunk_counts(rag, caplog, result_kind, expected_retrieved, expected_new):
    """Counts survive a later agent failure, including empty and repeated hits."""
    initial = AI4RAGChunk(text="private initial evidence", metadata={"source": "private"})
    new_chunks = [AI4RAGChunk(text=f"private new evidence {index}", metadata={}) for index in range(2)]
    hits = {"empty": [], "duplicate": [initial], "mixed": [initial, *new_chunks]}[result_kind]
    rag.retriever.retrieve.side_effect = [[initial], hits]

    def fail_after_retrieval(*_args, **kwargs):
        _call_tool(rag._create_agent_mock, "retriever", {"query": "private query"}, kwargs["context"])
        raise RuntimeError("private model error")

    rag._create_agent_mock.return_value.invoke.side_effect = fail_after_retrieval
    with pytest.raises(RuntimeError):
        rag.generate("private question")

    trace = _failure_traces(caplog)[0]
    assert trace["initial_document_count"] == 1
    assert trace["new_chunk_count"] == expected_new
    assert trace["retrieval_results"] == [
        {
            "retrieval_index": 0,
            "retrieved_chunk_count": expected_retrieved,
            "retained_chunk_count": expected_retrieved,
            "new_chunk_count": expected_new,
        }
    ]
    assert "private" not in "\n".join(
        record.message for record in caplog.records if record.message.startswith("RAG_FAILURE")
    )


def _failure_traces(caplog):
    return [
        json.loads(record.getMessage().split(" ", 1)[1])
        for record in caplog.records
        if record.getMessage().startswith("RAG_FAILURE_TRACE ")
    ]


def test_all_failed_questions_log_independently_before_pattern_failure(rag, caplog):
    """The unchanged executor.map path cannot suppress diagnostics from later workers."""
    from threading import Barrier

    from ai4rag.core.experiment.exception_handler import GenerationError
    from ai4rag.core.experiment.utils import query_rag

    rag.retriever.retrieve.return_value = [AI4RAGChunk(text="private evidence", metadata={})]
    workers_started = Barrier(3, timeout=10)

    def fail_after_retrieval(*_args, **kwargs):
        workers_started.wait()
        _call_tool(rag._create_agent_mock, "retriever", {"query": "private query"}, kwargs["context"])
        raise RuntimeError("private failure")

    rag._create_agent_mock.return_value.invoke.side_effect = fail_after_retrieval
    with pytest.raises(GenerationError):
        query_rag(rag, ["private Q0", "private Q1", "private Q2"], max_threads=3)

    traces = _failure_traces(caplog)
    assert len(traces) == 3
    assert len({trace["diagnostic_id"] for trace in traces}) == 3
    assert all(trace["initial_document_count"] == 1 and trace["new_chunk_count"] == 0 for trace in traces)
    assert all(len(trace["retrieval_results"]) == 1 for trace in traces)
    assert all("private" not in json.dumps(trace) for trace in traces)
