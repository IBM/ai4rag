import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from os import getenv
from typing import Any

from sqlite_shim import patch_sqlite3

patch_sqlite3()

import openai  # noqa: E402
from agentic_rag.agent import get_graph_closure  # noqa: E402
from agentic_rag.config import get_chat_base_url  # noqa: E402
from agentic_rag.tracing import enable_tracing  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from langchain_core.exceptions import OutputParserException  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402
from langgraph.errors import GraphRecursionError  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

logger = logging.getLogger(__name__)

_MAX_INVOKE_ATTEMPTS = 3
_GRACEFUL_ERROR_MESSAGE = (
    "I was unable to process this request due to repeated internal errors."
)
_RETRYABLE_EXCEPTIONS = (
    ValidationError,
    OutputParserException,
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
)
_GRACEFUL_EXCEPTIONS = _RETRYABLE_EXCEPTIONS + (GraphRecursionError,)


class ChatMessage(BaseModel):
    role: str = Field(
        ...,
        description="The role of the message author.",
        examples=["user", "assistant", "system", "tool"],
    )
    content: str = Field(
        ...,
        description="The contents of the message.",
        examples=["What is LangChain?"],
    )


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
        description="A list of messages comprising the conversation so far.",
    )
    model: str | None = Field(
        None,
        description="ID of the model to use. Defaults to the server's configured MODEL_ID.",
    )
    stream: bool = Field(
        False,
        description="If true, partial message deltas will be sent as SSE events.",
    )


class ChoiceMessage(BaseModel):
    role: str = Field(
        "assistant", description="The role of the author of this message."
    )
    content: str = Field(..., description="The contents of the message.")


class Choice(BaseModel):
    index: int = Field(
        ..., description="The index of the choice in the list of choices."
    )
    message: ChoiceMessage
    finish_reason: str = Field(
        ...,
        description="The reason the model stopped generating tokens.",
        examples=["stop", "tool_calls"],
    )


class ChatCompletionResponse(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the chat completion.",
        examples=["chatcmpl-abc123def456"],
    )
    object: str = Field(
        "chat.completion",
        description="The object type, which is always `chat.completion`.",
    )
    created: int = Field(
        ...,
        description="The Unix timestamp (in seconds) of when the chat completion was created.",
    )
    model: str = Field(..., description="The model used for the chat completion.")
    choices: list[Choice] = Field(..., description="A list of chat completion choices.")
    usage: dict | None = Field(
        None, description="Usage statistics for the completion request."
    )


class HealthResponse(BaseModel):
    status: str = Field(
        ..., description="Current service status.", examples=["healthy"]
    )
    agent_initialized: bool = Field(
        ...,
        description="Whether the agent has been initialized and is ready to serve requests.",
    )


agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global agent_graph
    enable_tracing()

    base_url = get_chat_base_url()
    model_id = getenv("MODEL_ID")

    if base_url and not base_url.endswith("/v1"):
        base_url = base_url.rstrip("/") + "/v1"

    graph_closure = get_graph_closure(
        model_id=model_id,
        base_url=base_url,
    )
    agent_graph = graph_closure()

    app.state.agent_graph = agent_graph

    yield

    agent_graph = None
    app.state.agent_graph = None


app = FastAPI(
    title="Agentic RAG API",
    description=(
        "FastAPI service for Agentic RAG Agent with "
        "OpenAI-compatible chat completions API. "
        "To access the sandbox playground, click "
        "[Sandbox Playground](/playground)."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Service health monitoring"},
        {"name": "Chat", "description": "Chat completion operations"},
    ],
)


def _build_langchain_messages(messages: list[ChatMessage]) -> list[HumanMessage]:
    for msg in reversed(messages):
        if msg.role == "user":
            return [HumanMessage(content=msg.content)]
    raise ValueError("No user message found in messages list")


def _extract_usage(messages: list) -> dict | None:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    found = False
    for message in messages:
        if isinstance(message, AIMessage) and getattr(message, "usage_metadata", None):
            meta = message.usage_metadata
            prompt_tokens += meta.get("input_tokens", 0) or 0
            completion_tokens += meta.get("output_tokens", 0) or 0
            total_tokens += meta.get("total_tokens", 0) or 0
            found = True
    if not found:
        return None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


async def _invoke_with_retry(
    input_data: dict,
    config: dict,
) -> dict:
    global agent_graph
    last_exception: Exception = RuntimeError("no invocation attempts were made")

    for attempt in range(1, _MAX_INVOKE_ATTEMPTS + 1):
        try:
            return await agent_graph.ainvoke(input_data, config=config)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < _MAX_INVOKE_ATTEMPTS:
                logger.warning(
                    "LLM/graph invocation failed (attempt %d/%d): %s. Retrying.",
                    attempt,
                    _MAX_INVOKE_ATTEMPTS,
                    type(exc).__name__,
                )
                await asyncio.sleep(0.5 * attempt)
            else:
                logger.error(
                    "LLM/graph invocation failed after %d attempts: %s",
                    _MAX_INVOKE_ATTEMPTS,
                    type(exc).__name__,
                )

    raise last_exception


@app.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    tags=["Chat"],
)
async def chat_completions(request: ChatCompletionRequest):
    global agent_graph

    if agent_graph is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    langchain_messages = _build_langchain_messages(request.messages)
    model_id = request.model or getenv("MODEL_ID") or "model"

    if request.stream:
        return await _handle_stream(langchain_messages, model_id)
    else:
        return await _handle_chat(langchain_messages, model_id)


async def _handle_chat(messages: list[HumanMessage], model_id: str) -> dict[str, Any]:
    global agent_graph

    try:
        try:
            result = await _invoke_with_retry(
                {"messages": messages}, config={"recursion_limit": 15}
            )
        except _GRACEFUL_EXCEPTIONS:
            return {
                "id": _make_completion_id(),
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": _GRACEFUL_ERROR_MESSAGE,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "context": [],
                "usage": None,
            }

        assistant_content = ""
        context_messages = []

        if "messages" in result and len(result["messages"]) > 0:
            for message in result["messages"]:
                if isinstance(message, HumanMessage):
                    context_messages.append(
                        {"role": "user", "content": message.content}
                    )
                elif isinstance(message, AIMessage):
                    msg_data = {"role": "assistant", "content": message.content or ""}
                    if message.tool_calls:
                        msg_data["tool_calls"] = [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["args"]),
                                },
                            }
                            for tc in message.tool_calls
                        ]
                    context_messages.append(msg_data)
                elif isinstance(message, ToolMessage):
                    context_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": message.tool_call_id,
                            "name": message.name,
                            "content": message.content,
                        }
                    )

            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage) and message.content:
                    assistant_content = message.content
                    break

        return {
            "id": _make_completion_id(),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": assistant_content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "context": context_messages,
            "usage": _extract_usage(result.get("messages", [])),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing request: {str(e)}"
        )


async def _handle_stream(
    messages: list[HumanMessage], model_id: str
) -> StreamingResponse:
    global agent_graph

    completion_id = _make_completion_id()
    created = int(time.time())

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for event in agent_graph.astream_events(
                {"messages": messages},
                config={"recursion_limit": 15},
                version="v2",
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": chunk.content},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                elif kind == "on_chat_model_end":
                    message = event["data"]["output"]
                    if hasattr(message, "tool_calls") and message.tool_calls:
                        tool_calls_delta = [
                            {
                                "index": i,
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["args"]),
                                },
                            }
                            for i, tc in enumerate(message.tool_calls)
                        ]
                        data = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": tool_calls_delta,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(data)}\n\n"

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    if hasattr(output, "content"):
                        output = output.content
                    data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "tool",
                                    "content": str(output),
                                    "name": event.get("name", ""),
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(data)}\n\n"

            final_data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception:
            logger.exception("Error in stream event_generator")
            error_data = {
                "error": {
                    "message": "Internal server error",
                    "type": "server_error",
                }
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get(
    "/health", response_model=HealthResponse, summary="Health check", tags=["Health"]
)
async def health():
    initialized = agent_graph is not None
    body = {
        "status": "healthy" if initialized else "not_ready",
        "agent_initialized": initialized,
    }
    if not initialized:
        return JSONResponse(status_code=503, content=body)
    return body


# ── Playground UI (sandbox mode) ─────────────────────────────────────────────
_SANDBOX_MODE = bool(getenv("K8S_REVIEWER_TOKEN", "").strip())
if _SANDBOX_MODE:
    from playground_sandbox import router as sandbox_router

    app.include_router(sandbox_router)


if __name__ == "__main__":
    import uvicorn

    port = int(getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
