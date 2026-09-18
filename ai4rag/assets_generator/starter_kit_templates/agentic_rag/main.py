# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------

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
from agentic_rag.agent import create_rag  # noqa: E402
from agentic_rag.tracing import enable_tracing  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

logger = logging.getLogger(__name__)

_MAX_INVOKE_ATTEMPTS = 3
_GRACEFUL_ERROR_MESSAGE = "I was unable to process this request due to repeated internal errors."
_RETRYABLE_EXCEPTIONS = (
    openai.InternalServerError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.RateLimitError,
)


class ChatMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant", "system"])
    content: str


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1)
    model: str | None = None
    stream: bool = False


class ChoiceMessage(BaseModel):
    role: str = "assistant"
    content: str


class Choice(BaseModel):
    index: int
    message: ChoiceMessage
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: dict | None = None


class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool


rag = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global rag
    enable_tracing()
    rag = create_rag()
    app.state.rag = rag
    yield
    rag = None
    app.state.rag = None


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


def _make_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:12]}"


async def _invoke_with_retry(messages: list[dict[str, str]]) -> dict[str, Any]:
    if rag is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    last_exception: Exception = RuntimeError("no invocation attempts were made")
    for attempt in range(1, _MAX_INVOKE_ATTEMPTS + 1):
        try:
            result = await asyncio.to_thread(rag.chat, messages)
            choice = result[0]
            return {
                "content": choice.message.content or "",
                "context": [
                    {"role": "assistant", "content": choice.message.content or ""},
                ],
            }
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < _MAX_INVOKE_ATTEMPTS:
                logger.warning(
                    "RAG invocation failed (attempt %d/%d): %s", attempt, _MAX_INVOKE_ATTEMPTS, type(exc).__name__
                )
                await asyncio.sleep(0.5 * attempt)

    raise last_exception


@app.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
    summary="Create chat completion",
    tags=["Chat"],
)
async def chat_completions(request: ChatCompletionRequest):
    messages = [message.model_dump() for message in request.messages]
    model_id = request.model or getenv("MODEL_ID") or "model"

    try:
        result = await _invoke_with_retry(messages)
    except _RETRYABLE_EXCEPTIONS as exc:
        logger.exception("RAG invocation failed", exc_info=exc)
        result = {"content": _GRACEFUL_ERROR_MESSAGE, "context": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing request: {exc}") from exc

    completion_id = _make_completion_id()
    if request.stream:
        return _stream_response(completion_id, model_id, result["content"])

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["content"]},
                "finish_reason": "stop",
            }
        ],
        "context": result["context"],
        "usage": None,
    }


def _stream_response(completion_id: str, model_id: str, content: str) -> StreamingResponse:
    created = int(time.time())

    async def event_generator() -> AsyncIterator[str]:
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield f"data: {json.dumps({'id': completion_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model_id, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["Health"])
async def health():
    initialized = rag is not None
    body = {"status": "healthy" if initialized else "not_ready", "agent_initialized": initialized}
    if not initialized:
        return JSONResponse(status_code=503, content=body)
    return body


_SANDBOX_MODE = bool(getenv("K8S_REVIEWER_TOKEN", "").strip())
if _SANDBOX_MODE:
    from playground_sandbox import router as sandbox_router

    app.include_router(sandbox_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(getenv("PORT", 8000)))
