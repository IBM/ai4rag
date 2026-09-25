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

from agentic_rag.sqlite_shim import patch_sqlite3

patch_sqlite3()

import openai  # noqa: E402
from agentic_rag.agent import create_rag  # noqa: E402
from agentic_rag.config import AgentConfig  # noqa: E402
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


class ResponseInputMessage(BaseModel):
    role: str = Field(..., examples=["user", "assistant", "system"])
    content: str


class ResponsesRequest(BaseModel):
    input: str | list[ResponseInputMessage]
    model: str | None = None
    instructions: str | None = None
    stream: bool = False


class ResponseOutputText(BaseModel):
    type: str = "output_text"
    text: str


class ResponseMessage(BaseModel):
    id: str
    type: str = "message"
    status: str = "completed"
    role: str = "assistant"
    content: list[ResponseOutputText]


class ResponsesResponse(BaseModel):
    id: str
    object: str = "response"
    created_at: int
    model: str
    status: str = "completed"
    output: list[ResponseMessage]
    output_text: str
    usage: dict | None = None


class HealthResponse(BaseModel):
    status: str
    agent_initialized: bool


rag = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global rag
    rag = create_rag()
    app.state.rag = rag
    yield
    rag = None
    app.state.rag = None


app = FastAPI(
    title="Agentic RAG API",
    description=("FastAPI service for Agentic RAG Agent with OpenAI-compatible Responses API."),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "Health", "description": "Service health monitoring"},
        {"name": "Responses", "description": "Responses API operations"},
    ],
)


def _make_response_id() -> str:
    return f"resp-{uuid.uuid4().hex[:12]}"


async def _invoke_with_retry(messages: list[dict[str, str]]) -> dict[str, Any]:
    if rag is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    last_exception: Exception = RuntimeError("no invocation attempts were made")
    for attempt in range(1, _MAX_INVOKE_ATTEMPTS + 1):
        try:
            result = await asyncio.to_thread(rag.respond, messages)
            return {"content": result.output_text or "", "context": []}
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exception = exc
            if attempt < _MAX_INVOKE_ATTEMPTS:
                logger.warning(
                    "RAG invocation failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_INVOKE_ATTEMPTS,
                    type(exc).__name__,
                )
                await asyncio.sleep(0.5 * attempt)

    raise last_exception


@app.post(
    "/v1/responses",
    response_model=ResponsesResponse,
    summary="Create a model response",
    tags=["Responses"],
)
async def responses(request: ResponsesRequest):
    if isinstance(request.input, str):
        messages = [{"role": "user", "content": request.input}]
    else:
        messages = [message.model_dump() for message in request.input]
    if request.instructions:
        messages.insert(0, {"role": "system", "content": request.instructions})
    model_id = request.model or getenv("MODEL_ID") or "model"

    try:
        result = await _invoke_with_retry(messages)
    except _RETRYABLE_EXCEPTIONS as exc:
        logger.exception("RAG invocation failed", exc_info=exc)
        result = {"content": _GRACEFUL_ERROR_MESSAGE, "context": []}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error processing request: {exc}") from exc

    response_id = _make_response_id()
    if request.stream:
        return _stream_response(response_id, model_id, result["content"])

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model_id,
        "status": "completed",
        "output": [
            {
                "id": f"msg-{uuid.uuid4().hex[:12]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": result["content"]}],
            }
        ],
        "output_text": result["content"],
        "context": result["context"],
        "usage": None,
    }


def _stream_response(response_id: str, model_id: str, content: str) -> StreamingResponse:

    async def event_generator() -> AsyncIterator[str]:
        data = {
            "type": "response.output_text.delta",
            "response_id": response_id,
            "delta": content,
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield f"data: {json.dumps({'type': 'response.completed', 'response': {'id': response_id, 'object': 'response', 'model': model_id, 'status': 'completed'}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health", response_model=HealthResponse, summary="Health check", tags=["Health"])
async def health():
    initialized = rag is not None
    body = {
        "status": "healthy" if initialized else "not_ready",
        "agent_initialized": initialized,
    }
    if not initialized:
        return JSONResponse(status_code=503, content=body)
    return body


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=AgentConfig.from_env().port)
