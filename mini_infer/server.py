"""
Phase 8 OpenAI-compatible HTTP server。

提供与 OpenAI Chat Completions API 兼容的接口：
  GET  /v1/models
  POST /v1/chat/completions（streaming + non-streaming）

使用方式：
  python serve.py --model /path/to/model

或直接通过 uvicorn：
  uvicorn mini_infer.server:app --host 0.0.0.0 --port 8000

启动时全局初始化 AsyncEngine，所有请求共享同一 step loop，
实现 continuous batching（多并发 HTTP 请求被合并进同一 decode_batch）。
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .async_engine import AsyncEngine
from .config import EngineConfig
from .openai_schema import (
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaMessage,
    ModelCard,
    ModelList,
    Usage,
)

# ---------------------------------------------------------------------------
# 全局 engine 实例（由 lifespan 初始化）
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_model_id: str = "mini-infer"


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return _engine


# ---------------------------------------------------------------------------
# Lifespan：启动/关闭 AsyncEngine
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _engine
    config: EngineConfig = app.state.engine_config  # type: ignore[attr-defined]
    _engine = AsyncEngine(config)
    await _engine.start()
    yield
    await _engine.stop()
    _engine = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="mini-infer", lifespan=lifespan)


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=_model_id, created=0)])


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):  # type: ignore[return]
    engine = get_engine()
    prompt = engine.format_prompt(request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_generator(engine, prompt, request),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream(engine, prompt, request)


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


async def _non_stream(
    engine: AsyncEngine,
    prompt: str,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    text = await engine.generate(prompt, max_new_tokens=request.max_tokens)
    completion_id = f"chatcmpl-{uuid4().hex[:8]}"
    tokenizer = engine.tokenizer
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return ChatCompletionResponse(
        id=completion_id,
        model=_model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming response（SSE）
# ---------------------------------------------------------------------------


async def _stream_generator(
    engine: AsyncEngine,
    prompt: str,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid4().hex[:8]}"
    created = int(time.time())

    # 第一个 chunk：role only
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=_model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {first_chunk.json()}\n\n"

    # 逐 token chunk
    async for token in engine.generate_stream(prompt, max_new_tokens=request.max_tokens):
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=_model_id,
            choices=[
                ChatCompletionChunkChoice(
                    index=0,
                    delta=DeltaMessage(content=token),
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {chunk.json()}\n\n"

    # 结束 chunk
    stop_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=_model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )
        ],
    )
    yield f"data: {stop_chunk.json()}\n\n"
    yield "data: [DONE]\n\n"
