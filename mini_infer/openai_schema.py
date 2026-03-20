"""
Phase 8 OpenAI-compatible API schema。

定义与 OpenAI Chat Completions API 兼容的请求/响应 Pydantic 模型，
供 server.py 的路由层使用。

覆盖接口：
  POST /v1/chat/completions（streaming + non-streaming）
  GET  /v1/models
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    # 以下字段接受但不处理（SDK 兼容性）
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] | str | None = None


# ---------------------------------------------------------------------------
# 非流式响应模型
# ---------------------------------------------------------------------------


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# 流式响应模型（SSE chunk）
# ---------------------------------------------------------------------------


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---------------------------------------------------------------------------
# /v1/models 端点模型
# ---------------------------------------------------------------------------


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
