"""
Phase 8 HTTP server 测试。

覆盖范围：
  - GET  /v1/models
  - POST /v1/chat/completions（non-streaming）
  - POST /v1/chat/completions（streaming SSE）

使用 httpx.AsyncClient + ASGITransport 直接调用 ASGI app（无需启动真实服务器）。
引擎使用 dry_run=True，无需真实模型权重。
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
import httpx
from asgi_lifespan import LifespanManager

from mini_infer.config import EngineConfig
from mini_infer.server import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """启动 dry_run AsyncEngine，通过 ASGI transport 直接访问 app。"""
    config = EngineConfig(
        model_name="dry",
        dry_run=True,
        num_gpu_blocks=32,
        block_size=16,
        max_batch_size=4,
    )
    app.state.engine_config = config  # type: ignore[attr-defined]
    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app), base_url="http://test"
        ) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models(client: httpx.AsyncClient):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    assert "id" in body["data"][0]


@pytest.mark.asyncio
async def test_chat_completion_non_stream(client: httpx.AsyncClient):
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert len(choice["message"]["content"]) > 0


@pytest.mark.asyncio
async def test_chat_completion_stream(client: httpx.AsyncClient):
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    lines = resp.text.strip().split("\n")
    data_lines = [l[6:] for l in lines if l.startswith("data: ")]

    # 最后一行是 [DONE]
    assert data_lines[-1] == "[DONE]"

    # 解析所有 JSON chunk（排除 [DONE]）
    chunks = [json.loads(d) for d in data_lines[:-1]]
    assert len(chunks) >= 3  # 至少 role chunk + 1 token chunk + stop chunk

    # 第一个 chunk 包含 role
    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # 最后一个 JSON chunk 的 finish_reason == "stop"
    last_chunk = chunks[-1]
    assert last_chunk["choices"][0]["finish_reason"] == "stop"

    # 中间 chunk 有 content
    content_chunks = [c for c in chunks[1:-1] if c["choices"][0]["delta"].get("content")]
    assert len(content_chunks) > 0


@pytest.mark.asyncio
async def test_stream_has_valid_ids(client: httpx.AsyncClient):
    """同一次请求的所有 chunk 应共享同一个 completion id。"""
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 3,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    data_lines = [l[6:] for l in resp.text.strip().split("\n") if l.startswith("data: ")]
    chunks = [json.loads(d) for d in data_lines if d != "[DONE]"]
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1  # 所有 chunk 共享同一 id
