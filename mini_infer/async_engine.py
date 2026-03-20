"""
Phase 8 AsyncEngine：将同步 LLMEngine 包装为异步接口。

核心设计：
  - 后台线程持续调用 engine.step()，不等待 HTTP 请求凑齐
  - 每个 HTTP 请求通过 add_request() 加入等待队列，通过 asyncio.Queue 接收 token
  - 多个并发 HTTP 请求的 prompt 被 continuous batching 合并到同一 decode_batch

正确架构（解决"每请求独立调用 generate() 无法 continuous batching"的问题）：

  ❌ 错误：HTTP 请求 A → 线程1 → engine.generate(["A"])  ← 独占 GPU
           HTTP 请求 B → 线程2 → engine.generate(["B"])  ← 等待线程1

  ✅ 正确：HTTP 请求 A ─┐
           HTTP 请求 B ─┤─→ asyncio.Queue ─→ 后台 step loop ─→ decode_batch([A, B])
           HTTP 请求 C ─┘                      ↓ token 分发到各请求的 asyncio.Queue

并发设计要点：
  - 后台线程通过 loop.call_soon_threadsafe(queue.put_nowait, token) 跨线程投递 token
  - 前台 async generator 通过 await queue.get() 接收 token
  - 后台线程 sleep(1ms) 避免空转时的 CPU 浪费
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import AsyncGenerator
from uuid import uuid4

from .config import EngineConfig
from .engine import LLMEngine

# 完成哨兵（不使用 None，避免与实际 token 冲突）
_DONE = object()


class AsyncEngine:
    """
    LLMEngine 的异步包装器。

    用法：
        engine = AsyncEngine(config)
        await engine.start()
        async for token in engine.generate_stream("hello"):
            print(token, end="", flush=True)
        await engine.stop()

    或使用 async context manager：
        async with AsyncEngine(config) as engine:
            result = await engine.generate("hello")
    """

    def __init__(self, config: EngineConfig) -> None:
        self._engine = LLMEngine(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        # request_id → asyncio.Queue（由后台线程投递 token，由 async generator 消费）
        self._token_queues: dict[str, asyncio.Queue] = {}

    async def start(self) -> None:
        """启动后台 step loop 线程。"""
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._thread = threading.Thread(target=self._engine_loop, daemon=True, name="mini-infer-step")
        self._thread.start()

    async def stop(self) -> None:
        """停止后台线程，等待其退出。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    async def __aenter__(self) -> "AsyncEngine":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # 公开生成接口
    # ------------------------------------------------------------------

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
    ) -> AsyncGenerator[str, None]:
        """逐 token 异步 yield，直到生成完毕。

        关键顺序：先生成 rid、注册 queue，再调用 add_request()。
        确保后台线程在 step() 中产出 token 并调用 _put() 时，queue 已存在。
        """
        rid = str(uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        self._token_queues[rid] = queue  # 注册在 add_request 之前
        try:
            self._engine.add_request(prompt, max_new_tokens, priority, request_id=rid)
            while True:
                token = await asyncio.wait_for(queue.get(), timeout=60.0)
                if token is _DONE:
                    return
                yield token  # type: ignore[misc]
        finally:
            self._token_queues.pop(rid, None)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
    ) -> str:
        """非流式版本：等待所有 token 后一次返回完整文本。"""
        parts: list[str] = []
        async for token in self.generate_stream(prompt, max_new_tokens, priority):
            parts.append(token)
        return "".join(parts)

    @property
    def tokenizer(self):
        """暴露底层 tokenizer，供 server.py 计算 token 数量。"""
        return self._engine.model_runner.tokenizer

    def format_prompt(self, messages: list) -> str:
        """
        将 ChatMessage 列表转换为模型输入字符串。

        优先使用 tokenizer.apply_chat_template（Qwen 等模型有正式格式）；
        dry_run / stub tokenizer 时退化为简单拼接。
        """
        tokenizer = self._engine.model_runner.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback：简单格式，仅用于 dry_run 测试
        parts = [f"{m.role}: {m.content}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 后台 step loop（在独立线程中运行）
    # ------------------------------------------------------------------

    def _engine_loop(self) -> None:
        while self._running:
            if self._engine.has_unfinished_requests():
                try:
                    new_tokens = self._engine.step()
                    for rid, tokens in new_tokens.items():
                        for text in tokens:
                            self._put(rid, text)
                        # 如果请求在本步完成，投递 DONE 哨兵，并清理追踪表
                        if self._engine.is_finished(rid):
                            self._put(rid, _DONE)
                            self._engine.cleanup_request(rid)
                except Exception:
                    import traceback
                    traceback.print_exc()
                    # 错误时通知所有等待中的消费者退出
                    for rid in list(self._token_queues):
                        self._put(rid, _DONE)
                    break
            else:
                time.sleep(0.001)  # 无请求时短暂休眠，避免空转

    def _put(self, rid: str, item: object) -> None:
        """线程安全地向 asyncio.Queue 投递 item。"""
        queue = self._token_queues.get(rid)
        if queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, item)
