"""tests/test_async_engine.py — AsyncEngine 关键边界测试。"""

from __future__ import annotations

import asyncio

import pytest

from mini_infer.async_engine import AsyncEngine
from mini_infer.config import EngineConfig


@pytest.mark.asyncio
async def test_generate_with_reason_finishes_on_eos_without_visible_delta() -> None:
    """
    即使最终 token 在 decode(skip_special_tokens=True) 后没有可见文本，
    AsyncEngine 也必须投递 DONE 事件，而不是让调用方永久等待。
    """
    engine = AsyncEngine(
        EngineConfig(
            model_name="stub",
            dry_run=True,
            block_size=4,
            num_gpu_blocks=32,
            max_batch_size=4,
        )
    )
    await engine.start()
    try:
        # 模拟 EOS token 对外不可见：decode 总是返回空串。
        engine._engine.model_runner.tokenizer.decode = lambda ids, skip_special_tokens=True: ""

        def fake_prefill(states):
            for state in states:
                state.append_generated(999, "")
                state.prefilled = True
                state.mark_finished("eos")

        engine._engine.model_runner.prefill = fake_prefill

        text, reason = await asyncio.wait_for(
            engine.generate_with_reason("hello", max_new_tokens=4),
            timeout=1.0,
        )
        assert text == ""
        assert reason == "eos"
    finally:
        await engine.stop()
