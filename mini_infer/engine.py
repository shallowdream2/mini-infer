"""
Phase 2 推理引擎：实现 continuous batching 调度循环。

与 Phase 1 的核心区别：
  - Phase 1：等待整个 batch 全部跑完再拉新请求（静态 batch）
  - Phase 2：每个 decode step 后检查是否有新请求可以加入，立即接入运行中的 batch
    （continuous batching）。这意味着运行中的不同请求可能处于不同的 decode 步骤。

主循环结构（每次迭代）：
  1. 准入：从 waiting 队列尽量接入新请求（受 max_batch_size 和 KV block 可用数量限制）
  2. Prefill：对新接入的请求做 prefill forward，写 KV 到 block tensor
  3. Batch decode：把所有 running 请求合并成一次 GPU forward（不再是串行 for 循环）
  4. 清理：将已完成请求从 running 中移除，释放 KV block

异常处理：try/finally 确保异常时 KV 块也能被归还，避免 GPU 显存泄漏（Phase 1 缺失这一点）。
"""

import math
from uuid import uuid4

from .config import EngineConfig
from .kv_cache import KVCacheManager
from .model_runner import ModelRunner
from .request import Request, RequestState, SamplingParams
from .scheduler import Scheduler


class LLMEngine:
    """提供 generate 接口，实现 continuous batching 推理调度。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.kv_cache = KVCacheManager(config=config)
        self.scheduler = Scheduler(max_batch_size=config.max_batch_size)
        self.model_runner = ModelRunner(config=config, kv_cache=self.kv_cache)

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        """
        批量生成，返回与输入 prompts 顺序一致的输出文本列表。

        采用 continuous batching：新请求在现有请求 decode 过程中动态加入，
        充分复用每次 batch forward 的 GPU 带宽。
        """
        states: list[RequestState] = []
        for prompt in prompts:
            request = Request(
                request_id=str(uuid4()),
                prompt=prompt,
                sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            )
            state = RequestState(
                request=request,
                prompt_token_ids=self._tokenize(prompt),
            )
            self.scheduler.add_request(state)
            states.append(state)

        outputs: dict[str, str] = {}

        try:
            while self.scheduler.has_waiting() or self.scheduler.num_running() > 0:
                # ── 1. 准入：接入尽可能多的等待请求 ────────────────────────
                newly_admitted: list[RequestState] = []
                while (
                    self.scheduler.has_waiting()
                    and self.scheduler.num_running() < self.config.max_batch_size
                ):
                    next_state = self.scheduler.peek_next_waiting()
                    assert next_state is not None

                    # 保守估计所需 block 数：prompt + 最大输出
                    prompt_len = len(next_state.prompt_token_ids)
                    max_out = next_state.request.sampling_params.max_new_tokens
                    blocks_needed = math.ceil((prompt_len + max_out) / self.config.block_size)

                    if self.kv_cache.num_free_blocks() < blocks_needed:
                        # 显存不足：如果同时没有其他请求在跑，则无法通过等待来释放块
                        # 直接报错，避免无限循环
                        if self.scheduler.num_running() == 0 and not newly_admitted:
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                f"（prompt={prompt_len} token + max_new_tokens={max_out}），"
                                f"但当前仅有 {self.kv_cache.num_free_blocks()} 个空闲块（总计 {self.config.num_gpu_blocks} 块）。"
                                f"请增大 num_gpu_blocks 或减小 max_new_tokens。"
                            )
                        break  # 有其他请求在跑，等待它们释放块

                    state = self.scheduler.pop_next_waiting()
                    self.kv_cache.init_request(state)
                    self.scheduler.add_to_running(state)
                    newly_admitted.append(state)

                # ── 2. Prefill：对新接入的请求做 prefill ─────────────────────
                if newly_admitted:
                    self.model_runner.prefill(newly_admitted)

                # ── 3. Batch decode：一次 forward 处理所有 running 请求 ──────
                running = self.scheduler.get_running_states()
                if running:
                    self.model_runner.decode_batch(running)

                # ── 4. 清理完成的请求 ────────────────────────────────────────
                for state in list(running):
                    if state.finished:
                        outputs[state.request.request_id] = (
                            self.model_runner.tokenizer.decode(
                                state.generated_token_ids, skip_special_tokens=True
                            )
                        )
                        self.kv_cache.free_request(state)
                        self.scheduler.finish_request(state)

        except Exception:
            # 异常时归还所有 running 请求的 KV 块，防止显存泄漏
            for state in self.scheduler.get_running_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            raise

        return [outputs.get(state.request.request_id, "") for state in states]

    def _tokenize(self, text: str) -> list[int]:
        return self.model_runner.tokenizer.encode(text, add_special_tokens=True)
