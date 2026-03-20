"""
Phase 7 推理引擎：在 Phase 2 continuous batching 基础上，新增 Preemption + Priority Scheduling。

Phase 7 新增机制：
  - generate() 接受 priorities 参数，每个请求可设置独立优先级（数值越小优先级越高）
  - 准入循环：KV 空间不足时，先尝试换出 running 中优先级最低的请求（swap_out），
    而不是直接报错——只有真的无法腾出空间（无可换出对象且 running 为空）才抛 RuntimeError
  - swap_in 恢复：每步末尾（清理完成请求后）尝试将 swapped 请求换回 GPU，
    有足够空闲块时加回 running，下一步参与 decode
  - 主循环条件扩展：has_swapped() 时继续循环，直到所有换出请求都完成

主循环结构（每次迭代）：
  1. 准入：接入等待请求，块不足时尝试抢占低优先级 running 请求
  2. Prefill：对新接入的请求做 prefill
  3. Batch decode：一次 forward 处理所有 running 请求
  4. 清理：移除已完成请求，释放 KV 块
  5. Swap_in：有空闲块时将 swapped 请求换回，加入 running

异常处理：运行请求的 GPU 块正常归还；换出请求的 CPU KV 清除以释放内存。
"""

import math
from uuid import uuid4

from .config import EngineConfig
from .kv_cache import KVCacheManager
from .model_runner import ModelRunner
from .request import Request, RequestState, SamplingParams
from .scheduler import Scheduler


class LLMEngine:
    """提供 generate 接口，实现 continuous batching + preemption 推理调度。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.kv_cache = KVCacheManager(config=config)
        self.scheduler = Scheduler(max_batch_size=config.max_batch_size)
        self.model_runner = ModelRunner(config=config, kv_cache=self.kv_cache)

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        priorities: list[int] | None = None,
    ) -> list[str]:
        """
        批量生成，返回与输入 prompts 顺序一致的输出文本列表。

        priorities: 每个 prompt 的调度优先级（0 = 最高，数值越大优先级越低）。
                    若为 None，所有请求优先级为 0（与 Phase 2 行为一致）。
        """
        if priorities is not None and len(priorities) != len(prompts):
            raise ValueError(f"priorities 长度 {len(priorities)} 与 prompts 长度 {len(prompts)} 不一致")

        states: list[RequestState] = []
        for i, prompt in enumerate(prompts):
            priority = priorities[i] if priorities is not None else 0
            request = Request(
                request_id=str(uuid4()),
                prompt=prompt,
                sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
                priority=priority,
            )
            state = RequestState(
                request=request,
                prompt_token_ids=self._tokenize(prompt),
            )
            self.scheduler.add_request(state)
            states.append(state)

        outputs: dict[str, str] = {}

        try:
            while (
                self.scheduler.has_waiting()
                or self.scheduler.num_running() > 0
                or self.scheduler.has_swapped()
            ):
                # ── 1. 准入：接入尽可能多的等待请求，块不足时尝试抢占 ──────────
                newly_admitted: list[RequestState] = []
                while (
                    self.scheduler.has_waiting()
                    and self.scheduler.num_running() < self.config.max_batch_size
                ):
                    next_state = self.scheduler.peek_next_waiting()
                    assert next_state is not None

                    prompt_len = len(next_state.prompt_token_ids)
                    max_out = next_state.request.sampling_params.max_new_tokens
                    blocks_needed = math.ceil((prompt_len + max_out) / self.config.block_size)

                    # 快速路径：即使清空所有 GPU 块也装不下，直接报错
                    if blocks_needed > self.config.num_gpu_blocks:
                        raise RuntimeError(
                            f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                            f"（prompt={prompt_len} + max_new_tokens={max_out}），"
                            f"超过系统总块数 {self.config.num_gpu_blocks}。"
                            f"请增大 num_gpu_blocks 或减小 max_new_tokens。"
                        )

                    if self.kv_cache.num_free_blocks() >= blocks_needed:
                        # 正常准入
                        state = self.scheduler.pop_next_waiting()
                        self.kv_cache.init_request(state)
                        self.scheduler.add_to_running(state)
                        newly_admitted.append(state)
                    else:
                        # 块不足：尝试换出优先级更低的 running 请求
                        victim = self.scheduler.get_lowest_priority_running()
                        if (
                            victim is not None
                            and victim.request.priority > next_state.request.priority
                        ):
                            if not victim.prefilled:
                                # 刚准入但尚未 prefill：撤销准入，块归还，放回 waiting 队尾
                                # 不能走 swap_out：此时 KV 为零值，保存到 CPU 无意义且会破坏续写
                                self.kv_cache.free_request(victim)
                                self.scheduler.un_admit(victim)
                                newly_admitted.remove(victim)
                            else:
                                # 已完成 prefill：KV 有效，换出到 CPU，加入换出队列
                                self.kv_cache.swap_out(victim)
                                self.scheduler.mark_swapped(victim)
                            continue  # 腾出块后重新检查空闲块数

                        # 无法换出：若确实卡死则报错，否则等待
                        if self.scheduler.num_running() == 0 and not newly_admitted:
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                f"（prompt={prompt_len} + max_new_tokens={max_out}），"
                                f"但当前仅有 {self.kv_cache.num_free_blocks()} 个空闲块，"
                                f"且没有优先级更低的 running 请求可换出。"
                            )
                        break  # 有 running 请求，等待它们完成后释放块

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

                # ── 5. 换入：将 swapped 请求换回 GPU（有足够块时）────────────
                for swapped_state in self.scheduler.get_swapped_states():
                    if self.scheduler.num_running() >= self.config.max_batch_size:
                        break
                    needed = max(
                        1, math.ceil(swapped_state.swapped_seq_len / self.config.block_size)
                    )
                    if self.kv_cache.num_free_blocks() >= needed:
                        self.kv_cache.swap_in(swapped_state)
                        self.scheduler.move_swapped_to_running(swapped_state)
                    else:
                        break  # FIFO：第一个换不回来则停止

        except Exception:
            # 异常时归还所有 running 请求的 GPU 块，防止显存泄漏
            for state in self.scheduler.get_running_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            # 清除 swapped 请求的 CPU KV，释放 CPU 内存
            for state in self.scheduler.get_swapped_states():
                state.cpu_kv = None
            raise

        return [outputs.get(state.request.request_id, "") for state in states]

    def _tokenize(self, text: str) -> list[int]:
        return self.model_runner.tokenizer.encode(text, add_special_tokens=True)
