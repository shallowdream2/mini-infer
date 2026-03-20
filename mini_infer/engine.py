"""
Phase 7/8 推理引擎。

Phase 7：continuous batching + preemption + priority scheduling（generate() 接口）。

Phase 8 新增：单步接口，供 AsyncEngine / HTTP 服务使用：
  - add_request(prompt, max_new_tokens, priority) → request_id
      将单个请求加入等待队列，返回 request_id。
  - step() → {request_id: [new_token_texts]}
      执行一次 prefill + decode_batch，返回本步每个请求新生成的 token 文本列表。
  - has_unfinished_requests() → bool
  - is_finished(request_id) → bool

注意：generate() 和 add_request/step() 使用同一个 scheduler/kv_cache，
不得同时混用——同一时刻只用一种接口。

主循环结构（每次 step 迭代）：
  1. 准入：接入尽可能多的等待请求，块不足时尝试抢占低优先级 running 请求
  2. Prefill：对新接入的请求做 prefill
  3. Batch decode：一次 forward 处理所有 running 请求
  4. 清理：移除已完成请求，释放 KV 块
  5. Swap_in：有空闲块时将 swapped 请求换回，加入 running
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
        # Phase 8：单步接口的请求状态追踪（request_id → RequestState）
        self._step_states: dict[str, RequestState] = {}

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

    # ------------------------------------------------------------------
    # Phase 8：单步接口（供 AsyncEngine / HTTP 服务使用）
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        request_id: str | None = None,
    ) -> str:
        """
        将单个请求加入等待队列，返回 request_id。

        request_id：外部预生成的 ID（供 AsyncEngine 在注册 token queue 后再调用此方法，
                    消除"queue 未注册时 step() 先投递 token"的竞态）。
                    若为 None，内部自动生成。

        与 step() 配合使用，实现流式/异步推理。
        不得与 generate() 同时混用——同一时刻只用一种接口。
        """
        if request_id is None:
            request_id = str(uuid4())
        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            priority=priority,
        )
        state = RequestState(
            request=request,
            prompt_token_ids=self._tokenize(prompt),
        )
        self.scheduler.add_request(state)
        self._step_states[request_id] = state
        return request_id

    def cleanup_request(self, request_id: str) -> None:
        """
        从内部追踪表中删除已完成请求的 state。

        供 AsyncEngine 在确认 DONE 哨兵已投递后调用，防止 _step_states 无限增长。
        调用前提：is_finished(request_id) 已返回 True。
        """
        self._step_states.pop(request_id, None)

    def has_unfinished_requests(self) -> bool:
        """True 当且仅当有 waiting、running 或 swapped 请求。"""
        return (
            self.scheduler.has_waiting()
            or self.scheduler.num_running() > 0
            or self.scheduler.has_swapped()
        )

    def is_finished(self, request_id: str) -> bool:
        """True 当且仅当该请求已完成生成。"""
        state = self._step_states.get(request_id)
        return state is not None and state.finished

    def step(self) -> dict[str, list[str]]:
        """
        执行一次 prefill + decode 迭代。

        返回 {request_id: [new_token_texts]}：本步每个请求新生成的 token 文本列表。
        完成的请求在本步最后一次出现（含最后 token），随后 is_finished() 返回 True。
        """
        new_tokens: dict[str, list[str]] = {}

        try:
            # ── 1. 准入（与 generate() 相同逻辑）────────────────────────
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

                if blocks_needed > self.config.num_gpu_blocks:
                    raise RuntimeError(
                        f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块，"
                        f"超过系统总块数 {self.config.num_gpu_blocks}。"
                    )

                if self.kv_cache.num_free_blocks() >= blocks_needed:
                    state = self.scheduler.pop_next_waiting()
                    self.kv_cache.init_request(state)
                    self.scheduler.add_to_running(state)
                    newly_admitted.append(state)
                else:
                    victim = self.scheduler.get_lowest_priority_running()
                    if (
                        victim is not None
                        and victim.request.priority > next_state.request.priority
                    ):
                        if not victim.prefilled:
                            self.kv_cache.free_request(victim)
                            self.scheduler.un_admit(victim)
                            newly_admitted.remove(victim)
                        else:
                            self.kv_cache.swap_out(victim)
                            self.scheduler.mark_swapped(victim)
                        continue

                    if self.scheduler.num_running() == 0 and not newly_admitted:
                        raise RuntimeError(
                            f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个块，"
                            f"仅有 {self.kv_cache.num_free_blocks()} 个空闲块，无法换出。"
                        )
                    break

            # ── 2. Prefill ──────────────────────────────────────────────
            if newly_admitted:
                self.model_runner.prefill(newly_admitted)

            # ── 3. Decode（收集前记录 pre_lens，捕获 prefill + decode 两阶段的新 token）
            running = self.scheduler.get_running_states()
            pre_lens: dict[str, int] = {
                s.request.request_id: len(s.generated_token_ids) for s in running
            }
            # 对刚 prefill 的请求：从 0 开始计，使 prefill token 也进入 new_tokens
            for s in newly_admitted:
                pre_lens[s.request.request_id] = 0

            if running:
                self.model_runner.decode_batch(running)

            # ── 4. 收集新 token（清理前，确保 finished 请求的最后 token 也被捕获）
            for state in self.scheduler.get_running_states():
                rid = state.request.request_id
                pre = pre_lens.get(rid, 0)
                parts = list(state.generated_text_parts[pre:])
                if parts:
                    new_tokens[rid] = parts

            # ── 5. 清理完成的请求 ────────────────────────────────────────
            for state in list(self.scheduler.get_running_states()):
                if state.finished:
                    self.kv_cache.free_request(state)
                    self.scheduler.finish_request(state)

            # ── 6. Swap_in ───────────────────────────────────────────────
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
                    break

        except Exception:
            for state in self.scheduler.get_running_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            for state in self.scheduler.get_swapped_states():
                state.cpu_kv = None
            raise

        return new_tokens
