"""
Phase 7/8/9/10 推理引擎。

Phase 7：continuous batching + preemption + priority scheduling（generate() 接口）。

Phase 8 新增：单步接口，供 AsyncEngine / HTTP 服务使用：
  - add_request(prompt, max_new_tokens, priority) → request_id
      将单个请求加入等待队列，返回 request_id。
  - step() → {request_id: [new_token_texts]}
      执行一次 prefill + decode_batch，返回本步每个请求新生成的 token 文本列表。
  - has_unfinished_requests() → bool
  - is_finished(request_id) → bool

Phase 9 新增：Chunked Prefill（chunk_prefill_size > 0 时启用）：
  - 长 prompt 请求分多步 prefill，每步只处理 chunk_prefill_size 个 token
  - PREFILLING 状态介于 WAITING 和 RUNNING 之间，一次最多 1 个请求处于该状态
  - 中间 DynamicCache 保存在 _prefilling_caches（当前实现保留在模型设备上），最后一个 chunk 后写入 block tensor
  - generate() 和 step() 两条路径均支持，chunk_prefill_size=0 时行为与 Phase 8 完全一致

注意：generate() 和 add_request/step() 使用同一个 scheduler/kv_cache，
不得同时混用——同一时刻只用一种接口。

主循环结构（每次 step 迭代）：
  1. 准入：接入尽可能多的等待请求，块不足时尝试抢占低优先级 running 请求
     （chunk_prefill_size > 0 时：准入到 PREFILLING 而非直接 RUNNING）
  2. Prefill：对新接入的请求做 prefill（或推进当前 PREFILLING 请求的一个 chunk）
  3. Batch decode：一次 forward 处理所有 running 请求
  4. 清理：移除已完成请求，释放 KV 块
  5. Swap_in：有空闲块时将 swapped 请求换回，加入 running
"""

import math
from uuid import uuid4

from ..cache.kv_cache import KVCacheManager
from ..core.config import EngineConfig
from ..core.request import Request, RequestState, SamplingParams
from ..modeling.model_runner import ModelRunner
from .scheduler import Scheduler


def _sync_cache_config_from_model(config: EngineConfig) -> None:
    """Load model architecture metadata before allocating KV cache tensors."""
    if config.dry_run:
        return

    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(config.model_name, trust_remote_code=True)

    num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    num_attention_heads = getattr(hf_config, "num_attention_heads", None)
    num_kv_heads = getattr(hf_config, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(hf_config, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(hf_config, "hidden_size", None)
        if hidden_size is not None and num_attention_heads is not None:
            head_dim = hidden_size // num_attention_heads

    missing = [
        name
        for name, value in (
            ("num_hidden_layers", num_hidden_layers),
            ("num_key_value_heads/num_attention_heads", num_kv_heads),
            ("head_dim or hidden_size/num_attention_heads", head_dim),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"无法从模型配置推导 KV cache 参数: {', '.join(missing)}")

    config.num_hidden_layers = int(num_hidden_layers)
    config.num_kv_heads = int(num_kv_heads)
    config.head_dim = int(head_dim)


class LLMEngine:
    """提供 generate 接口，实现 continuous batching + preemption 推理调度。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        if (
            not config.dry_run
            and config.device.startswith("cuda")
            and config.block_size % 256 != 0
        ):
            raise ValueError(
                "CUDA 推理路径使用 flash_attn_with_kvcache，"
                f"block_size 必须是 256 的倍数，当前为 {config.block_size}。"
                "CPU/MPS 路径无此限制，可使用任意 block_size（建议 16 或 32）。"
            )
        _sync_cache_config_from_model(config)
        self.kv_cache = KVCacheManager(config=config)
        self.scheduler = Scheduler(max_batch_size=config.max_batch_size)
        self.model_runner = ModelRunner(config=config, kv_cache=self.kv_cache)
        # Phase 8：单步接口的请求状态追踪（request_id → RequestState）
        self._step_states: dict[str, RequestState] = {}
        # Phase 9：chunked prefill 时保存中间 DynamicCache（request_id → DynamicCache | None）
        self._prefilling_caches: dict[str, object] = {}
        # Phase 12：CUDA Graph warmup（config.use_cuda_graph=True 时触发）
        if config.use_cuda_graph and not config.dry_run:
            self.model_runner.warmup_cuda_graphs()

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        priorities: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
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
                sampling_params=SamplingParams(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
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
                or self.scheduler.has_prefilling()
                or self.scheduler.num_running() > 0
                or self.scheduler.has_swapped()
            ):
                if self.config.chunk_prefill_size > 0:
                    # ── Chunked Prefill 路径（Phase 9）────────────────────────
                    # 1a. 若无正在进行的 prefill，从 waiting 准入一个请求到 PREFILLING
                    if (
                        not self.scheduler.has_prefilling()
                        and self.scheduler.has_waiting()
                        and (self.scheduler.num_running() + self.scheduler.num_prefilling())
                            < self.config.max_batch_size
                    ):
                        next_state = self.scheduler.peek_next_waiting()
                        assert next_state is not None
                        prompt_len = len(next_state.prompt_token_ids)
                        max_out = next_state.request.sampling_params.max_new_tokens
                        blocks_needed = math.ceil((prompt_len + max_out) / self.config.block_size)
                        if blocks_needed > self.config.num_gpu_blocks:
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                f"（prompt={prompt_len} + max_new_tokens={max_out}），"
                                f"超过系统总块数 {self.config.num_gpu_blocks}。"
                            )
                        if self.kv_cache.num_free_blocks() >= blocks_needed:
                            state = self.scheduler.pop_next_waiting()
                            self.kv_cache.init_request(state)
                            self.scheduler.add_to_prefilling(state)
                        else:
                            victim = self.scheduler.get_lowest_priority_running()
                            if (
                                victim is not None
                                and victim.request.priority > next_state.request.priority
                            ):
                                if not victim.prefilled:
                                    self.kv_cache.free_request(victim)
                                    self.scheduler.un_admit(victim)
                                else:
                                    self.kv_cache.swap_out(victim)
                                    self.scheduler.mark_swapped(victim)
                                # 腾出块后立即尝试准入
                                if self.kv_cache.num_free_blocks() >= blocks_needed:
                                    state = self.scheduler.pop_next_waiting()
                                    self.kv_cache.init_request(state)
                                    self.scheduler.add_to_prefilling(state)

                    # 2a. 推进当前 PREFILLING 请求的一个 chunk
                    pf_state = self.scheduler.get_next_prefilling()
                    if pf_state is not None:
                        rid = pf_state.request.request_id
                        t_start = pf_state.prefilled_tokens
                        t_end = min(
                            t_start + self.config.chunk_prefill_size,
                            len(pf_state.prompt_token_ids),
                        )
                        is_last = (t_end == len(pf_state.prompt_token_ids))
                        cache = self._prefilling_caches.get(rid)
                        new_cache = self.model_runner.prefill_chunk(
                            pf_state, t_start, t_end, cache, is_last
                        )
                        if is_last:
                            self._prefilling_caches.pop(rid, None)
                            self.scheduler.move_prefilling_to_running(pf_state)
                        else:
                            self._prefilling_caches[rid] = new_cache

                else:
                    # ── 原始路径（Phase 8 行为，chunk_prefill_size == 0）────────
                    # Phase 10：此路径支持 prefix cache（chunked prefill 路径不走 prefix cache）
                    # 1. 准入：接入尽可能多的等待请求，块不足时尝试抢占
                    newly_admitted: list[RequestState] = []
                    while (
                        self.scheduler.has_waiting()
                        and self.scheduler.num_running() < self.config.max_batch_size
                    ):
                        next_state = self.scheduler.peek_next_waiting()
                        assert next_state is not None

                        prompt_len = len(next_state.prompt_token_ids)
                        max_out = next_state.request.sampling_params.max_new_tokens
                        # Phase 10：预先查询 prefix cache，仅对 suffix + decode 部分计算所需新块数。
                        # cached_len > 0 时，这些块已在 cache 中，不需要新分配，避免准入时过度保守。
                        cached_len_peek, _ = self.kv_cache.find_prefix_cache(
                            next_state.prompt_token_ids
                        )
                        suffix_len = prompt_len - cached_len_peek  # no hit → suffix_len = prompt_len
                        blocks_needed = math.ceil((suffix_len + max_out) / self.config.block_size)

                        if blocks_needed > self.config.num_gpu_blocks:
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                f"（suffix={suffix_len} + max_new_tokens={max_out}，"
                                f"prefix_cached={cached_len_peek} tokens），"
                                f"超过系统总块数 {self.config.num_gpu_blocks}。"
                                f"请增大 num_gpu_blocks 或减小 max_new_tokens。"
                            )

                        if self.kv_cache.num_free_blocks() >= blocks_needed:
                            state = self.scheduler.pop_next_waiting()
                            self._admit_with_prefix(state)  # Phase 10：prefix cache 感知准入
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
                                    f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                    f"（suffix={suffix_len} + max_new_tokens={max_out}），"
                                    f"但当前仅有 {self.kv_cache.num_free_blocks()} 个空闲块，"
                                    f"且没有优先级更低的 running 请求可换出。"
                                )
                            break

                    # 2. Prefill（Phase 10：prefix hit 请求走 prefill_with_prefix，并在完成后注册缓存）
                    if newly_admitted:
                        self._prefill_and_register(newly_admitted)

                # ── 3. Batch decode：两条路径共用 ────────────────────────────
                running = self.scheduler.get_running_states()
                if running:
                    self.model_runner.decode_batch(running)

                # ── 4. 清理完成的请求 ────────────────────────────────────────
                for state in list(running):
                    if state.finished:
                        state.decoded_text = self.model_runner.tokenizer.decode(
                            state.generated_token_ids, skip_special_tokens=True
                        )
                        outputs[state.request.request_id] = state.decoded_text
                        self.kv_cache.free_request(state)
                        self.scheduler.finish_request(state)

                # ── 5. 换入：将 swapped 请求换回 GPU（有足够块时）────────────
                # 注意：检查时同时计入 num_prefilling()，防止 PREFILLING 请求完成后
                # 导致 running + 1 超出 max_batch_size（chunked prefill 场景下的边界条件）
                for swapped_state in self.scheduler.get_swapped_states():
                    if (self.scheduler.num_running() + self.scheduler.num_prefilling()
                            >= self.config.max_batch_size):
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
            # Phase 9：归还 PREFILLING 请求的 GPU 块
            for state in self.scheduler.get_prefilling_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            self._prefilling_caches.clear()
            # 清除 swapped 请求的 CPU KV，释放 CPU 内存
            for state in self.scheduler.get_swapped_states():
                state.cpu_kv = None
            raise

        return [outputs.get(state.request.request_id, "") for state in states]

    def _tokenize(self, text: str) -> list[int]:
        return self.model_runner.tokenizer.encode(text, add_special_tokens=True)

    # ------------------------------------------------------------------
    # Phase 10：Prefix Cache 辅助方法（仅用于 chunk_prefill_size == 0 路径）
    # ------------------------------------------------------------------

    def _admit_with_prefix(self, state: RequestState) -> None:
        """准入时查找 prefix cache：命中则 init_request_with_prefix，否则 init_request。"""
        cached_len, cached_blocks = self.kv_cache.find_prefix_cache(state.prompt_token_ids)
        if cached_len > 0:
            self.kv_cache.init_request_with_prefix(state, cached_len, cached_blocks)
            state.prefix_cached_len = cached_len
            state.prefix_cached_blocks = cached_blocks
        else:
            self.kv_cache.init_request(state)

    def _prefill_and_register(self, newly_admitted: list[RequestState]) -> None:
        """对新准入请求执行 prefill，然后将结果注册到 prefix cache。

        prefix cache miss 的请求批量调用 prefill()；
        prefix cache hit 的请求逐个调用 prefill_with_prefix()。
        所有请求 prefill 完成后统一调用 register_prefix_blocks_for_request() 注册缓存。
        """
        miss_states = [s for s in newly_admitted if s.prefix_cached_len == 0]
        hit_states = [s for s in newly_admitted if s.prefix_cached_len > 0]
        if miss_states:
            self.model_runner.prefill(miss_states)
        for state in hit_states:
            self.model_runner.prefill_with_prefix(
                state, state.prefix_cached_len, state.prefix_cached_blocks
            )
        # 注册所有新 prefill 的 block（含 miss 和 hit 的 suffix blocks）
        for state in newly_admitted:
            self.kv_cache.register_prefix_blocks_for_request(
                state.request.request_id, state.prompt_token_ids
            )

    # ------------------------------------------------------------------
    # Phase 8：单步接口（供 AsyncEngine / HTTP 服务使用）
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        request_id: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
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
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
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

    def get_finish_reason(self, request_id: str) -> str | None:
        """返回请求的内部结束原因（如 length / eos）。"""
        state = self._step_states.get(request_id)
        return state.finish_reason if state is not None else None

    def cancel_request(self, request_id: str) -> bool:
        """
        主动取消请求，并回收其 KV / CPU swap 资源。

        返回值：
          - True: 成功找到并移除了该请求
          - False: 请求不存在或已清理
        """
        state = self._step_states.pop(request_id, None)
        if state is None:
            return False

        removed = self.scheduler.remove_request(request_id)
        state = removed if removed is not None else state

        if request_id in self.kv_cache._block_tables:
            self.kv_cache.free_request(state)
        state.cpu_kv = None
        # Phase 9：清除中间 prefill cache
        self._prefilling_caches.pop(request_id, None)
        return removed is not None

    def has_unfinished_requests(self) -> bool:
        """True 当且仅当有 waiting、prefilling、running 或 swapped 请求。"""
        return (
            self.scheduler.has_waiting()
            or self.scheduler.has_prefilling()
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
            # just_prefilled：本步刚完成 prefill（整体 prefill 或最后一个 chunk）进入 running 的请求。
            # 供下方 pre_lens 逻辑将其初始 pre 置 0，使 prefill 采样的第一个 token 被捕获进入 new_tokens。
            just_prefilled: list[RequestState] = []

            if self.config.chunk_prefill_size > 0:
                # ── Chunked Prefill 路径（Phase 9）────────────────────────────
                # 1a. 若无正在进行的 prefill，从 waiting 准入一个请求到 PREFILLING
                if (
                    not self.scheduler.has_prefilling()
                    and self.scheduler.has_waiting()
                    and (self.scheduler.num_running() + self.scheduler.num_prefilling())
                        < self.config.max_batch_size
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
                        self.scheduler.add_to_prefilling(state)
                    else:
                        victim = self.scheduler.get_lowest_priority_running()
                        if (
                            victim is not None
                            and victim.request.priority > next_state.request.priority
                        ):
                            if not victim.prefilled:
                                self.kv_cache.free_request(victim)
                                self.scheduler.un_admit(victim)
                            else:
                                self.kv_cache.swap_out(victim)
                                self.scheduler.mark_swapped(victim)
                            if self.kv_cache.num_free_blocks() >= blocks_needed:
                                state = self.scheduler.pop_next_waiting()
                                self.kv_cache.init_request(state)
                                self.scheduler.add_to_prefilling(state)

                # 2a. 推进当前 PREFILLING 请求的一个 chunk
                pf_state = self.scheduler.get_next_prefilling()
                if pf_state is not None:
                    rid = pf_state.request.request_id
                    t_start = pf_state.prefilled_tokens
                    t_end = min(
                        t_start + self.config.chunk_prefill_size,
                        len(pf_state.prompt_token_ids),
                    )
                    is_last = (t_end == len(pf_state.prompt_token_ids))
                    cache = self._prefilling_caches.get(rid)
                    new_cache = self.model_runner.prefill_chunk(
                        pf_state, t_start, t_end, cache, is_last
                    )
                    if is_last:
                        self._prefilling_caches.pop(rid, None)
                        self.scheduler.move_prefilling_to_running(pf_state)
                        just_prefilled.append(pf_state)
                    else:
                        self._prefilling_caches[rid] = new_cache

            else:
                # ── 原始路径（Phase 8 行为，chunk_prefill_size == 0）──────────
                # Phase 10：此路径支持 prefix cache（chunked prefill 路径不走 prefix cache）
                # 1. 准入（与 generate() 相同逻辑）
                newly_admitted: list[RequestState] = []
                while (
                    self.scheduler.has_waiting()
                    and self.scheduler.num_running() < self.config.max_batch_size
                ):
                    next_state = self.scheduler.peek_next_waiting()
                    assert next_state is not None

                    prompt_len = len(next_state.prompt_token_ids)
                    max_out = next_state.request.sampling_params.max_new_tokens
                    # Phase 10：预先查询 prefix cache，仅对 suffix + decode 部分计算所需新块数
                    cached_len_peek, _ = self.kv_cache.find_prefix_cache(
                        next_state.prompt_token_ids
                    )
                    suffix_len = prompt_len - cached_len_peek
                    blocks_needed = math.ceil((suffix_len + max_out) / self.config.block_size)

                    if blocks_needed > self.config.num_gpu_blocks:
                        raise RuntimeError(
                            f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块，"
                            f"超过系统总块数 {self.config.num_gpu_blocks}。"
                        )

                    if self.kv_cache.num_free_blocks() >= blocks_needed:
                        state = self.scheduler.pop_next_waiting()
                        self._admit_with_prefix(state)  # Phase 10：prefix cache 感知准入
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

                # 2. Prefill（Phase 10：prefix hit 请求走 prefill_with_prefix，完成后注册缓存）
                if newly_admitted:
                    self._prefill_and_register(newly_admitted)
                    just_prefilled = newly_admitted

            # ── 3. Decode（收集前记录 pre_lens，捕获 prefill + decode 两阶段的新 token）
            running = self.scheduler.get_running_states()
            pre_lens: dict[str, int] = {
                s.request.request_id: len(s.generated_token_ids) for s in running
            }
            # 对刚完成 prefill 进入 running 的请求：pre 置 0，使 prefill 采样的第一个 token 被捕获
            for s in just_prefilled:
                pre_lens[s.request.request_id] = 0

            if running:
                self.model_runner.decode_batch(running)

            # ── 4. 收集新 token（清理前，确保 finished 请求的最后 token 也被捕获）
            #
            # 注意：real GPU 路径的 prefill/decode_batch 始终以 "" 存入 generated_text_parts
            # （逐 token decode 对多字节字符会返回空串），因此不能依赖 generated_text_parts。
            # 改用增量 tokenizer.decode：decode(all_ids[:curr]) - decode(all_ids[:pre])
            # 的文本差值，确保 real GPU 路径也能正确生成可读文本。
            # dry_run 路径同样走此逻辑（_StubTokenizer.decode 返回 " [1] [2]..." 格式正确）。
            for state in self.scheduler.get_running_states():
                rid = state.request.request_id
                pre = pre_lens.get(rid, 0)
                curr = len(state.generated_token_ids)
                if curr > pre:
                    tok_ids = state.generated_token_ids
                    old_text = (
                        self.model_runner.tokenizer.decode(
                            tok_ids[:pre], skip_special_tokens=True
                        )
                        if pre > 0 else ""
                    )
                    new_text = self.model_runner.tokenizer.decode(
                        tok_ids[:curr], skip_special_tokens=True
                    )
                    state.decoded_text = new_text
                    delta = new_text[len(old_text):]
                    if delta:
                        new_tokens[rid] = [delta]

            # ── 5. 清理完成的请求 ────────────────────────────────────────
            for state in list(self.scheduler.get_running_states()):
                if state.finished:
                    self.kv_cache.free_request(state)
                    self.scheduler.finish_request(state)

            # ── 6. Swap_in ───────────────────────────────────────────────
            # 注意：检查时同时计入 num_prefilling()，防止 PREFILLING 请求完成后
            # 导致 running + 1 超出 max_batch_size（chunked prefill 场景下的边界条件）
            for swapped_state in self.scheduler.get_swapped_states():
                if (self.scheduler.num_running() + self.scheduler.num_prefilling()
                        >= self.config.max_batch_size):
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
            # Phase 9：归还 PREFILLING 请求的 GPU 块
            for state in self.scheduler.get_prefilling_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            self._prefilling_caches.clear()
            for state in self.scheduler.get_swapped_states():
                state.cpu_kv = None
            raise

        return new_tokens
