"""
Phase 3/5/6/9 模型执行器。

Phase 9 新增（相比 Phase 6）：
  - prefill_chunk(state, token_start, token_end, past_cache, is_last_chunk)：
      对 prompt_token_ids[token_start:token_end] 做部分 prefill
      非最后 chunk：返回积累的 DynamicCache（供下一 chunk 使用）
      最后一个 chunk：写入 block tensor，采样第一个 token，设 prefilled=True，返回 None

Phase 6 变化（相比 Phase 5）：
  - decode_batch() 切换到 True PagedAttention 路径：
      - 删除 gather_batch_kv / DynamicCache / write_decode_kv
      - 改用 kv_cache.ensure_next_slot + build_block_tables + PagedDecodeContext
      - flash_attn_with_kvcache 直接从 block tensor 寻址，in-place 写入新 KV
  - __init__() 调用 patch_model_for_paged_decode() 永久 patch Qwen2 attention 层
  - prefill() 路径不变（仍用 DynamicCache，patch 在 prefill 时自动回退原始 HF forward）

Phase 5 变化（相比 Phase 3）：
  - decode_batch() 内三个关键段添加 torch.profiler.record_function 标签

Phase 3 变化（相比 Phase 2）：
  - decode_batch() 改用 DynamicCache 替代 tuple 格式的 past_key_values

Phase 2 已有的设计：
  - prefill() 写完 past_key_values 后，将 KV 写入 KVCacheManager 的 block tensor
  - decode_batch() batch 所有活跃请求做一次 GPU forward

dry_run=True 时保留桩实现，不加载真实模型，供无 GPU 的单元测试使用。
"""

import torch
from transformers import DynamicCache

from .config import EngineConfig
from .kv_cache import KVCacheManager
from .request import RequestState, SamplingParams


def _resolve_torch_dtype(dtype: str) -> torch.dtype:
    """把配置里的 dtype 字符串转换成 torch dtype。"""
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype]


def _sample_token(logits: torch.Tensor, params: SamplingParams) -> int:
    """从最后一个位置的 logits 采样下一个 token，支持 greedy（temperature=0）和 top-p。"""
    if params.temperature == 0.0:
        return int(logits.argmax(dim=-1).item())

    logits = logits / params.temperature
    probs = torch.softmax(logits, dim=-1)

    if params.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative - sorted_probs > params.top_p] = 0.0
        probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
        probs = probs / probs.sum()

    return int(torch.multinomial(probs, num_samples=1).item())


class _StubTokenizer:
    """dry_run 模式下的占位 tokenizer，不依赖 transformers，仅供本地测试用。"""

    eos_token_id = -1

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(f" [{tid}]" for tid in token_ids)


class ModelRunner:
    """模型执行器。dry_run=True 时使用桩实现（不加载模型），False 时加载真实模型和 tokenizer。"""

    def __init__(self, config: EngineConfig, kv_cache: KVCacheManager) -> None:
        self.config = config
        self.kv_cache = kv_cache

        if config.dry_run:
            self.tokenizer: _StubTokenizer | "AutoTokenizer" = _StubTokenizer()
            self.eos_token_id: int = -1
            self.model = None
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("配置要求 CUDA 设备，但当前环境未检测到可用 GPU。")

            tokenizer_name = config.tokenizer_name if config.tokenizer_name is not None else config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # 修复 Phase 1 bug：eos_token_id or 0 在 eos_token_id=None 时静默赋 0
            eos_id = self.tokenizer.eos_token_id
            self.eos_token_id = eos_id if eos_id is not None else -1

            dtype = _resolve_torch_dtype(config.dtype)
            # 使用 device_map 直接加载到目标 GPU，避免 CPU 中转导致的峰值显存 ×2
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
                device_map=config.device,
            )
            self.model.eval()

            # Phase 6：永久 patch attention 层，decode 时走 paged attention 路径
            from .attention import patch_model_for_paged_decode
            self._paged_ctx = patch_model_for_paged_decode(self.model, self.kv_cache)

    def prefill(self, states: list[RequestState]) -> None:
        """
        对每个请求独立跑 prefill forward，从 prefill logits 采样第一个 token，
        并将 KV 写入 KVCacheManager 的 block tensor。
        """
        if self.config.dry_run:
            for state in states:
                # 桩实现：生成 token [1] 作为第一个生成 token
                state.append_generated(1, " [1]")
                state.prefilled = True
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
            return

        for state in states:
            input_ids = torch.tensor(
                [state.prompt_token_ids], dtype=torch.long, device=self.config.device
            )
            with torch.no_grad():
                # 传入空 DynamicCache，使模型返回 DynamicCache 而非 tuple，消除弃用警告
                out = self.model(input_ids=input_ids, past_key_values=DynamicCache(), use_cache=True)

            # 将 prefill KV 写入 block tensor（替代 Phase 1 的 _past_kv dict）
            self.kv_cache.write_prefill_kv(state.request.request_id, out.past_key_values)

            next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
            state.append_generated(next_token_id, "")
            state.prefilled = True

            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")

    def prefill_chunk(
        self,
        state: RequestState,
        token_start: int,
        token_end: int,
        past_cache,
        is_last_chunk: bool,
    ):
        """
        Phase 9：对 prompt_token_ids[token_start:token_end] 做部分 prefill。

        past_cache: 上一 chunk 返回的 DynamicCache（第一个 chunk 传 None）。
        is_last_chunk=False：返回积累的 DynamicCache，不采样 token，不设 prefilled=True。
        is_last_chunk=True：写入 block tensor，采样第一个 token，设 prefilled=True，返回 None。

        调用方职责：
          - 非最后 chunk：保存返回值供下一次调用使用
          - 最后 chunk：丢弃返回值（None），并将请求移入 running
        """
        state.prefilled_tokens = token_end

        if self.config.dry_run:
            if is_last_chunk:
                state.append_generated(1, " [1]")
                state.prefilled = True
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
            return None

        chunk_ids = state.prompt_token_ids[token_start:token_end]
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=self.config.device)

        if past_cache is None:
            past_cache = DynamicCache()

        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=past_cache, use_cache=True)

        if is_last_chunk:
            self.kv_cache.write_prefill_kv(state.request.request_id, out.past_key_values)
            next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
            state.append_generated(next_token_id, "")
            state.prefilled = True
            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")
            return None
        else:
            return out.past_key_values  # 积累的 DynamicCache，供下一 chunk 使用

    def decode_batch(self, states: list[RequestState]) -> None:
        """
        Batch decode（Phase 6 True PagedAttention 路径）：
        flash_attn_with_kvcache 直接从 block tensor 寻址，无 gather / DynamicCache / write_kv。

        算法：
          1. ensure_next_slot：确保下一写入位置已有物理块
          2. build_block_tables：构造 block_table / cache_seqlens 张量
          3. _paged_ctx.set：注入上下文，触发 patched attention 层走 paged 路径
          4. model.forward（无 past_key_values，无 attention_mask）：
             flash_attn_with_kvcache 在每层 in-place 写入新 KV，并完成 attention 计算
          5. advance_seq_lens：递增各请求的 seq_len（KV 已由 flash_attn 写入）
          6. 采样下一个 token，更新请求状态
        """
        active = [s for s in states if not s.finished]
        if not active:
            return

        if self.config.dry_run:
            active_ids = []
            for state in active:
                next_idx = len(state.generated_token_ids) + 1
                state.append_generated(next_idx, f" [{next_idx}]")
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
                else:
                    active_ids.append(state.request.request_id)
            # 更新 dry_run 模式下的块管理元数据（不写 GPU 张量）
            if active_ids:
                self.kv_cache.write_decode_kv(active_ids, None, None)
            return

        request_ids = [s.request.request_id for s in active]

        # Phase 6 True PagedAttention 路径：
        # 1. 确保每个请求的下一个写入位置已有物理块（原 write_decode_kv 的块分配职责）
        self.kv_cache.ensure_next_slot(request_ids)

        # 2. 构造 flash_attn 所需张量
        block_table, cache_seqlens = self.kv_cache.build_block_tables(request_ids)

        # 3. input_ids：每个请求最后生成的 token，shape [batch, 1]
        last_tokens = [s.generated_token_ids[-1] for s in active]
        input_ids = torch.tensor(
            [[t] for t in last_tokens], dtype=torch.long, device=self.config.device
        )

        # 4. position_ids：新 token 的真实位置 = 当前 cache 长度
        #    形状 [batch, 1]，供各层 attention patched_forward 做 RoPE
        position_ids = cache_seqlens.long().unsqueeze(1)

        # 5. 注入 paged context，触发所有 attention 层走 paged 路径
        #    max_kv_len 在此处做唯一一次 .item() 同步，避免 28 层各同步一次（性能关键）
        max_kv_len = int(cache_seqlens.max().item()) + 1
        self._paged_ctx.set(block_table, cache_seqlens, max_kv_len)

        # 6. 一次 batch forward（无 past_key_values，无 attention_mask；
        #    flash_attn_with_kvcache 通过 block_table+cache_seqlens 管理 KV）
        try:
            with torch.profiler.record_function("model_forward"):
                with torch.no_grad():
                    out = self.model(
                        input_ids=input_ids,
                        position_ids=position_ids,
                        use_cache=False,
                    )

            logits_batch = out.logits[:, 0, :].clone()  # [batch, vocab_size]
            del out

            # 7. flash_attn 已 in-place 写入新 KV，只需递增 seq_len
            self.kv_cache.advance_seq_lens(request_ids)
        finally:
            # 无论 forward 是否抛出异常，都必须清除 paged context
            # 否则下一次 prefill 会误走 decode 路径
            self._paged_ctx.clear()

        # 8. 采样下一个 token，更新请求状态
        for b, state in enumerate(active):
            next_token_id = _sample_token(logits_batch[b], state.request.sampling_params)
            state.append_generated(next_token_id, "")

            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")

    def free_request(self, state: RequestState) -> None:
        """Phase 2 中 KV 由 KVCacheManager 管理，此处为接口兼容保留，无需操作。"""
        pass
