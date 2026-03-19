"""
Phase 3/5 模型执行器。

Phase 5 变化（相比 Phase 3）：
  - decode_batch() 内三个关键段添加 torch.profiler.record_function 标签：
    "gather_batch_kv" / "model_forward" / "write_decode_kv"
  - 标签在 profiler 未激活时为 no-op，不影响正常推理性能

Phase 3 变化（相比 Phase 2）：
  - decode_batch() 改用 DynamicCache 替代 tuple 格式的 past_key_values，
    消除 transformers 4.40+ 的弃用警告，兼容后续版本

Phase 2 已有的设计：
  - prefill() 写完 past_key_values 后，将 KV 写入 KVCacheManager 的 block tensor
  - decode_batch() 从 block tensor 聚合 KV（左填充对齐），一次 batch forward，写回新 KV

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

    def decode_batch(self, states: list[RequestState]) -> None:
        """
        Batch decode：把所有未完成请求合并成一次 GPU forward，而非串行 for 循环。

        算法：
          1. 从 block tensor 聚合每个请求的 KV（左填充到 max_seq_len）
          2. 构造 past_key_values（HF 格式）和 attention_mask（左填充区域为 0）
          3. 一次 model forward，batch_size = 活跃请求数
          4. 在 del 大张量前提取 logits 和新 KV
          5. 把新 token 的 KV 写回 block tensor
          6. 从 logits 采样下一个 token，更新请求状态
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
        batch_size = len(active)

        # 1. 从 block tensor 聚合 KV，left-pad 对齐到 max_seq_len
        with torch.profiler.record_function("gather_batch_kv"):
            k_batch, v_batch, seq_lens = self.kv_cache.gather_batch_kv(request_ids)
        max_seq_len = max(seq_lens)
        num_layers = len(k_batch)

        # 2. 构造 DynamicCache：pre-populate with gathered KV（替代 tuple，消除弃用警告）
        cache = DynamicCache()
        for l in range(num_layers):
            cache.update(k_batch[l], v_batch[l], l)

        # 3. 构造 input_ids：每个请求最后生成的 token，shape [batch, 1]
        last_tokens = [s.generated_token_ids[-1] for s in active]
        input_ids = torch.tensor(
            [[t] for t in last_tokens], dtype=torch.long, device=self.config.device
        )

        # 4. 构造 attention_mask：shape [batch, max_seq_len + 1]
        # 左填充区域为 0，真实 token 区域 + 新 token 位置为 1
        attn_mask = torch.zeros(
            batch_size, max_seq_len + 1, dtype=torch.long, device=self.config.device
        )
        for b, seq_len in enumerate(seq_lens):
            attn_mask[b, max_seq_len - seq_len:] = 1

        # 5. 一次 batch forward（past_key_values 传 DynamicCache，model 会 in-place append 新 KV）
        with torch.profiler.record_function("model_forward"):
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=cache,
                    attention_mask=attn_mask,
                    use_cache=True,
                )

        # 6. 提取新 token KV 和 logits
        # forward 后 out.past_key_values.key_cache[l] shape: [batch, num_kv_heads, max_seq_len+1, head_dim]
        # 最后位置（-1）是本次新 token 的 KV
        k_new = [out.past_key_values.key_cache[l][:, :, -1, :].clone() for l in range(num_layers)]
        v_new = [out.past_key_values.value_cache[l][:, :, -1, :].clone() for l in range(num_layers)]
        logits_batch = out.logits[:, 0, :].clone()  # [batch, vocab_size]

        # 写回 block tensor，并释放大张量
        with torch.profiler.record_function("write_decode_kv"):
            self.kv_cache.write_decode_kv(request_ids, k_new, v_new)
        del k_batch, v_batch, cache, out, k_new, v_new

        # 7. 采样下一个 token，更新请求状态
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
