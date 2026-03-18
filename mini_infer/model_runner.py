"""这个文件定义模型执行器，Phase 1 接入真实模型和 tokenizer，实现 prefill 和 autoregressive decode 两条路径。dry_run=True 时保留桩实现供本地无 GPU 测试用。"""

import torch

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
        # Phase 1：用 HuggingFace past_key_values 存储每个请求的 KV，Phase 2 替换为分页 GPU tensor
        self._past_kv: dict[str, tuple] = {}

        if config.dry_run:
            self.tokenizer: _StubTokenizer | "AutoTokenizer" = _StubTokenizer()
            self.eos_token_id: int = -1
            self.model = None
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if config.device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError("配置要求 CUDA 设备，但当前环境未检测到可用 GPU。")

            tokenizer_name = config.tokenizer_name or config.model_name
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.eos_token_id = self.tokenizer.eos_token_id or 0

            dtype = _resolve_torch_dtype(config.dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            self.model.to(config.device)
            self.model.eval()

    def prefill(self, states: list[RequestState]) -> None:
        """对每个请求独立跑 prefill forward，从 prefill logits 中采样第一个 token。"""
        if self.config.dry_run:
            for state in states:
                state.prefilled = True
            return

        for state in states:
            input_ids = torch.tensor(
                [state.prompt_token_ids], dtype=torch.long, device=self.config.device
            )
            with torch.no_grad():
                out = self.model(input_ids=input_ids, use_cache=True)

            next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
            state.append_generated(next_token_id, "")
            self.kv_cache.append_token(state, next_token_id)
            self._past_kv[state.request.request_id] = out.past_key_values
            state.prefilled = True

            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")

    def decode_step(self, states: list[RequestState]) -> None:
        """对所有未完成请求各自执行一步 decode，读取 past_key_values 并更新。"""
        if self.config.dry_run:
            for state in states:
                if state.finished:
                    continue
                next_index = len(state.generated_token_ids) + 1
                state.append_generated(next_index, f" [{next_index}]")
                self.kv_cache.append_token(state, next_index)
                if len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                    state.mark_finished("length")
            return

        for state in states:
            if state.finished:
                continue

            last_token_id = state.generated_token_ids[-1]
            input_ids = torch.tensor(
                [[last_token_id]], dtype=torch.long, device=self.config.device
            )
            past_kv = self._past_kv[state.request.request_id]

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids, past_key_values=past_kv, use_cache=True
                )

            next_token_id = _sample_token(out.logits[0, -1], state.request.sampling_params)
            state.append_generated(next_token_id, "")
            self.kv_cache.append_token(state, next_token_id)
            self._past_kv[state.request.request_id] = out.past_key_values

            if next_token_id == self.eos_token_id:
                state.mark_finished("eos")
            elif len(state.generated_token_ids) >= state.request.sampling_params.max_new_tokens:
                state.mark_finished("length")

    def free_request(self, state: RequestState) -> None:
        """释放请求占用的 past_key_values，避免显存泄漏。"""
        self._past_kv.pop(state.request.request_id, None)
