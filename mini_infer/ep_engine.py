"""Phase 17：2 卡 Expert Parallel 引擎原型。

这个文件复用 Phase 13 的 `mp.spawn + file:// rendezvous` 约定，
把 `EPMoELayer` 接成可直接运行的 2 卡功能原型。
当前目标是验证 all-to-all forward 和结果回收，而不是提供生产级常驻服务。
"""

from __future__ import annotations

import os
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .moe_layer import EPMoELayer, MoELayer


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _ep_worker(
    rank: int,
    ep_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    bias: bool,
    dtype: str,
    hidden_states_cpu: torch.Tensor,
    layer_state_dict: dict[str, torch.Tensor],
    src_rank: int,
    warmup: int,
    runs: int,
    measure_steady_state: bool,
    result_file: str,
    rendezvous_file: str,
) -> None:
    init_method = f"file://{rendezvous_file}"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=rank,
        world_size=ep_size,
    )

    try:
        torch.cuda.set_device(rank)
        device = f"cuda:{rank}"
        torch_dtype = _DTYPE_MAP[dtype]
        layer = EPMoELayer(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            top_k=top_k,
            ep_size=ep_size,
            rank=rank,
            bias=bias,
            dist_group=None,
            src_rank=src_rank,
        ).to(device=device, dtype=torch_dtype)

        state_dict = {
            key: value.to(dtype=torch_dtype)
            if torch.is_floating_point(value)
            else value
            for key, value in layer_state_dict.items()
        }
        layer.load_state_dict(state_dict, strict=True)
        layer.eval()

        num_tokens = hidden_states_cpu.reshape(-1, hidden_states_cpu.shape[-1]).shape[0]
        hidden_states = hidden_states_cpu.to(device=device, dtype=torch_dtype) if rank == src_rank else None

        with torch.no_grad():
            for _ in range(warmup):
                _ = layer(hidden_states, return_aux=False, num_tokens=num_tokens)

            dist.barrier()
            start_time = None
            if measure_steady_state:
                torch.cuda.synchronize(rank)
                start_time = time.perf_counter()

            result = None
            for _ in range(runs):
                result = layer(hidden_states, return_aux=True, num_tokens=num_tokens)

            dist.barrier()
            elapsed_s = None
            if measure_steady_state:
                torch.cuda.synchronize(rank)
                assert start_time is not None
                elapsed_s = time.perf_counter() - start_time

        if rank == src_rank:
            assert isinstance(result, tuple)
            output, aux = result
            assert output is not None and aux is not None
            torch.save(
                {
                    "output": output.cpu(),
                    "send_counts": aux.send_counts.cpu(),
                    "expert_loads": aux.routing_stats.expert_loads.cpu(),
                    "expert_score_sums": aux.routing_stats.expert_score_sums.cpu(),
                    "elapsed_s": elapsed_s,
                    "warmup": warmup,
                    "runs": runs,
                },
                result_file,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class EPEngine:
    """2 卡 EP 功能原型。

    当前接口以单层 `EPMoELayer` 为核心，输入为 synthetic hidden states。
    benchmark 阶段会明确标注：默认 `mp.spawn` 路径包含 worker 启动开销。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
        ep_size: int = 2,
        dtype: str = "float16",
        layer_state_dict: dict[str, torch.Tensor] | None = None,
        src_rank: int = 0,
    ) -> None:
        if ep_size < 2:
            raise ValueError(f"ep_size 必须 >= 2，当前 {ep_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("EPEngine 需要 CUDA GPU")
        if torch.cuda.device_count() < ep_size:
            raise RuntimeError(f"需要 {ep_size} 个 GPU，但只有 {torch.cuda.device_count()} 个")
        if dtype not in _DTYPE_MAP:
            raise ValueError(f"不支持的 dtype: {dtype!r}")
        if src_rank < 0 or src_rank >= ep_size:
            raise ValueError(f"src_rank 必须落在 [0, ep_size) 内，当前 src_rank={src_rank}, ep_size={ep_size}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias = bias
        self.ep_size = ep_size
        self.dtype = dtype
        self.src_rank = src_rank
        self.layer_state_dict = None if layer_state_dict is None else {
            key: value.detach().cpu()
            for key, value in layer_state_dict.items()
        }

    @classmethod
    def from_moe_layer(
        cls,
        layer: MoELayer,
        ep_size: int = 2,
        dtype: str = "float16",
        src_rank: int = 0,
    ) -> "EPEngine":
        return cls(
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size,
            num_experts=layer.num_experts,
            top_k=layer.top_k,
            bias=layer.router.gate.bias is not None,
            ep_size=ep_size,
            dtype=dtype,
            layer_state_dict=layer.state_dict(),
            src_rank=src_rank,
        )

    def _run(
        self,
        hidden_states: torch.Tensor,
        warmup: int,
        runs: int,
        measure_steady_state: bool,
    ) -> dict[str, object]:
        if warmup < 0:
            raise ValueError(f"warmup 必须 >= 0，当前 {warmup}")
        if runs <= 0:
            raise ValueError(f"runs 必须 > 0，当前 {runs}")
        if self.layer_state_dict is None:
            raise RuntimeError("EPEngine.forward 需要 layer_state_dict；请使用 from_moe_layer() 或显式传入")

        result_file = tempfile.mktemp(suffix=".pt")
        rendezvous_file = tempfile.mktemp()

        try:
            mp.spawn(
                _ep_worker,
                args=(
                    self.ep_size,
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_experts,
                    self.top_k,
                    self.bias,
                    self.dtype,
                    hidden_states.detach().cpu(),
                    self.layer_state_dict,
                    self.src_rank,
                    warmup,
                    runs,
                    measure_steady_state,
                    result_file,
                    rendezvous_file,
                ),
                nprocs=self.ep_size,
                join=True,
            )
            return torch.load(result_file, map_location="cpu")
        finally:
            for path in [result_file, rendezvous_file]:
                if os.path.exists(path):
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden_states.ndim < 2:
            raise ValueError(f"hidden_states 至少需要 2 维，当前 shape={tuple(hidden_states.shape)}")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states 最后一维必须等于 hidden_size={self.hidden_size}，"
                f"当前 {hidden_states.shape[-1]}"
            )
        result = self._run(
            hidden_states=hidden_states,
            warmup=0,
            runs=1,
            measure_steady_state=False,
        )
        if not return_aux:
            return result["output"]
        return result["output"], {
            "send_counts": result["send_counts"],
            "expert_loads": result["expert_loads"],
            "expert_score_sums": result["expert_score_sums"],
        }

    def benchmark_forward(
        self,
        hidden_states: torch.Tensor,
        warmup: int = 1,
        runs: int = 3,
    ) -> dict[str, object]:
        """单次 spawn 内完成 warmup + steady-state 计时，排除 worker 启动噪声。"""
        if hidden_states.ndim < 2:
            raise ValueError(f"hidden_states 至少需要 2 维，当前 shape={tuple(hidden_states.shape)}")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_states 最后一维必须等于 hidden_size={self.hidden_size}，"
                f"当前 {hidden_states.shape[-1]}"
            )
        return self._run(
            hidden_states=hidden_states,
            warmup=warmup,
            runs=runs,
            measure_steady_state=True,
        )


__all__ = ["EPEngine"]
