"""Phase 17-21：2 卡 Expert Parallel 引擎原型。

这个文件复用 Phase 13 的 `mp.spawn + file:// rendezvous` 约定，
把 `EPMoELayer` 接成可直接运行的 2 卡功能原型。Phase 18 开始 worker
不再接收完整 expert 权重，而是按 rank 只加载本地 expert shard；为了避免
正式 benchmark 配置下 `mp.spawn` 因大体积 tensor 传参触发 `fds_to_keep`
失败，rank-local shard 会先落到临时文件，再由各 worker 按 rank 读取。
Phase 19 新增 `comm_mode`，用于在 `padded` 与 `packed` EP 通信路径之间切换。
Phase 20 继续把 packed 路径的 worker-side control plane 收敛成显式 helper，并在
benchmark 结果里单独暴露 split-size 控制面成本。
Phase 21 继续增加 `expert_exec_mode`，用于在 `naive` 与 `grouped` local expert
execution 之间切换。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .moe_layer import (
    EPMoELayer,
    GroupedExpertMetadata,
    MoELayer,
    build_grouped_expert_metadata_from_local_counts,
    build_grouped_local_expert_counts,
    build_packed_control_plane,
    shard_moe_state_dict,
    _validate_comm_mode,
    _validate_expert_exec_mode,
)


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def _prepare_packed_control_plane(
    packed_send_counts: torch.Tensor,
    ep_size: int,
    rank: int,
    src_rank: int,
    device: int | None = None,
) -> tuple[object, float]:
    """把 broadcast 后的 send-counts 收敛成 worker 可直接使用的 packed control plane。"""
    # 先等 source-rank router/dispatch 和 send-count broadcast 完成，再单独计量
    # GPU -> CPU 的 send-count 读回与 Python split-size helper 本身。
    if device is not None:
        torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    packed_send_counts_cpu = packed_send_counts.cpu().tolist()
    control_plane = build_packed_control_plane(
        send_counts_cpu=packed_send_counts_cpu,
        world_size=ep_size,
        rank=rank,
        src_rank=src_rank,
    )
    return control_plane, time.perf_counter() - start_time


def _prepare_grouped_metadata(
    local_grouped_counts: torch.Tensor,
    *,
    local_expert_offset: int,
) -> tuple[GroupedExpertMetadata, float]:
    """把 rank-local grouped expert 计数收敛成可直接执行的 contiguous metadata。"""
    start_time = time.perf_counter()
    local_counts_cpu = local_grouped_counts.cpu().tolist()
    metadata = build_grouped_expert_metadata_from_local_counts(
        local_counts_cpu,
        local_expert_offset=local_expert_offset,
    )
    return metadata, time.perf_counter() - start_time


def _rank_state_dict_path(rank_state_dict_dir: str, rank: int) -> str:
    return os.path.join(rank_state_dict_dir, f"rank_{rank}.pt")


def _dump_rank_state_dicts(
    rank_state_dicts: list[dict[str, torch.Tensor]],
    rank_state_dict_dir: str,
) -> None:
    os.makedirs(rank_state_dict_dir, exist_ok=True)
    for rank, rank_state_dict in enumerate(rank_state_dicts):
        torch.save(rank_state_dict, _rank_state_dict_path(rank_state_dict_dir, rank))


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
    rank_state_dict_dir: str,
    src_rank: int,
    comm_mode: str,
    expert_exec_mode: str,
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
            comm_mode=comm_mode,
            expert_exec_mode=expert_exec_mode,
        ).to(device=device, dtype=torch_dtype)

        rank_state_dict = torch.load(
            _rank_state_dict_path(rank_state_dict_dir, rank),
            map_location="cpu",
        )
        state_dict = {
            key: value.to(dtype=torch_dtype)
            if torch.is_floating_point(value)
            else value
            for key, value in rank_state_dict.items()
        }
        layer.load_state_dict(state_dict, strict=True)
        layer.eval()

        num_tokens = hidden_states_cpu.reshape(-1, hidden_states_cpu.shape[-1]).shape[0]
        hidden_states = hidden_states_cpu.to(device=device, dtype=torch_dtype) if rank == src_rank else None

        def _forward_once(return_aux: bool):
            packed_forward_kwargs: dict[str, object] = {}
            control_plane_elapsed_s = 0.0
            grouped_metadata = None
            if comm_mode == "packed":
                packed_send_counts = torch.zeros(ep_size, dtype=torch.int64, device=device)
                packed_source_context = None
                grouped_counts = None
                if rank == src_rank:
                    if hidden_states is None:
                        raise ValueError("packed EP source rank 需要 hidden_states 输入")
                    packed_source_context = layer.prepare_packed_source_context(hidden_states)
                    packed_send_counts.copy_(
                        packed_source_context.layout.send_counts.to(device=device, dtype=torch.int64)
                    )
                    if expert_exec_mode == "grouped":
                        grouped_counts = build_grouped_local_expert_counts(
                            packed_source_context.layout,
                            ep_size=ep_size,
                            num_local_experts=layer.experts_per_rank,
                        ).to(device=device, dtype=torch.int64)
                dist.broadcast(packed_send_counts, src=src_rank, group=None)
                packed_control_plane, control_plane_elapsed_s = _prepare_packed_control_plane(
                    packed_send_counts=packed_send_counts,
                    ep_size=ep_size,
                    rank=rank,
                    src_rank=src_rank,
                    device=rank,
                )
                if expert_exec_mode == "grouped":
                    if grouped_counts is None:
                        grouped_counts = torch.zeros(
                            (ep_size, layer.experts_per_rank),
                            dtype=torch.int64,
                            device=device,
                        )
                    dist.broadcast(grouped_counts, src=src_rank, group=None)
                    grouped_metadata, grouped_metadata_elapsed_s = _prepare_grouped_metadata(
                        grouped_counts[rank],
                        local_expert_offset=rank * layer.experts_per_rank,
                    )
                    control_plane_elapsed_s += grouped_metadata_elapsed_s
                packed_forward_kwargs = {
                    "packed_source_context": packed_source_context,
                    "packed_control_plane": packed_control_plane,
                    "grouped_metadata": grouped_metadata,
                }
            return (
                layer(
                    hidden_states,
                    return_aux=return_aux,
                    num_tokens=num_tokens,
                    **packed_forward_kwargs,
                ),
                control_plane_elapsed_s,
            )

        with torch.no_grad():
            for _ in range(warmup):
                _ = _forward_once(return_aux=False)

            dist.barrier()
            start_time = None
            if measure_steady_state:
                torch.cuda.synchronize(rank)
                start_time = time.perf_counter()

            result = None
            control_plane_elapsed_local_s = 0.0
            for _ in range(runs):
                result, control_plane_elapsed_s = _forward_once(return_aux=True)
                if measure_steady_state and comm_mode == "packed":
                    control_plane_elapsed_local_s += control_plane_elapsed_s

            dist.barrier()
            elapsed_s = None
            if measure_steady_state:
                torch.cuda.synchronize(rank)
                assert start_time is not None
                elapsed_s = time.perf_counter() - start_time

        control_plane_elapsed_s = 0.0
        if measure_steady_state and comm_mode == "packed":
            control_plane_elapsed = torch.tensor(control_plane_elapsed_local_s, dtype=torch.float64, device=device)
            dist.all_reduce(control_plane_elapsed, op=dist.ReduceOp.MAX, group=None)
            control_plane_elapsed_s = float(control_plane_elapsed.item())

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
                    "control_plane_elapsed_s": control_plane_elapsed_s,
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
        rank_state_dicts: list[dict[str, torch.Tensor]] | None = None,
        src_rank: int = 0,
        comm_mode: str = "padded",
        expert_exec_mode: str = "naive",
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
        if rank_state_dicts is not None and len(rank_state_dicts) != ep_size:
            raise ValueError(
                f"rank_state_dicts 长度必须等于 ep_size={ep_size}，当前 {len(rank_state_dicts)}"
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.bias = bias
        self.ep_size = ep_size
        self.dtype = dtype
        self.src_rank = src_rank
        self.comm_mode = _validate_comm_mode(comm_mode)
        self.expert_exec_mode = _validate_expert_exec_mode(expert_exec_mode)
        if ep_size > 1 and self.expert_exec_mode == "grouped" and self.comm_mode != "packed":
            raise ValueError("distributed grouped expert execution 当前只支持 comm_mode='packed'")
        self.rank_state_dicts = None if rank_state_dicts is None else [
            {
                key: value.detach().cpu()
                for key, value in rank_state_dict.items()
            }
            for rank_state_dict in rank_state_dicts
        ]

    @classmethod
    def from_moe_layer(
        cls,
        layer: MoELayer,
        ep_size: int = 2,
        dtype: str = "float16",
        src_rank: int = 0,
        comm_mode: str = "padded",
        expert_exec_mode: str = "naive",
    ) -> "EPEngine":
        rank_state_dicts = shard_moe_state_dict(
            layer.state_dict(),
            num_experts=layer.num_experts,
            ep_size=ep_size,
        )
        return cls(
            hidden_size=layer.hidden_size,
            intermediate_size=layer.intermediate_size,
            num_experts=layer.num_experts,
            top_k=layer.top_k,
            bias=layer.router.gate.bias is not None,
            ep_size=ep_size,
            dtype=dtype,
            rank_state_dicts=rank_state_dicts,
            src_rank=src_rank,
            comm_mode=comm_mode,
            expert_exec_mode=expert_exec_mode,
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
        if self.rank_state_dicts is None:
            raise RuntimeError(
                "EPEngine.forward 需要 rank_state_dicts；请使用 from_moe_layer() 或显式传入"
            )

        temp_dir = tempfile.mkdtemp(prefix="mini_infer_ep_")
        result_file = os.path.join(temp_dir, "result.pt")
        rendezvous_file = os.path.join(temp_dir, "rendezvous")
        rank_state_dict_dir = os.path.join(temp_dir, "rank_state_dicts")

        try:
            _dump_rank_state_dicts(self.rank_state_dicts, rank_state_dict_dir)
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
                    rank_state_dict_dir,
                    self.src_rank,
                    self.comm_mode,
                    self.expert_exec_mode,
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
            shutil.rmtree(temp_dir, ignore_errors=True)

    def get_rank_state_dict(self, rank: int) -> dict[str, torch.Tensor]:
        """返回某个 rank 的 local expert shard state_dict。"""
        if self.rank_state_dicts is None:
            raise RuntimeError("当前 EPEngine 没有 rank_state_dicts")
        if rank < 0 or rank >= self.ep_size:
            raise ValueError(f"rank 必须落在 [0, ep_size) 内，当前 rank={rank}, ep_size={self.ep_size}")
        return self.rank_state_dicts[rank]

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
