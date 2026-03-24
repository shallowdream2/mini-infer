"""Phase 17：synthetic MoE 层与 Expert Parallel 核心逻辑。

这个文件提供两条路径：
- `MoELayer`：单进程 dense 参考实现，作为数值 oracle
- `EPMoELayer`：按 rank 对 expert 分片，并通过 dispatch / gather
  组织 all-to-all 所需的数据布局
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RouterOutput:
    """Router 的 top-k 输出。"""

    expert_indices: torch.Tensor
    expert_weights: torch.Tensor
    router_logits: torch.Tensor


@dataclass
class RoutingStats:
    """MoE 路由统计，供测试和 benchmark 解释使用。"""

    expert_loads: torch.Tensor
    expert_score_sums: torch.Tensor
    num_tokens: int
    top_k: int


@dataclass
class DispatchLayout:
    """dispatch 后的 token 重排信息。"""

    token_indices: torch.Tensor
    expert_ids: torch.Tensor
    local_expert_ids: torch.Tensor
    weights: torch.Tensor
    dest_ranks: torch.Tensor
    send_counts: torch.Tensor


@dataclass
class EPForwardAux:
    """EP 前向的辅助输出，供测试和 benchmark 解释使用。"""

    route: RouterOutput
    routing_stats: RoutingStats
    send_counts: torch.Tensor


def _flatten_tokens(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    """把 [B,S,H] 或 [T,H] 统一展平到 [N,H]。"""
    if x.ndim < 2:
        raise ValueError(f"MoE 输入至少需要 2 维 [tokens, hidden]，当前 shape={tuple(x.shape)}")
    hidden_size = x.shape[-1]
    return x.reshape(-1, hidden_size), tuple(x.shape[:-1])


def _validate_ep_partition(num_experts: int, ep_size: int) -> int:
    if ep_size <= 0:
        raise ValueError(f"ep_size 必须 > 0，当前 {ep_size}")
    if num_experts % ep_size != 0:
        raise ValueError(f"num_experts={num_experts} 必须能被 ep_size={ep_size} 整除")
    return num_experts // ep_size


def summarize_routing(route: RouterOutput, num_experts: int) -> RoutingStats:
    """统计每个 expert 被分到多少 token，以及累计分数。"""
    expert_indices = route.expert_indices.reshape(-1)
    # 统计口径固定用 fp32，避免 half/bfloat16 在长 token 集合上累积误差过大。
    expert_weights = route.expert_weights.reshape(-1).float()
    expert_loads = torch.bincount(expert_indices, minlength=num_experts)
    expert_score_sums = torch.zeros(
        num_experts,
        dtype=torch.float32,
        device=route.expert_weights.device,
    )
    expert_score_sums.scatter_add_(0, expert_indices, expert_weights)
    return RoutingStats(
        expert_loads=expert_loads,
        expert_score_sums=expert_score_sums,
        num_tokens=route.expert_indices.shape[0],
        top_k=route.expert_indices.shape[1],
    )


def build_dispatch_layout(
    route: RouterOutput,
    num_experts: int,
    ep_size: int,
) -> DispatchLayout:
    """把 top-k 路由结果重排成按 rank / expert 分桶的 dispatch 布局。"""
    experts_per_rank = _validate_ep_partition(num_experts, ep_size)
    num_tokens, top_k = route.expert_indices.shape

    token_indices = torch.arange(
        num_tokens,
        device=route.expert_indices.device,
        dtype=torch.long,
    )
    token_indices = token_indices.unsqueeze(1).expand(-1, top_k).reshape(-1)
    expert_ids = route.expert_indices.reshape(-1).long()
    weights = route.expert_weights.reshape(-1)
    dest_ranks = torch.div(expert_ids, experts_per_rank, rounding_mode="floor")
    local_expert_ids = expert_ids - dest_ranks * experts_per_rank

    sort_key = dest_ranks * num_experts + expert_ids
    order = torch.argsort(sort_key)

    token_indices = token_indices.index_select(0, order)
    expert_ids = expert_ids.index_select(0, order)
    local_expert_ids = local_expert_ids.index_select(0, order)
    weights = weights.index_select(0, order)
    dest_ranks = dest_ranks.index_select(0, order)
    send_counts = torch.bincount(dest_ranks, minlength=ep_size)

    return DispatchLayout(
        token_indices=token_indices,
        expert_ids=expert_ids,
        local_expert_ids=local_expert_ids,
        weights=weights,
        dest_ranks=dest_ranks,
        send_counts=send_counts,
    )


class TopKRouter(nn.Module):
    """最小可用的 top-k router。"""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size 必须 > 0，当前 {hidden_size}")
        if num_experts <= 0:
            raise ValueError(f"num_experts 必须 > 0，当前 {num_experts}")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError(f"top_k 必须落在 [1, num_experts]，当前 top_k={top_k}, num_experts={num_experts}")

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=bias)

    def forward(self, x: torch.Tensor) -> RouterOutput:
        flat_x, _ = _flatten_tokens(x)
        router_logits = self.gate(flat_x)
        topk_logits, expert_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        expert_weights = torch.softmax(topk_logits.float(), dim=-1).to(router_logits.dtype)
        return RouterOutput(
            expert_indices=expert_indices,
            expert_weights=expert_weights,
            router_logits=router_logits,
        )


class MoEFFNExpert(nn.Module):
    """模仿 Qwen MLP 的单 expert FFN。"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size 必须 > 0，当前 {hidden_size}")
        if intermediate_size <= 0:
            raise ValueError(f"intermediate_size 必须 > 0，当前 {intermediate_size}")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)


class MoELayer(nn.Module):
    """dense 参考 MoE 层。

    这里显式对每个 expert 做 token 选择、专家前向和 weighted combine。
    这条路径刻意不做 dispatch 优化，优先保证后续 EP 数值对齐时有稳定 oracle。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
        )
        self.experts = nn.ModuleList(
            [
                MoEFFNExpert(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        return_router_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RouterOutput, RoutingStats]:
        flat_x, leading_shape = _flatten_tokens(x)
        route = self.router(flat_x)

        combined = torch.zeros(
            (flat_x.shape[0], self.hidden_size),
            device=flat_x.device,
            dtype=flat_x.dtype,
        )

        for expert_id, expert in enumerate(self.experts):
            token_idx, slot_idx = torch.where(route.expert_indices == expert_id)
            if token_idx.numel() == 0:
                continue
            expert_in = flat_x.index_select(0, token_idx)
            expert_out = expert(expert_in)
            expert_weight = route.expert_weights[token_idx, slot_idx].to(expert_out.dtype).unsqueeze(-1)
            combined.index_add_(0, token_idx, expert_out * expert_weight)

        output = combined.reshape(*leading_shape, self.hidden_size)
        if not return_router_stats:
            return output

        stats = summarize_routing(route, num_experts=self.num_experts)
        return output, route, stats


class EPMoELayer(nn.Module):
    """Expert Parallel MoE 层。

    当前实现优先保证 dispatch / gather 数学路径和 2 卡 all-to-all 生命周期正确。
    为了降低实现复杂度，每个 rank 当前仍持有完整 expert 权重，但只执行本 rank
    负责的 expert 子集；后续若需要，再继续把权重物理分片收紧到真正的内存节省版本。
    distributed 路径当前使用 padded `all_to_all_single` + valid mask，避免在 per-layer
    forward 里把 CUDA count tensor 同步回 CPU。benchmark 会显式区分理想 EP bytes 和
    当前 prototype padded bytes，防止把两者混为一谈。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int = 2,
        ep_size: int = 2,
        rank: int = 0,
        bias: bool = False,
        dist_group: Optional[dist.ProcessGroup] = None,
        src_rank: int = 0,
    ) -> None:
        super().__init__()
        if src_rank < 0 or src_rank >= ep_size:
            raise ValueError(f"src_rank 必须落在 [0, ep_size) 内，当前 src_rank={src_rank}, ep_size={ep_size}")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.rank = rank
        self.dist_group = dist_group
        self.src_rank = src_rank
        self.experts_per_rank = _validate_ep_partition(num_experts, ep_size)

        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            bias=bias,
        )
        self.experts = nn.ModuleList(
            [
                MoEFFNExpert(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    bias=bias,
                )
                for _ in range(num_experts)
            ]
        )

    def local_expert_ids(self, rank: Optional[int] = None) -> range:
        owner_rank = self.rank if rank is None else rank
        start = owner_rank * self.experts_per_rank
        return range(start, start + self.experts_per_rank)

    def dispatch_tokens(
        self,
        x: torch.Tensor,
        route: RouterOutput,
    ) -> tuple[torch.Tensor, DispatchLayout]:
        """按 rank / expert 顺序重排 token 副本。"""
        flat_x, _ = _flatten_tokens(x)
        layout = build_dispatch_layout(
            route=route,
            num_experts=self.num_experts,
            ep_size=self.ep_size,
        )
        dispatched = flat_x.index_select(0, layout.token_indices)
        return dispatched, layout

    def combine_dispatched(
        self,
        dispatched_outputs: torch.Tensor,
        layout: DispatchLayout,
        num_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """把 expert 输出按原 token 顺序 gather 回去并加权求和。"""
        combined = torch.zeros(
            (num_tokens, self.hidden_size),
            dtype=dtype,
            device=device,
        )
        if dispatched_outputs.numel() == 0:
            return combined
        weights = layout.weights.to(dispatched_outputs.dtype).unsqueeze(-1)
        combined.index_add_(0, layout.token_indices, dispatched_outputs * weights)
        return combined

    def _apply_experts(
        self,
        dispatched_x: torch.Tensor,
        expert_ids: torch.Tensor,
        valid_expert_ids: Optional[range] = None,
    ) -> torch.Tensor:
        outputs = torch.zeros(
            (dispatched_x.shape[0], self.hidden_size),
            device=dispatched_x.device,
            dtype=dispatched_x.dtype,
        )
        if dispatched_x.numel() == 0:
            return outputs

        expert_id_iter = range(self.num_experts) if valid_expert_ids is None else valid_expert_ids
        for expert_id in expert_id_iter:
            expert_mask = torch.where(expert_ids == expert_id)[0]
            if expert_mask.numel() == 0:
                continue
            expert_in = dispatched_x.index_select(0, expert_mask)
            expert_out = self.experts[expert_id](expert_in)
            outputs.index_copy_(0, expert_mask, expert_out)
        return outputs

    def _forward_local(
        self,
        x: torch.Tensor,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, EPForwardAux]:
        flat_x, leading_shape = _flatten_tokens(x)
        route = self.router(flat_x)
        dispatched_x, layout = self.dispatch_tokens(flat_x, route)
        dispatched_out = self._apply_experts(dispatched_x, layout.expert_ids)
        combined = self.combine_dispatched(
            dispatched_outputs=dispatched_out,
            layout=layout,
            num_tokens=flat_x.shape[0],
            dtype=flat_x.dtype,
            device=flat_x.device,
        )
        output = combined.reshape(*leading_shape, self.hidden_size)
        if not return_aux:
            return output
        return output, EPForwardAux(
            route=route,
            routing_stats=summarize_routing(route, self.num_experts),
            send_counts=layout.send_counts,
        )

    def _forward_distributed(
        self,
        x: Optional[torch.Tensor],
        num_tokens: int,
        return_aux: bool = False,
    ) -> Optional[torch.Tensor] | tuple[Optional[torch.Tensor], Optional[EPForwardAux]]:
        if not dist.is_initialized():
            raise RuntimeError("EPMoELayer 的 distributed 路径要求已初始化 torch.distributed")

        world_size = dist.get_world_size(self.dist_group)
        rank = dist.get_rank(self.dist_group)
        if world_size != self.ep_size:
            raise RuntimeError(f"dist world_size={world_size} 与 ep_size={self.ep_size} 不一致")

        chunk_size = num_tokens * self.top_k
        hidden_dtype = self.experts[0].gate_proj.weight.dtype
        device = self.experts[0].gate_proj.weight.device
        chunk_splits = [chunk_size] * world_size

        route = None
        layout = None
        leading_shape: tuple[int, ...] = ()
        send_hidden = torch.zeros(
            (world_size * chunk_size, self.hidden_size),
            dtype=hidden_dtype,
            device=device,
        )
        send_expert_ids = torch.full(
            (world_size * chunk_size,),
            -1,
            dtype=torch.int64,
            device=device,
        )
        send_valid = torch.zeros(
            (world_size * chunk_size,),
            dtype=torch.int32,
            device=device,
        )
        if rank == self.src_rank:
            if x is None:
                raise ValueError("source rank 的 EPMoELayer.forward 需要提供输入 x")
            flat_x, leading_shape = _flatten_tokens(x)
            route = self.router(flat_x)
            dispatched_x, layout = self.dispatch_tokens(flat_x, route)

            for dest_rank in range(world_size):
                mask = torch.where(layout.dest_ranks == dest_rank)[0]
                if mask.numel() == 0:
                    continue
                start = dest_rank * chunk_size
                end = start + mask.shape[0]
                send_hidden[start:end] = dispatched_x.index_select(0, mask)
                send_expert_ids[start:end] = layout.expert_ids.index_select(0, mask)
                send_valid[start:end] = 1

        recv_hidden = torch.empty_like(send_hidden)
        recv_expert_ids = torch.empty_like(send_expert_ids)
        recv_valid = torch.empty_like(send_valid)

        dist.all_to_all_single(
            recv_hidden,
            send_hidden,
            output_split_sizes=chunk_splits,
            input_split_sizes=chunk_splits,
            group=self.dist_group,
        )
        dist.all_to_all_single(
            recv_expert_ids,
            send_expert_ids,
            output_split_sizes=chunk_splits,
            input_split_sizes=chunk_splits,
            group=self.dist_group,
        )
        dist.all_to_all_single(
            recv_valid,
            send_valid,
            output_split_sizes=chunk_splits,
            input_split_sizes=chunk_splits,
            group=self.dist_group,
        )

        source_start = self.src_rank * chunk_size
        source_end = source_start + chunk_size
        valid_mask = recv_valid[source_start:source_end].bool()
        recv_hidden_local = recv_hidden[source_start:source_end][valid_mask]
        recv_expert_ids_local = recv_expert_ids[source_start:source_end][valid_mask]

        local_outputs = self._apply_experts(
            dispatched_x=recv_hidden_local,
            expert_ids=recv_expert_ids_local,
            valid_expert_ids=self.local_expert_ids(rank),
        )

        send_back_hidden = torch.zeros_like(send_hidden)
        send_back_valid = torch.zeros_like(send_valid)
        if local_outputs.shape[0] > 0:
            return_start = self.src_rank * chunk_size
            return_end = return_start + local_outputs.shape[0]
            send_back_hidden[return_start:return_end] = local_outputs
            send_back_valid[return_start:return_end] = 1

        recv_back_hidden = torch.empty_like(send_back_hidden)
        recv_back_valid = torch.empty_like(send_back_valid)
        dist.all_to_all_single(
            recv_back_hidden,
            send_back_hidden,
            output_split_sizes=chunk_splits,
            input_split_sizes=chunk_splits,
            group=self.dist_group,
        )
        dist.all_to_all_single(
            recv_back_valid,
            send_back_valid,
            output_split_sizes=chunk_splits,
            input_split_sizes=chunk_splits,
            group=self.dist_group,
        )

        if rank != self.src_rank:
            if not return_aux:
                return None
            return None, None

        assert x is not None
        assert route is not None and layout is not None
        returned_chunks: list[torch.Tensor] = []
        for peer_rank in range(world_size):
            start = peer_rank * chunk_size
            end = start + chunk_size
            peer_valid = recv_back_valid[start:end].bool()
            returned_chunks.append(recv_back_hidden[start:end][peer_valid])

        if returned_chunks:
            dispatched_out = torch.cat(returned_chunks, dim=0)
        else:
            dispatched_out = torch.empty((0, self.hidden_size), dtype=hidden_dtype, device=device)

        combined = self.combine_dispatched(
            dispatched_outputs=dispatched_out,
            layout=layout,
            num_tokens=num_tokens,
            dtype=x.dtype,
            device=x.device,
        )
        output = combined.reshape(*leading_shape, self.hidden_size)
        if not return_aux:
            return output
        return output, EPForwardAux(
            route=route,
            routing_stats=summarize_routing(route, self.num_experts),
            send_counts=layout.send_counts,
        )

    def forward(
        self,
        x: Optional[torch.Tensor],
        return_aux: bool = False,
        num_tokens: Optional[int] = None,
    ) -> Optional[torch.Tensor] | tuple[Optional[torch.Tensor], Optional[EPForwardAux]]:
        if self.dist_group is None and not dist.is_initialized():
            if x is None:
                raise ValueError("local EPMoELayer.forward 需要输入 x")
            return self._forward_local(x, return_aux=return_aux)

        if num_tokens is None:
            if x is None:
                raise ValueError("distributed EPMoELayer.forward 在 x=None 时必须提供 num_tokens")
            flat_x, _ = _flatten_tokens(x)
            num_tokens = flat_x.shape[0]
        return self._forward_distributed(x, num_tokens=num_tokens, return_aux=return_aux)


__all__ = [
    "DispatchLayout",
    "EPForwardAux",
    "EPMoELayer",
    "MoEFFNExpert",
    "MoELayer",
    "RouterOutput",
    "RoutingStats",
    "TopKRouter",
    "build_dispatch_layout",
    "summarize_routing",
]
