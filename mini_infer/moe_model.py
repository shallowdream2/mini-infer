"""Phase 17：synthetic MoE 模型。

这个文件把 dense `MoELayer` 组装成最小可运行的 synthetic 模型，
用于单进程正确性验证，以及后续 EP 路径的对照 benchmark。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .moe_layer import MoELayer, RouterOutput, RoutingStats


@dataclass
class SyntheticMoEConfig:
    """Synthetic MoE 模型配置。"""

    vocab_size: int = 256
    hidden_size: int = 64
    intermediate_size: int = 128
    num_layers: int = 2
    num_experts: int = 8
    top_k: int = 2
    bias: bool = False
    layer_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        for field_name in [
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_experts",
            "top_k",
        ]:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} 必须 > 0，当前 {value}")
        if self.top_k > self.num_experts:
            raise ValueError(
                f"top_k 必须 <= num_experts，当前 top_k={self.top_k}, num_experts={self.num_experts}"
            )


@dataclass
class SyntheticMoEBlockAux:
    """单层 MoE block 的辅助输出。"""

    route: RouterOutput
    stats: RoutingStats


class SyntheticMoEBlock(nn.Module):
    """最小 residual + norm + MoE block。"""

    def __init__(self, config: SyntheticMoEConfig) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.moe = MoELayer(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            bias=config.bias,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_router_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, SyntheticMoEBlockAux]:
        normed = self.input_norm(hidden_states)
        if not return_router_stats:
            return hidden_states + self.moe(normed)

        moe_out, route, stats = self.moe(normed, return_router_stats=True)
        return hidden_states + moe_out, SyntheticMoEBlockAux(route=route, stats=stats)


class SyntheticMoEModel(nn.Module):
    """最小可运行的 synthetic MoE 模型。"""

    def __init__(self, config: SyntheticMoEConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([SyntheticMoEBlock(config) for _ in range(config.num_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_router_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[SyntheticMoEBlockAux]]:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"input_ids 必须是整型 token ids，当前 dtype={input_ids.dtype}")

        hidden_states = self.embed_tokens(input_ids)
        layer_aux: list[SyntheticMoEBlockAux] = []
        for layer in self.layers:
            if not return_router_stats:
                hidden_states = layer(hidden_states)
                continue
            hidden_states, aux = layer(hidden_states, return_router_stats=True)
            layer_aux.append(aux)

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        if not return_router_stats:
            return logits
        return logits, layer_aux


__all__ = [
    "SyntheticMoEBlock",
    "SyntheticMoEBlockAux",
    "SyntheticMoEConfig",
    "SyntheticMoEModel",
]
