"""Phase 17-18 MoE 测试。

覆盖：
- TopKRouter 的 top-k 范围和分数归一化
- 路由统计在 half 输入下仍使用 fp32 累加
- dense MoELayer 的 weighted combine 数值正确性
- dense MoELayer 的路由统计
- dispatch / gather 重排与逆置换
- local expert shard 的 state_dict 切分与所有权
- dense MoELayer vs EPMoELayer 数值等价
- 2 卡 EPEngine 最小功能路径
- 2 卡 EPEngine 的 zero-send-count 边界
- SyntheticMoEConfig 参数校验
- SyntheticMoEModel 的前向形状、逐层 router 辅助输出和最小 CUDA fp16 路径
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from mini_infer import EPEngine
from mini_infer.moe_layer import (
    EPMoELayer,
    MoELayer,
    RouterOutput,
    TopKRouter,
    build_dispatch_layout,
    shard_moe_state_dict,
    summarize_routing,
)
from mini_infer.moe_model import SyntheticMoEConfig, SyntheticMoEModel


@dataclass
class _FixedRoute:
    expert_indices: torch.Tensor
    expert_weights: torch.Tensor


class _FixedRouter(nn.Module):
    def __init__(self, route: _FixedRoute) -> None:
        super().__init__()
        self.route = route

    def forward(self, x: torch.Tensor) -> RouterOutput:
        num_tokens = x.reshape(-1, x.shape[-1]).shape[0]
        assert num_tokens == self.route.expert_indices.shape[0]
        return RouterOutput(
            expert_indices=self.route.expert_indices,
            expert_weights=self.route.expert_weights,
            router_logits=torch.zeros(
                (num_tokens, int(self.route.expert_indices.max().item()) + 1),
                dtype=x.dtype,
                device=x.device,
            ),
        )


class _ScaleExpert(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class TestTopKRouter:
    def test_router_scores_are_renormalized(self):
        torch.manual_seed(0)
        router = TopKRouter(hidden_size=8, num_experts=6, top_k=2)
        x = torch.randn(7, 8)
        route = router(x)

        assert route.expert_indices.shape == (7, 2)
        assert route.expert_weights.shape == (7, 2)
        assert torch.allclose(route.expert_weights.sum(dim=-1), torch.ones(7), atol=1e-6)
        assert torch.all(route.expert_indices >= 0)
        assert torch.all(route.expert_indices < 6)

    def test_router_stats_match_topk_assignments(self):
        torch.manual_seed(1)
        router = TopKRouter(hidden_size=4, num_experts=5, top_k=2)
        route = router(torch.randn(3, 4))
        stats = summarize_routing(route, num_experts=5)

        assert int(stats.expert_loads.sum().item()) == 3 * 2
        assert torch.allclose(stats.expert_score_sums.sum(), torch.tensor(3.0), atol=1e-6)

    def test_router_stats_accumulate_scores_in_fp32(self):
        route = RouterOutput(
            expert_indices=torch.tensor([[0, 1], [1, 2]]),
            expert_weights=torch.tensor([[0.75, 0.25], [0.10, 0.90]], dtype=torch.float16),
            router_logits=torch.zeros(2, 3, dtype=torch.float16),
        )

        stats = summarize_routing(route, num_experts=3)

        assert stats.expert_score_sums.dtype == torch.float32
        expected = torch.zeros(3, dtype=torch.float32)
        expected.scatter_add_(0, route.expert_indices.reshape(-1), route.expert_weights.reshape(-1).float())
        assert torch.allclose(
            stats.expert_score_sums,
            expected,
            atol=1e-6,
        )


class TestDenseMoE:
    def test_dense_weighted_combine_matches_manual_reference(self):
        layer = MoELayer(hidden_size=2, intermediate_size=4, num_experts=3, top_k=2)
        layer.router = _FixedRouter(
            _FixedRoute(
                expert_indices=torch.tensor([[0, 1], [1, 2]]),
                expert_weights=torch.tensor([[0.75, 0.25], [0.10, 0.90]], dtype=torch.float32),
            )
        )
        layer.experts = nn.ModuleList([_ScaleExpert(1.0), _ScaleExpert(10.0), _ScaleExpert(-1.0)])

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out, _, stats = layer(x, return_router_stats=True)

        expected = torch.tensor([[3.25, 6.50], [0.30, 0.40]])
        assert torch.allclose(out, expected, atol=1e-6)
        assert stats.num_tokens == 2
        assert stats.top_k == 2
        assert torch.equal(stats.expert_loads.cpu(), torch.tensor([1, 2, 1]))

    def test_dense_moe_preserves_3d_shape(self):
        torch.manual_seed(2)
        layer = MoELayer(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2)
        x = torch.randn(2, 3, 8)
        out = layer(x)
        assert out.shape == x.shape


class TestDispatchAndEP:
    def test_shard_moe_state_dict_keeps_only_local_experts(self):
        dense = MoELayer(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2)

        rank_state_dicts = shard_moe_state_dict(dense.state_dict(), num_experts=4, ep_size=2)

        assert len(rank_state_dicts) == 2
        assert "router.gate.weight" in rank_state_dicts[0]
        assert "router.gate.weight" in rank_state_dicts[1]
        assert any(key.startswith("experts.0.") for key in rank_state_dicts[0])
        assert any(key.startswith("experts.1.") for key in rank_state_dicts[0])
        assert not any(key.startswith("experts.2.") for key in rank_state_dicts[0])
        assert not any(key.startswith("experts.3.") for key in rank_state_dicts[0])

    def test_ep_layer_owns_only_local_expert_shard(self):
        layer = EPMoELayer(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2, ep_size=2, rank=1)

        assert len(layer.experts) == 2
        assert list(layer.local_expert_ids()) == [2, 3]

    def test_dispatch_layout_groups_entries_by_rank(self):
        route = RouterOutput(
            expert_indices=torch.tensor([[0, 3], [1, 2]]),
            expert_weights=torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32),
            router_logits=torch.zeros(2, 4),
        )

        layout = build_dispatch_layout(route, num_experts=4, ep_size=2)

        assert torch.equal(layout.send_counts.cpu(), torch.tensor([2, 2]))
        assert torch.equal(layout.dest_ranks.cpu(), torch.tensor([0, 0, 1, 1]))
        assert torch.equal(layout.local_expert_ids.cpu(), torch.tensor([0, 1, 0, 1]))

    def test_dispatch_and_combine_recover_weighted_tokens(self):
        layer = EPMoELayer(hidden_size=2, intermediate_size=4, num_experts=4, top_k=2, ep_size=2)
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        route = RouterOutput(
            expert_indices=torch.tensor([[0, 3], [1, 2]]),
            expert_weights=torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.float32),
            router_logits=torch.zeros(2, 4),
        )

        dispatched, layout = layer.dispatch_tokens(x, route)
        combined = layer.combine_dispatched(
            dispatched_outputs=dispatched,
            layout=layout,
            num_tokens=2,
            dtype=x.dtype,
            device=x.device,
        )

        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        assert torch.allclose(combined, expected, atol=1e-6)

    def test_ep_layer_matches_dense_reference_cpu(self):
        torch.manual_seed(5)
        dense = MoELayer(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2)
        ep = EPMoELayer(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2, ep_size=1)
        ep.load_state_dict(dense.state_dict(), strict=True)

        x = torch.randn(2, 3, 8)
        out_dense = dense(x)
        out_ep, aux = ep(x, return_aux=True)

        assert torch.allclose(out_dense, out_ep, atol=1e-6, rtol=1e-6)
        assert int(aux.send_counts.sum().item()) == x.shape[0] * x.shape[1] * dense.top_k

    def test_ep_engine_rejects_rank_state_dict_length_mismatch(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("需要至少 2 张 CUDA GPU")

        shard = {"router.gate.weight": torch.zeros(4, 8)}
        with pytest.raises(ValueError, match="rank_state_dicts"):
            EPEngine(
                hidden_size=8,
                intermediate_size=16,
                num_experts=4,
                top_k=2,
                ep_size=2,
                dtype="float32",
                rank_state_dicts=[shard],
                src_rank=0,
            )

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要至少 2 张 CUDA GPU",
    )
    def test_ep_engine_matches_dense_reference_cuda(self):
        torch.manual_seed(6)
        dense = MoELayer(hidden_size=8, intermediate_size=16, num_experts=4, top_k=2).float().eval()
        x = torch.randn(2, 3, 8)

        with torch.no_grad():
            ref = dense.cuda()(x.cuda()).cpu()

        engine = EPEngine.from_moe_layer(dense, ep_size=2, dtype="float32", src_rank=1)
        out, aux = engine.forward(x, return_aux=True)
        bench = engine.benchmark_forward(x, warmup=0, runs=1)

        assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)
        assert int(aux["send_counts"].sum().item()) == x.shape[0] * x.shape[1] * dense.top_k
        assert int(aux["expert_loads"].sum().item()) == x.shape[0] * x.shape[1] * dense.top_k
        assert float(bench["elapsed_s"]) > 0.0

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="需要至少 2 张 CUDA GPU",
    )
    def test_ep_engine_handles_zero_send_count_cuda(self):
        dense = MoELayer(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            top_k=2,
            bias=True,
        ).float().eval()
        with torch.no_grad():
            dense.router.gate.weight.zero_()
            dense.router.gate.bias.copy_(torch.tensor([10.0, 9.0, -10.0, -11.0]))

        x = torch.randn(2, 3, 8)
        with torch.no_grad():
            ref = dense.to("cuda:1")(x.to("cuda:1")).cpu()

        engine = EPEngine.from_moe_layer(dense, ep_size=2, dtype="float32", src_rank=1)
        out, aux = engine.forward(x, return_aux=True)
        bench = engine.benchmark_forward(x, warmup=0, runs=1)

        assert torch.allclose(ref, out, atol=1e-4, rtol=1e-4)
        assert torch.equal(aux["send_counts"].cpu(), torch.tensor([x.shape[0] * x.shape[1] * dense.top_k, 0]))
        assert torch.equal(bench["send_counts"].cpu(), aux["send_counts"].cpu())
        assert float(bench["elapsed_s"]) > 0.0


class TestSyntheticMoEModel:
    def test_config_rejects_invalid_topk(self):
        with pytest.raises(ValueError, match="top_k"):
            SyntheticMoEConfig(num_experts=4, top_k=5)

    def test_synthetic_model_forward_and_router_aux(self):
        torch.manual_seed(3)
        config = SyntheticMoEConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_experts=4,
            top_k=2,
        )
        model = SyntheticMoEModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 5))

        logits, aux = model(input_ids, return_router_stats=True)

        assert logits.shape == (2, 5, config.vocab_size)
        assert len(aux) == config.num_layers
        for layer_aux in aux:
            assert int(layer_aux.stats.expert_loads.sum().item()) == input_ids.numel() * config.top_k
            assert torch.allclose(
                layer_aux.route.expert_weights.sum(dim=-1),
                torch.ones(input_ids.numel()),
                atol=1e-6,
            )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
    def test_synthetic_model_forward_cuda_fp16(self):
        torch.manual_seed(4)
        config = SyntheticMoEConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_layers=2,
            num_experts=4,
            top_k=2,
        )
        model = SyntheticMoEModel(config).cuda().half().eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 6), device="cuda")

        with torch.no_grad():
            logits, aux = model(input_ids, return_router_stats=True)

        assert logits.shape == (2, 6, config.vocab_size)
        assert logits.dtype == torch.float16
        assert len(aux) == config.num_layers
        assert aux[0].stats.expert_score_sums.dtype == torch.float32
        assert int(aux[0].stats.expert_loads.sum().item()) == input_ids.numel() * config.top_k
