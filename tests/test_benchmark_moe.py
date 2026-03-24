"""Phase 17 benchmark 口径测试。

覆盖：
- TP / EP 通信量公式
- synthetic hidden state 构造
- benchmark dry-run 所需的参数构造 helper
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_moe.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_moe", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_moe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_moe)


def test_compute_tp_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_tp_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        dtype_bytes=2,
    )
    assert bytes_per_layer == 512


def test_compute_ep_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_ep_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        top_k=2,
        dtype_bytes=2,
    )
    assert bytes_per_layer == 1024


def test_compute_ep_padded_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_ep_padded_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        top_k=2,
        dtype_bytes=2,
        ep_size=2,
    )
    assert bytes_per_layer == 2048


def test_build_comm_summary_contains_formulas() -> None:
    summary = benchmark_moe.build_comm_summary(
        num_tokens=32,
        hidden_size=64,
        top_k=2,
        dtype="float16",
        ep_size=2,
    )

    assert summary["tp_formula"] == "2 * num_tokens * hidden_size * dtype_bytes"
    assert summary["ep_ideal_formula"] == "2 * num_tokens * top_k * hidden_size * dtype_bytes"
    assert summary["ep_prototype_formula"] == "2 * ep_size * num_tokens * top_k * hidden_size * dtype_bytes"
    assert summary["ep_ideal_bytes_per_layer"] == summary["tp_bytes_per_layer"] * 2
    assert summary["ep_prototype_bytes_per_layer"] == summary["ep_ideal_bytes_per_layer"] * 2
    assert "padded all_to_all_single" in summary["ep_impl_note"]


def test_build_hidden_states_shape_and_dtype() -> None:
    hidden_states = benchmark_moe.build_hidden_states(
        batch_size=2,
        seq_len=3,
        hidden_size=8,
        dtype=torch.float32,
        device="cpu",
        seed=7,
    )

    assert hidden_states.shape == (2, 3, 8)
    assert hidden_states.dtype == torch.float32


def test_build_argparser_defaults() -> None:
    parser = benchmark_moe.build_argparser()
    args = parser.parse_args([])

    assert args.mode == "dense"
    assert args.compare is False
    assert args.ep_size == 2
    assert args.src_rank == 0
    assert args.top_k == 2
    assert args.hidden_size > 0


def test_resolve_dense_device_uses_src_rank_in_compare() -> None:
    parser = benchmark_moe.build_argparser()
    args = parser.parse_args(["--compare", "--device", "0", "--src-rank", "1"])

    dense_device = benchmark_moe.resolve_dense_device(args)

    assert dense_device == 1


def test_run_dry_run_does_not_instantiate_ep_engine(monkeypatch) -> None:
    class _FailingEngine:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run 不应实例化 EPEngine")

    monkeypatch.setattr(benchmark_moe, "EPEngine", _FailingEngine)
    args = benchmark_moe.build_argparser().parse_args(["--mode", "ep", "--dry-run"])

    result = benchmark_moe.run_dry_run(args)

    assert result["mode"] == "ep"
    assert result["comm"]["ep_prototype_bytes_per_layer"] > 0


def test_run_compare_benchmark_uses_shared_layer_and_inputs(monkeypatch) -> None:
    seen: dict[str, int] = {}

    def fake_dense(args, shared_layer=None, shared_hidden_states=None, device=None):
        assert shared_layer is not None and shared_hidden_states is not None
        assert device == args.src_rank
        seen["dense_weight_ptr"] = shared_layer.router.gate.weight.data_ptr()
        seen["dense_input_ptr"] = shared_hidden_states.data_ptr()
        return {
            "mode": "dense",
            "throughput_tok_s": 10.0,
            "note": "dense",
            "output": torch.tensor([[1.0]]),
            "expert_loads": torch.tensor([1, 2]),
            "expert_score_sums": torch.tensor([0.4, 0.6]),
        }

    def fake_ep(args, shared_layer=None, shared_hidden_states=None):
        assert shared_layer is not None and shared_hidden_states is not None
        seen["ep_weight_ptr"] = shared_layer.router.gate.weight.data_ptr()
        seen["ep_input_ptr"] = shared_hidden_states.data_ptr()
        return {
            "mode": "ep",
            "throughput_tok_s": 20.0,
            "note": "ep",
            "output": torch.tensor([[1.25]]),
            "send_counts": torch.tensor([2, 2]),
            "expert_loads": torch.tensor([1, 2]),
            "expert_score_sums": torch.tensor([0.4, 0.6]),
            "elapsed_s": 0.1,
        }

    monkeypatch.setattr(benchmark_moe, "run_dense_benchmark", fake_dense)
    monkeypatch.setattr(benchmark_moe, "run_ep_benchmark", fake_ep)

    args = benchmark_moe.build_argparser().parse_args(["--compare"])
    result = benchmark_moe.run_compare_benchmark(args)

    assert seen["dense_weight_ptr"] == seen["ep_weight_ptr"]
    assert seen["dense_input_ptr"] == seen["ep_input_ptr"]
    assert result["max_abs_diff"] == 0.25
