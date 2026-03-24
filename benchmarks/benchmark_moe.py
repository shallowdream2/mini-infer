"""Phase 17：synthetic MoE / Expert Parallel benchmark。

对比对象：
- dense `MoELayer`
- `EPEngine`（2 卡，all-to-all + expert dispatch/gather）

输出内容：
- dense / EP token throughput
- per-rank send_counts 与 per-expert token load
- TP vs EP 通信量公式

当前口径：
- 输入是 synthetic hidden states，不接真实 HuggingFace 权重
- `--mode ep` 的正式计时在 worker 内部完成，默认排除 `mp.spawn` 与进程组初始化开销
- `--compare` 使用同一组权重和同一组 hidden states 对比 dense / EP
- 通信结果同时区分：
  - ideal EP hidden-state bytes（按真实 token 副本数估算）
  - current padded prototype bytes（按当前 `all_to_all_single` fixed chunk 实现估算）
- `--dry-run` 只验证参数构造、通信量公式和 benchmark 主流程
"""

from __future__ import annotations

import argparse
import time

import torch

from mini_infer.ep_engine import EPEngine
from mini_infer.moe_layer import MoELayer
from mini_infer.moe_model import SyntheticMoEConfig


_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def build_hidden_states(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
) -> torch.Tensor:
    """构造可复现的 synthetic hidden states。"""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)


def build_shared_layer(args: argparse.Namespace) -> MoELayer:
    """构造 compare/dense/ep 共用的 base layer。"""
    torch.manual_seed(args.seed)
    return MoELayer(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        bias=args.bias,
    ).float().eval()


def build_shared_hidden_states(args: argparse.Namespace) -> torch.Tensor:
    """构造 compare/dense/ep 共用的 CPU hidden states。"""
    return build_hidden_states(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        dtype=torch.float32,
        device="cpu",
        seed=args.seed + 1,
    )


def resolve_dense_device(args: argparse.Namespace) -> int:
    """compare 模式下 dense 与 EP 共享同一 source device 口径。"""
    return args.src_rank if args.compare else args.device


def compute_tp_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    dtype_bytes: int,
) -> int:
    """粗略估算 dense TP 在一层上的通信量。"""
    return 2 * num_tokens * hidden_size * dtype_bytes


def compute_ep_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype_bytes: int,
) -> int:
    """估算 ideal EP 的 dispatch + gather hidden-state 通信量。"""
    return 2 * num_tokens * top_k * hidden_size * dtype_bytes


def compute_ep_padded_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype_bytes: int,
    ep_size: int,
) -> int:
    """估算当前 padded EP prototype 的 hidden-state 通信量。"""
    return 2 * ep_size * num_tokens * top_k * hidden_size * dtype_bytes


def build_comm_summary(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype: str,
    ep_size: int,
) -> dict[str, object]:
    dtype_bytes = torch.tensor([], dtype=_DTYPE_MAP[dtype]).element_size()
    tp_bytes = compute_tp_bytes_per_layer(num_tokens, hidden_size, dtype_bytes)
    ep_ideal_bytes = compute_ep_bytes_per_layer(num_tokens, hidden_size, top_k, dtype_bytes)
    ep_prototype_bytes = compute_ep_padded_bytes_per_layer(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype_bytes=dtype_bytes,
        ep_size=ep_size,
    )
    return {
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "tp_bytes_per_layer": tp_bytes,
        "ep_ideal_bytes_per_layer": ep_ideal_bytes,
        "ep_prototype_bytes_per_layer": ep_prototype_bytes,
        "tp_formula": "2 * num_tokens * hidden_size * dtype_bytes",
        "ep_ideal_formula": "2 * num_tokens * top_k * hidden_size * dtype_bytes",
        "ep_prototype_formula": "2 * ep_size * num_tokens * top_k * hidden_size * dtype_bytes",
        "ep_impl_note": (
            "current EPMoELayer uses padded all_to_all_single chunks to avoid per-layer "
            "host sync; hidden-state bytes only, expert-id/valid metadata excluded"
        ),
    }


def build_dense_layer(
    args: argparse.Namespace,
    shared_layer: MoELayer | None = None,
    device: int | None = None,
) -> MoELayer:
    layer = MoELayer(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        bias=args.bias,
    )
    if shared_layer is not None:
        layer.load_state_dict(shared_layer.state_dict(), strict=True)
    dense_device = args.device if device is None else device
    return layer.to(device=f"cuda:{dense_device}", dtype=_DTYPE_MAP[args.dtype]).eval()


def run_dense_benchmark(
    args: argparse.Namespace,
    shared_layer: MoELayer | None = None,
    shared_hidden_states: torch.Tensor | None = None,
    device: int | None = None,
) -> dict[str, object]:
    dense_device = args.device if device is None else device
    layer = build_dense_layer(args, shared_layer=shared_layer, device=dense_device)
    hidden_states_cpu = shared_hidden_states if shared_hidden_states is not None else build_shared_hidden_states(args)
    hidden_states = hidden_states_cpu.to(device=f"cuda:{dense_device}", dtype=_DTYPE_MAP[args.dtype])

    aux = None
    output_cpu = None
    with torch.no_grad():
        for _ in range(args.warmup):
            layer(hidden_states)
        torch.cuda.synchronize(device=dense_device)
        t0 = time.perf_counter()
        output = None
        stats = None
        for _ in range(args.runs):
            output, _, stats = layer(hidden_states, return_router_stats=True)
        torch.cuda.synchronize(device=dense_device)
        elapsed = time.perf_counter() - t0
        if output is not None and stats is not None:
            aux = {
                "expert_loads": stats.expert_loads.cpu(),
                "expert_score_sums": stats.expert_score_sums.cpu(),
            }
            output_cpu = output.cpu()

    num_tokens = args.batch_size * args.seq_len * args.runs
    throughput = num_tokens / elapsed
    result = {
        "mode": "dense",
        "throughput_tok_s": throughput,
        "note": f"single GPU dense MoELayer on cuda:{dense_device}",
        "output": output_cpu,
    }
    if aux is not None:
        result.update(aux)
    return result


def run_ep_benchmark(
    args: argparse.Namespace,
    shared_layer: MoELayer | None = None,
    shared_hidden_states: torch.Tensor | None = None,
) -> dict[str, object]:
    dense_layer = shared_layer if shared_layer is not None else build_shared_layer(args)
    engine = EPEngine.from_moe_layer(
        dense_layer,
        ep_size=args.ep_size,
        dtype=args.dtype,
        src_rank=args.src_rank,
    )
    hidden_states = shared_hidden_states if shared_hidden_states is not None else build_shared_hidden_states(args)

    bench = engine.benchmark_forward(
        hidden_states,
        warmup=args.warmup,
        runs=args.runs,
    )
    elapsed = float(bench["elapsed_s"])

    num_tokens = args.batch_size * args.seq_len * args.runs
    throughput = num_tokens / elapsed
    result = {
        "mode": "ep",
        "throughput_tok_s": throughput,
        "note": "2-GPU EPEngine steady-state worker timing; spawn/init excluded",
        "output": bench["output"],
        "send_counts": bench["send_counts"],
        "expert_loads": bench["expert_loads"],
        "expert_score_sums": bench["expert_score_sums"],
        "elapsed_s": elapsed,
    }
    return result


def run_compare_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """用同一组权重和 hidden states 对比 dense vs EP。"""
    shared_layer = build_shared_layer(args)
    shared_hidden_states = build_shared_hidden_states(args)
    dense_device = resolve_dense_device(args)
    dense_result = run_dense_benchmark(
        args,
        shared_layer=shared_layer,
        shared_hidden_states=shared_hidden_states,
        device=dense_device,
    )
    ep_result = run_ep_benchmark(
        args,
        shared_layer=shared_layer,
        shared_hidden_states=shared_hidden_states,
    )
    assert dense_result["output"] is not None and ep_result["output"] is not None
    max_abs_diff = (
        dense_result["output"].float() - ep_result["output"].float()
    ).abs().max().item()
    return {
        "dense": dense_result,
        "ep": ep_result,
        "max_abs_diff": float(max_abs_diff),
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, object]:
    """只做参数和通信量口径验证，不依赖 GPU。"""
    cfg = SyntheticMoEConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    _ = build_shared_layer(args)
    _ = build_shared_hidden_states(args)
    comm = build_comm_summary(
        num_tokens=args.batch_size * args.seq_len,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        dtype=args.dtype,
        ep_size=args.ep_size,
    )
    return {
        "mode": "compare" if args.compare else args.mode,
        "comm": comm,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 17 synthetic MoE / EP benchmark")
    parser.add_argument("--mode", choices=["dense", "ep"], default="dense")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--dtype", choices=sorted(_DTYPE_MAP), default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--src-rank", type=int, default=0)
    parser.add_argument("--bias", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.dry_run:
        dry_run = run_dry_run(args)
        comm = dry_run["comm"]
        print("=== Phase 17 benchmark dry-run ===")
        print(f"mode={dry_run['mode']}")
        print(
            f"batch_size={args.batch_size}, seq_len={args.seq_len}, hidden={args.hidden_size}, "
            f"intermediate={args.intermediate_size}, num_experts={args.num_experts}, top_k={args.top_k}"
        )
        print(f"tp_formula={comm['tp_formula']}")
        print(f"ep_ideal_formula={comm['ep_ideal_formula']}")
        print(f"ep_prototype_formula={comm['ep_prototype_formula']}")
        print(f"tp_bytes_per_layer={comm['tp_bytes_per_layer']}")
        print(f"ep_ideal_bytes_per_layer={comm['ep_ideal_bytes_per_layer']}")
        print(f"ep_prototype_bytes_per_layer={comm['ep_prototype_bytes_per_layer']}")
        print(f"ep_impl_note={comm['ep_impl_note']}")
        print("dry_run=ok")
        return

    comm = build_comm_summary(
        num_tokens=args.batch_size * args.seq_len,
        hidden_size=args.hidden_size,
        top_k=args.top_k,
        dtype=args.dtype,
        ep_size=args.ep_size,
    )
    if args.compare:
        result = run_compare_benchmark(args)
        dense = result["dense"]
        ep = result["ep"]
        print("=== Phase 17 benchmark (compare) ===")
        print(f"dense_throughput_tok_s={dense['throughput_tok_s']:.2f}")
        print(f"ep_throughput_tok_s={ep['throughput_tok_s']:.2f}")
        print(f"max_abs_diff={result['max_abs_diff']:.6f}")
        print(f"dense_note={dense['note']}")
        print(f"ep_note={ep['note']}")
        print(f"ep_send_counts={ep['send_counts'].tolist()}")
        print(f"dense_expert_loads={dense['expert_loads'].tolist()}")
        print(f"ep_expert_loads={ep['expert_loads'].tolist()}")
        print(
            f"ep_expert_score_sums={[round(float(v), 4) for v in ep['expert_score_sums']]}"
        )
    else:
        if args.mode == "dense":
            result = run_dense_benchmark(args)
        else:
            result = run_ep_benchmark(args)

        print(f"=== Phase 17 benchmark ({result['mode']}) ===")
        print(f"throughput_tok_s={result['throughput_tok_s']:.2f}")
        print(f"note={result['note']}")
        if "send_counts" in result:
            print(f"send_counts={result['send_counts'].tolist()}")
        if "expert_loads" in result:
            print(f"expert_loads={result['expert_loads'].tolist()}")
        if "expert_score_sums" in result:
            print(f"expert_score_sums={[round(float(v), 4) for v in result['expert_score_sums']]}")
    print(f"tp_formula={comm['tp_formula']}")
    print(f"ep_ideal_formula={comm['ep_ideal_formula']}")
    print(f"ep_prototype_formula={comm['ep_prototype_formula']}")
    print(f"tp_bytes_per_layer={comm['tp_bytes_per_layer']}")
    print(f"ep_ideal_bytes_per_layer={comm['ep_ideal_bytes_per_layer']}")
    print(f"ep_prototype_bytes_per_layer={comm['ep_prototype_bytes_per_layer']}")
    print(f"ep_impl_note={comm['ep_impl_note']}")


if __name__ == "__main__":
    main()
