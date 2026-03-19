"""
benchmarks/profile_decode.py — Phase 5 Profiling

对 mini-infer decode_batch 内部的关键操作进行 torch.profiler 分析，输出：
  - gather_batch_kv：从 KV block tensor 聚合当前 batch 的 KV（Phase 3 向量化版本）
  - model_forward：Transformer batch forward
  - write_decode_kv：将新 token 的 KV 写回 block tensor

用法：
    HF_HUB_OFFLINE=1 python benchmarks/profile_decode.py \\
        --model /path/to/model \\
        --batch-size 4 \\
        --decode-steps 20

说明：
  - record_function 标签在 model_runner.py 的 decode_batch 中定义
  - profiler 未激活时这些标签是 no-op，不影响正常推理性能
  - 预热步骤不计入 profiler，避免 CUDA 内核首次编译的冷启动影响
  - generate() 同时包含 prefill 和 decode 两个阶段；prefill 的 model forward 没有
    "model_forward" 标签，因此统计结果（count / 时间）仅反映 decode 步骤
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from mini_infer import LLMEngine
from mini_infer.config import EngineConfig

_PROMPTS = [
    "The future of large language model inference is",
    "Explain how paged attention works in detail:",
    "The key difference between data parallelism and tensor parallelism is",
    "In machine learning, gradient descent is used to",
    "Transformer architecture consists of attention layers that",
    "The main challenge in deploying LLMs at scale is",
    "Memory bandwidth is the primary bottleneck during",
    "Continuous batching improves throughput by",
]

# decode_batch 中三个被 record_function 标注的关键操作
_KEY_OPS = {"gather_batch_kv", "model_forward", "write_decode_kv"}


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer decode profiler")
    parser.add_argument("--model", required=True, help="模型路径或 HuggingFace ID")
    parser.add_argument(
        "--batch-size", type=int, default=4, choices=[1, 2, 4, 8],
        help="并发请求数（default: 4）",
    )
    parser.add_argument(
        "--decode-steps", type=int, default=20,
        help="profile 阶段的 max_new_tokens，即 decode 步数（default: 20）",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="目标 GPU（default: cuda:0）",
    )
    args = parser.parse_args()

    config = EngineConfig(
        model_name=args.model,
        device=args.device,
        dtype="float16",
        num_gpu_blocks=2048,
        block_size=16,
    )

    print(f"[profiler] 加载模型: {args.model}")
    engine = LLMEngine(config)
    prompts = (_PROMPTS * 4)[: args.batch_size]

    # ── 预热：确保 CUDA 内核已编译、显存分配稳定 ────────────────────────────
    print("[profiler] 预热（warmup）中...")
    engine.generate(prompts, max_new_tokens=5)
    torch.cuda.synchronize(args.device)

    # ── Profiling ────────────────────────────────────────────────────────────
    print(f"[profiler] 开始 profiling：batch={args.batch_size}, decode_steps={args.decode_steps}")
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
    ) as prof:
        engine.generate(prompts, max_new_tokens=args.decode_steps)
        torch.cuda.synchronize(args.device)

    # ── 提取三个关键标签 ─────────────────────────────────────────────────────
    avgs = prof.key_averages()
    results = [avg for avg in avgs if avg.key in _KEY_OPS]
    total_cuda_us = sum(avg.cuda_time_total for avg in results)

    print()
    print("=" * 72)
    print(
        f"  mini-infer decode profiling"
        f"  |  batch={args.batch_size}"
        f"  |  steps={args.decode_steps}"
    )
    print("=" * 72)
    print(f"{'操作':<22} {'调用次数':>8} {'CUDA 总(ms)':>12} {'CUDA 均值(ms)':>14} {'占比':>8}")
    print("-" * 72)

    for avg in sorted(results, key=lambda x: -x.cuda_time_total):
        pct = avg.cuda_time_total / total_cuda_us * 100 if total_cuda_us > 0 else 0.0
        print(
            f"{avg.key:<22} {avg.count:>8}"
            f" {avg.cuda_time_total / 1000:>12.2f}"
            f" {avg.cuda_time / 1000:>14.3f}"
            f" {pct:>7.1f}%"
        )

    print("-" * 72)
    print(f"{'三段合计':<22} {'':>8} {total_cuda_us / 1000:>12.2f}")
    print("=" * 72)

    # ── 完整 top-20 表（供详细调试）────────────────────────────────────────
    print("\n[profiler] 完整 top-20 CUDA 操作（调试信息）：")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    main()
