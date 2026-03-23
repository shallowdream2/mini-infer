"""Phase 16：W8A8 量化 benchmark。

对比 fp16 vs W8A8 的：
- 权重显存占用（MB）
- prefill / decode / e2e 三段耗时
- decode 吞吐（tokens/s）与 TPOT（ms/token）
- e2e 吞吐（tokens/s）
- greedy token match rate / sequence exact match
- W8A8 quant 路径中 _int_mm vs fallback 命中情况

用法示例：
  # 单模式运行
  HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_quant.py \\
      --model <path> --mode fp16 --batch-size 4 --max-new-tokens 32

  # 对比运行（fp16 + W8A8 一次性比较）
  HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_quant.py \\
      --model <path> --compare --batch-size 4 --max-new-tokens 64
"""

import argparse
import gc
import time
import torch
from typing import Optional

from mini_infer.config import EngineConfig
from mini_infer.kv_cache import KVCacheManager
from mini_infer.model_runner import ModelRunner
from mini_infer.quantization import QuantLinear
from mini_infer.request import Request, RequestState, SamplingParams


# --------------------------------------------------------------------------- #
# 固定 prompt 集（exact match 评估用，12 条）
# --------------------------------------------------------------------------- #

PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "The largest planet in the solar system is",
    "Water boils at",
    "The speed of light is approximately",
    "Albert Einstein was born in",
    "The chemical symbol for gold is",
    "The first element in the periodic table is",
    "Mount Everest is located in",
    "Shakespeare wrote",
    "The Great Wall of China was built during",
    "DNA stands for",
]


def measure_weight_memory(model: torch.nn.Module) -> float:
    """统计模型所有参数和 buffer 的 GPU 显存（MB）。"""
    total_bytes = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_cuda:
            total_bytes += t.numel() * t.element_size()
    return total_bytes / 1024 / 1024


def compute_match_metrics(
    reference_token_ids: list[list[int]],
    candidate_token_ids: list[list[int]],
) -> dict[str, float]:
    """计算 greedy token match rate 和 sequence exact match。

    口径：
      - token_match_rate：逐位置对齐比较；长度差异按 mismatch 计入分母
      - sequence_exact_rate：整段 token 序列完全相同的 prompt 占比
    """
    if len(reference_token_ids) != len(candidate_token_ids):
        raise ValueError("reference_token_ids 与 candidate_token_ids 长度不一致")

    token_matches = 0
    token_total = 0
    sequence_exact = 0

    for ref_ids, cand_ids in zip(reference_token_ids, candidate_token_ids):
        if ref_ids == cand_ids:
            sequence_exact += 1
        token_total += max(len(ref_ids), len(cand_ids))
        token_matches += sum(
            1
            for idx in range(max(len(ref_ids), len(cand_ids)))
            if idx < len(ref_ids)
            and idx < len(cand_ids)
            and ref_ids[idx] == cand_ids[idx]
        )

    prompt_total = len(reference_token_ids)
    return {
        "token_match_rate": (token_matches / token_total) if token_total > 0 else 1.0,
        "token_matches": float(token_matches),
        "token_total": float(token_total),
        "sequence_exact_rate": (sequence_exact / prompt_total) if prompt_total > 0 else 1.0,
        "sequence_exact_count": float(sequence_exact),
        "prompt_total": float(prompt_total),
    }


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _accumulate_quant_stats(total: Optional[dict[str, int]], current: Optional[dict[str, int]]) -> Optional[dict[str, int]]:
    if current is None:
        return total
    if total is None:
        return dict(current)
    for key, value in current.items():
        total[key] = total.get(key, 0) + int(value)
    return total


def _format_quant_stats(stats: Optional[dict[str, int]]) -> str:
    if not stats:
        return "N/A"
    total_rows = stats["int_mm_rows"] + stats["fallback_rows"]
    fallback_ratio = (stats["fallback_rows"] / total_rows) if total_rows > 0 else 0.0
    return (
        f"int_mm_calls={stats['int_mm_calls']}, fallback_calls={stats['fallback_calls']}, "
        f"fallback_rows_ratio={fallback_ratio:.1%}"
    )


def run_benchmark(
    model_path: str,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    num_gpu_blocks: int = 200,
    warmup_iters: int = 2,
    bench_iters: int = 5,
    reference_token_ids: Optional[list[list[int]]] = None,
) -> dict:
    """运行单次 benchmark，返回结果字典。

    Args:
        reference_token_ids: fp16 模式的参考 token 序列，用于 W8A8 token match 计算。
                             如果 mode == "fp16"，传 None；W8A8 模式传 fp16 的 token id 列表。
    """
    # 构建最小 EngineConfig
    cfg = EngineConfig(
        model_name=model_path,
        device="cuda:0",
        dtype="float16",
        max_batch_size=batch_size,
        max_model_len=512,
        block_size=256,
        num_gpu_blocks=num_gpu_blocks,
        # 1.5B 参数
        num_hidden_layers=28,
        num_kv_heads=2,
        head_dim=128,
        quant_mode="w8a8" if mode == "w8a8" else "",
    )

    kv_cache = KVCacheManager(cfg)
    runner = ModelRunner(cfg, kv_cache)

    weight_mem_mb = measure_weight_memory(runner.model)

    # 准备 prompt
    prompts = PROMPTS[:batch_size]

    def run_one_batch() -> dict:
        params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
        states = []
        for i, p in enumerate(prompts):
            token_ids = runner.tokenizer.encode(p, add_special_tokens=True)
            req = Request(request_id=f"r{i}", prompt=p, sampling_params=params)
            st = RequestState(request=req, prompt_token_ids=token_ids)
            states.append(st)

        # 分配 KV 块（必须在 prefill 前调用）
        for st in states:
            kv_cache.init_request(st)

        prefill_quant_stats = None
        decode_quant_stats = None

        if mode == "w8a8":
            QuantLinear.reset_runtime_stats()
        _sync_cuda()
        t_prefill_start = time.perf_counter()
        runner.prefill(states)
        _sync_cuda()
        t_prefill_end = time.perf_counter()
        if mode == "w8a8":
            prefill_quant_stats = QuantLinear.get_runtime_stats()
            QuantLinear.reset_runtime_stats()

        # decode 循环
        active = [s for s in states if not s.finished]
        _sync_cuda()
        t_decode_start = time.perf_counter()
        for _ in range(max_new_tokens - 1):
            if not active:
                break
            runner.decode_batch(active)
            active = [s for s in active if not s.finished]
        _sync_cuda()
        t_decode_end = time.perf_counter()
        if mode == "w8a8":
            decode_quant_stats = QuantLinear.get_runtime_stats()

        output_token_ids = [list(st.generated_token_ids) for st in states]
        output_texts = [
            runner.tokenizer.decode(token_ids, skip_special_tokens=True)
            for token_ids in output_token_ids
        ]
        total_generated_tokens = sum(len(token_ids) for token_ids in output_token_ids)
        decode_generated_tokens = sum(max(len(token_ids) - 1, 0) for token_ids in output_token_ids)

        # 释放 KV 块
        for st in states:
            kv_cache.free_request(st)

        return {
            "texts": output_texts,
            "token_ids": output_token_ids,
            "prefill_s": t_prefill_end - t_prefill_start,
            "decode_s": t_decode_end - t_decode_start,
            "e2e_s": (t_prefill_end - t_prefill_start) + (t_decode_end - t_decode_start),
            "generated_tokens": total_generated_tokens,
            "decode_generated_tokens": decode_generated_tokens,
            "prefill_quant_stats": prefill_quant_stats,
            "decode_quant_stats": decode_quant_stats,
        }

    # warmup
    for _ in range(warmup_iters):
        run_one_batch()
        gc.collect()
        _sync_cuda()

    total_prefill_s = 0.0
    total_decode_s = 0.0
    total_e2e_s = 0.0
    total_generated_tokens = 0
    total_decode_generated_tokens = 0
    outputs = None
    output_token_ids = None
    prefill_quant_stats_total = None
    decode_quant_stats_total = None

    for _ in range(bench_iters):
        batch_result = run_one_batch()
        gc.collect()
        total_prefill_s += batch_result["prefill_s"]
        total_decode_s += batch_result["decode_s"]
        total_e2e_s += batch_result["e2e_s"]
        total_generated_tokens += batch_result["generated_tokens"]
        total_decode_generated_tokens += batch_result["decode_generated_tokens"]
        outputs = batch_result["texts"]
        output_token_ids = batch_result["token_ids"]
        prefill_quant_stats_total = _accumulate_quant_stats(
            prefill_quant_stats_total,
            batch_result["prefill_quant_stats"],
        )
        decode_quant_stats_total = _accumulate_quant_stats(
            decode_quant_stats_total,
            batch_result["decode_quant_stats"],
        )

    decode_throughput = (
        total_decode_generated_tokens / total_decode_s
        if total_decode_generated_tokens > 0 and total_decode_s > 0
        else None
    )
    decode_tpot_ms = (
        1000.0 * total_decode_s / total_decode_generated_tokens
        if total_decode_generated_tokens > 0
        else None
    )
    e2e_throughput = (
        total_generated_tokens / total_e2e_s
        if total_generated_tokens > 0 and total_e2e_s > 0
        else 0.0
    )

    token_match_rate: Optional[float] = None
    sequence_exact_rate: Optional[float] = None
    match_metrics = None
    if reference_token_ids is not None and output_token_ids is not None:
        match_metrics = compute_match_metrics(reference_token_ids, output_token_ids)
        token_match_rate = match_metrics["token_match_rate"]
        sequence_exact_rate = match_metrics["sequence_exact_rate"]

    result = {
        "mode": mode,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "weight_mem_mb": weight_mem_mb,
        "avg_prefill_ms": 1000.0 * total_prefill_s / bench_iters,
        "decode_tps": decode_throughput,
        "decode_tpot_ms": decode_tpot_ms,
        "e2e_tps": e2e_throughput,
        "token_match_rate": token_match_rate,
        "sequence_exact_rate": sequence_exact_rate,
        "match_metrics": match_metrics,
        "prefill_quant_stats": prefill_quant_stats_total,
        "decode_quant_stats": decode_quant_stats_total,
        "outputs": outputs,
        "output_token_ids": output_token_ids,
    }
    return result


def print_result(r: dict) -> None:
    mode = r["mode"].upper()
    print(f"\n{'='*50}")
    print(f"Mode          : {mode}")
    print(f"Batch size    : {r['batch_size']}")
    print(f"Max new tokens: {r['max_new_tokens']}")
    print(f"Weight memory : {r['weight_mem_mb']:.1f} MB")
    print(f"Avg prefill   : {r['avg_prefill_ms']:.2f} ms/batch")
    if r["decode_tps"] is not None:
        print(f"Decode TPS    : {r['decode_tps']:.1f} tokens/s")
        print(f"Decode TPOT   : {r['decode_tpot_ms']:.2f} ms/token")
    else:
        print("Decode TPS    : N/A")
        print("Decode TPOT   : N/A")
    print(f"E2E TPS       : {r['e2e_tps']:.1f} tokens/s")
    if r["token_match_rate"] is not None:
        print(f"Token match   : {r['token_match_rate']*100:.1f}% vs fp16")
    if r["sequence_exact_rate"] is not None:
        print(f"Seq exact     : {r['sequence_exact_rate']*100:.1f}% vs fp16")
    if r["prefill_quant_stats"] is not None:
        print(f"Prefill quant : {_format_quant_stats(r['prefill_quant_stats'])}")
    if r["decode_quant_stats"] is not None:
        print(f"Decode quant  : {_format_quant_stats(r['decode_quant_stats'])}")
    print(f"{'='*50}")


def print_comparison(fp16: dict, w8a8: dict) -> None:
    print("\n" + "=" * 60)
    print("fp16 vs W8A8 对比")
    print("=" * 60)
    print(f"{'指标':<20} {'fp16':>12} {'W8A8':>12} {'比值':>10}")
    print("-" * 60)
    print(f"{'权重显存 (MB)':<20} {fp16['weight_mem_mb']:>12.1f} {w8a8['weight_mem_mb']:>12.1f} "
          f"{w8a8['weight_mem_mb']/fp16['weight_mem_mb']:>10.3f}×")
    if fp16["decode_tps"] is not None and w8a8["decode_tps"] is not None:
        print(f"{'Decode 吞吐':<20} {fp16['decode_tps']:>12.1f} {w8a8['decode_tps']:>12.1f} "
              f"{w8a8['decode_tps']/fp16['decode_tps']:>10.3f}×")
        print(f"{'Decode TPOT (ms)':<20} {fp16['decode_tpot_ms']:>12.2f} {w8a8['decode_tpot_ms']:>12.2f} "
              f"{w8a8['decode_tpot_ms']/fp16['decode_tpot_ms']:>10.3f}×")
    print(f"{'E2E 吞吐':<20} {fp16['e2e_tps']:>12.1f} {w8a8['e2e_tps']:>12.1f} "
          f"{w8a8['e2e_tps']/fp16['e2e_tps']:>10.3f}×")
    if w8a8["token_match_rate"] is not None:
        print(f"{'Token match':<20} {'—':>12} {w8a8['token_match_rate']*100:>11.1f}% {'—':>10}")
    if w8a8["sequence_exact_rate"] is not None:
        print(f"{'Seq exact':<20} {'—':>12} {w8a8['sequence_exact_rate']*100:>11.1f}% {'—':>10}")
    print("=" * 60)
    print("\n验收标准检查：")
    mem_ratio = w8a8['weight_mem_mb'] / fp16['weight_mem_mb']
    print(f"  权重显存降低 ≥ 30%: {'✓' if mem_ratio <= 0.70 else '✗'} (降低 {(1-mem_ratio)*100:.1f}%)")
    if w8a8["token_match_rate"] is not None:
        em = w8a8["token_match_rate"]
        print(f"  Greedy token match ≥ 70%: {'✓' if em >= 0.70 else '✗'} ({em*100:.1f}%)")
    if fp16["decode_tps"] is not None and w8a8["decode_tps"] is not None:
        thr_ratio = w8a8["decode_tps"] / fp16["decode_tps"]
        print(f"  Stretch: decode 吞吐 ≥ 1.10×: {'✓' if thr_ratio >= 1.10 else '✗'} ({thr_ratio:.3f}×)")
        if thr_ratio < 1.10 and w8a8["decode_quant_stats"] is not None:
            print(f"  Decode 量化路径: {_format_quant_stats(w8a8['decode_quant_stats'])}")


def main():
    parser = argparse.ArgumentParser(description="W8A8 量化 benchmark")
    parser.add_argument("--model", required=True, help="本地模型路径")
    parser.add_argument("--mode", choices=["fp16", "w8a8"], default="fp16",
                        help="量化模式（单模式运行时使用）")
    parser.add_argument("--compare", action="store_true",
                        help="同时运行 fp16 和 W8A8 并输出对比表")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--bench-iters", type=int, default=5)
    args = parser.parse_args()

    if args.compare:
        print("运行 fp16 基线...")
        fp16_result = run_benchmark(
            args.model, "fp16", args.batch_size, args.max_new_tokens,
            args.num_gpu_blocks, args.warmup_iters, args.bench_iters,
        )
        print_result(fp16_result)

        # 清理 GPU 显存
        gc.collect()
        torch.cuda.empty_cache()

        print("\n运行 W8A8 量化...")
        w8a8_result = run_benchmark(
            args.model, "w8a8", args.batch_size, args.max_new_tokens,
            args.num_gpu_blocks, args.warmup_iters, args.bench_iters,
            reference_token_ids=fp16_result["output_token_ids"],
        )
        print_result(w8a8_result)
        print_comparison(fp16_result, w8a8_result)
    else:
        result = run_benchmark(
            args.model, args.mode, args.batch_size, args.max_new_tokens,
            args.num_gpu_blocks, args.warmup_iters, args.bench_iters,
        )
        print_result(result)


if __name__ == "__main__":
    main()
