"""这个文件实现 mini-infer 引擎的 benchmark，对标 benchmark_hf.py 的同款 prompt 和指标。
当前阶段（Phase 1）：真实模型 + HuggingFace past_key_values，无 Paged KV Cache，无 Continuous Batching。
运行环境：默认在 Ubuntu 项目环境中执行；如使用 CUDA 设备，需要已就绪的 GPU 和模型权重。
"""

import argparse
import time
from dataclasses import dataclass

import torch

from mini_infer import EngineConfig, LLMEngine

# 与 benchmark_hf.py 保持一致的 prompt 集合
PROMPTS = [
    "请介绍一下大语言模型的推理优化技术。",
    "什么是 PagedAttention？它解决了什么问题？",
    "解释 KV Cache 在 Transformer 推理中的作用。",
    "Continuous Batching 和静态 Batching 的区别是什么？",
    "如何在多 GPU 上做张量并行推理？",
    "介绍一下 FlashAttention 的核心思想。",
    "大模型推理时显存占用的主要来源有哪些？",
    "什么是 Prefill 阶段和 Decode 阶段的区别？",
]


@dataclass
class BenchmarkResult:
    engine_phase: str
    model_name: str
    batch_size: int
    max_new_tokens: int
    total_time_s: float
    throughput_tok_s: float
    peak_memory_gb: float


def benchmark_mini(
    model_name: str,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "float16",
) -> BenchmarkResult:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("benchmark_mini 需要可用的 CUDA GPU，但当前环境未检测到。")

    config = EngineConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_batch_size=batch_size,
        block_size=16,
    )

    print(f"初始化 LLMEngine: {model_name}")
    engine = LLMEngine(config)

    prompts = PROMPTS[:batch_size]

    print("热身中...")
    _ = engine.generate(prompts[:1], max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    total_tokens = sum(
        len(engine.model_runner.tokenizer.encode(o, add_special_tokens=False))
        for o in outputs
    )
    throughput = total_tokens / total_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return BenchmarkResult(
        engine_phase="mini-infer-phase1",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=throughput,
        peak_memory_gb=peak_mem_gb,
    )


def print_result(result: BenchmarkResult) -> None:
    print("\n========== mini-infer Benchmark ==========")
    print(f"引擎:           {result.engine_phase}")
    print(f"模型:           {result.model_name}")
    print(f"batch_size:     {result.batch_size}")
    print(f"max_new_tokens: {result.max_new_tokens}")
    print(f"总耗时:         {result.total_time_s:.2f} s")
    print(f"Throughput:     {result.throughput_tok_s:.1f} tokens/s")
    print(f"Peak Mem:       {result.peak_memory_gb:.2f} GB")
    print("==========================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    result = benchmark_mini(
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    print_result(result)


if __name__ == "__main__":
    main()
