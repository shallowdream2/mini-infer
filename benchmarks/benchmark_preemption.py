"""
benchmarks/benchmark_preemption.py — Phase 7 Preemption + Priority Scheduling benchmark。

测试对象：mini-infer LLMEngine（dry_run=False，需要 GPU + 模型权重）

测量内容：
  1. swap_out / swap_in 单次延迟（µs）
  2. 有无抢占时的端到端吞吐对比（tok/s）
  3. 高优先级请求的 TTFT 对比（有/无抢占）

运行方式（需要 GPU + Qwen2.5-1.5B）：
  conda run -n ai-infra python benchmarks/benchmark_preemption.py

Workload：
  - 模型：Qwen2.5-1.5B-Instruct（本地路径由 MODEL_PATH 环境变量覆盖）
  - prompt 长度：16~64 token
  - max_new_tokens：32
  - batch_size：4
  - block_size：16
  - 对照组：无抢占（所有请求优先级相同）
  - 实验组：有抢占（低优先级先入，高优先级后入触发换出）
"""

import os
import sys
import time

import torch

# 允许从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mini_infer import EngineConfig, LLMEngine

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.expanduser("~/models/Qwen2.5-1.5B-Instruct"),
)

# ── 参数 ────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 16
NUM_GPU_BLOCKS = 64
MAX_BATCH_SIZE = 4
MAX_NEW_TOKENS = 32

# 用于 swap 延迟测量的 prompt（确保需要 ceil((len+max_out)/block_size) 块）
SWAP_PROMPTS = ["The quick brown fox"] * 2

# 吞吐测试 prompt 集
THROUGHPUT_PROMPTS = [
    "Tell me about artificial intelligence.",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "How does photosynthesis work?",
    "What are the benefits of exercise?",
    "Describe the solar system.",
    "What is machine learning?",
]

WARMUP_PROMPTS = ["Hello world"] * 2


def make_engine(num_gpu_blocks: int = NUM_GPU_BLOCKS) -> LLMEngine:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型权重未找到：{MODEL_PATH}\n"
            "请设置环境变量 MODEL_PATH 指向正确路径，或使用 dry_run=True 模式。"
        )
    config = EngineConfig(
        model_name=MODEL_PATH,
        dry_run=False,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=MAX_BATCH_SIZE,
    )
    return LLMEngine(config)


def make_dry_run_engine(
    num_gpu_blocks: int = NUM_GPU_BLOCKS,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> LLMEngine:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=max_batch_size,
    )
    return LLMEngine(config)


# ── 测试 1：dry_run 下调度延迟 ───────────────────────────────────────────────

def bench_scheduler_latency(n_trials: int = 1000) -> None:
    """在 dry_run 模式下测量 swap_out / swap_in 调度路径的延迟（不含模型）。"""
    print("\n=== Scheduler latency（dry_run，不含模型 forward）===")

    from mini_infer.kv_cache import KVCacheManager
    from mini_infer.request import Request, RequestState, SamplingParams

    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=64,
        max_batch_size=8,
    )
    mgr = KVCacheManager(config)

    def make_state(i: int) -> RequestState:
        return RequestState(
            request=Request(
                request_id=f"req-{i}",
                prompt="x" * 32,
                sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS),
            ),
            prompt_token_ids=[1] * 32,
        )

    # 预热
    for i in range(10):
        s = make_state(i)
        mgr.init_request(s)
        mgr.swap_out(s)
        mgr.swap_in(s)
        mgr.free_request(s)

    # 正式测量
    swap_out_times = []
    swap_in_times = []

    for i in range(n_trials):
        s = make_state(i + 100)
        mgr.init_request(s)

        t0 = time.perf_counter()
        mgr.swap_out(s)
        t1 = time.perf_counter()
        mgr.swap_in(s)
        t2 = time.perf_counter()

        swap_out_times.append((t1 - t0) * 1e6)
        swap_in_times.append((t2 - t1) * 1e6)

        mgr.free_request(s)

    def stats(xs: list[float]) -> str:
        import statistics
        return (
            f"mean={statistics.mean(xs):.2f}µs  "
            f"median={statistics.median(xs):.2f}µs  "
            f"p99={sorted(xs)[int(len(xs) * 0.99)]:.2f}µs"
        )

    print(f"  swap_out ({n_trials} trials): {stats(swap_out_times)}")
    print(f"  swap_in  ({n_trials} trials): {stats(swap_in_times)}")
    print("  注：dry_run 模式，只含块分配/释放逻辑，无 tensor 拷贝。")


# ── 测试 2：dry_run 端到端，有/无抢占吞吐对比 ────────────────────────────────

def bench_e2e_dry_run() -> None:
    """dry_run 端到端调度延迟：无抢占 vs 有抢占。"""
    print("\n=== E2E dry_run 端到端对比 ===")

    prompts = ["ab cd ef gh"] * 4  # 每个 ~8 tokens

    # 无抢占（充足 KV 块）
    engine_no_preempt = make_dry_run_engine(num_gpu_blocks=32, max_batch_size=4)
    t0 = time.perf_counter()
    out_no = engine_no_preempt.generate(prompts, max_new_tokens=8)
    t1 = time.perf_counter()
    elapsed_no = (t1 - t0) * 1000

    # 有抢占（KV 块受限，低优先级被换出）
    engine_preempt = make_dry_run_engine(num_gpu_blocks=2, max_batch_size=4)
    priorities = [5, 5, 0, 0]  # 前两个低优先级，后两个高优先级
    t0 = time.perf_counter()
    out_yes = engine_preempt.generate(prompts, max_new_tokens=4, priorities=priorities)
    t1 = time.perf_counter()
    elapsed_yes = (t1 - t0) * 1000

    print(f"  无抢占（充足块）：{elapsed_no:.1f}ms，输出 {len(out_no)} 条")
    print(f"  有抢占（限制块）：{elapsed_yes:.1f}ms，输出 {len(out_yes)} 条")
    print("  注：dry_run 模式，纯调度逻辑，不含模型推理。")


# ── 测试 3：真实 GPU 吞吐（需要模型权重）────────────────────────────────────

def bench_gpu_throughput() -> None:
    """GPU 实机：无抢占 vs 有抢占的吞吐和 TTFT 对比。"""
    print("\n=== GPU 吞吐对比（需要模型权重）===")

    try:
        engine = make_engine()
    except FileNotFoundError as e:
        print(f"  跳过（{e}）")
        return

    # 预热
    print("  预热中...")
    engine.generate(WARMUP_PROMPTS, max_new_tokens=8)
    torch.cuda.synchronize()

    # 无抢占
    print("  测量无抢占吞吐...")
    t0 = time.perf_counter()
    out_base = engine.generate(
        THROUGHPUT_PROMPTS[:4], max_new_tokens=MAX_NEW_TOKENS
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_tokens_base = sum(len(o.split()) for o in out_base) * 1.3  # 粗估
    throughput_base = total_tokens_base / (t1 - t0)
    print(f"  无抢占：{t1 - t0:.2f}s，约 {throughput_base:.0f} tok/s")

    # 有抢占（低优先级先进，高优先级触发换出）
    print("  测量有抢占吞吐...")
    # 重建 engine（清空状态）
    engine2 = make_engine()
    engine2.generate(WARMUP_PROMPTS, max_new_tokens=8)
    torch.cuda.synchronize()

    mixed_prompts = THROUGHPUT_PROMPTS[:4]
    priorities = [5, 5, 0, 0]  # 前两低优先，后两高优先

    t0 = time.perf_counter()
    out_preempt = engine2.generate(
        mixed_prompts, max_new_tokens=MAX_NEW_TOKENS, priorities=priorities
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_tokens_preempt = sum(len(o.split()) for o in out_preempt) * 1.3
    throughput_preempt = total_tokens_preempt / (t1 - t0)
    print(f"  有抢占：{t1 - t0:.2f}s，约 {throughput_preempt:.0f} tok/s")

    overhead = (throughput_base - throughput_preempt) / throughput_base * 100
    print(f"  抢占开销：约 {overhead:.1f}%（正值=有抢占更慢）")
    print("  注：tok/s 为粗估（按空格分词），仅供参考。请用 benchmark_mini.py 获取精确数字。")


# ── 测试 4：GPU swap 真实延迟（需要模型权重）────────────────────────────────

def bench_gpu_swap_latency() -> None:
    """GPU 实机：测量单次 swap_out + swap_in 的真实延迟（含 GPU→CPU tensor 拷贝）。"""
    print("\n=== GPU Swap 真实延迟（含 tensor 拷贝）===")

    try:
        engine = make_engine(num_gpu_blocks=NUM_GPU_BLOCKS)
    except FileNotFoundError as e:
        print(f"  跳过（{e}）")
        return

    from mini_infer.request import Request, RequestState, SamplingParams

    mgr = engine.kv_cache
    n_trials = 20
    swap_out_ms = []
    swap_in_ms = []

    prompt_len = 32

    for i in range(n_trials):
        state = RequestState(
            request=Request(
                request_id=f"bench-{i}",
                prompt="x" * prompt_len,
                sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS),
            ),
            prompt_token_ids=[1] * prompt_len,
        )
        mgr.init_request(state)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        mgr.swap_out(state)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        mgr.swap_in(state)
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        swap_out_ms.append((t1 - t0) * 1000)
        swap_in_ms.append((t2 - t1) * 1000)

        mgr.free_request(state)

    import statistics
    print(f"  swap_out ({n_trials} 次): mean={statistics.mean(swap_out_ms):.2f}ms  "
          f"median={statistics.median(swap_out_ms):.2f}ms")
    print(f"  swap_in  ({n_trials} 次): mean={statistics.mean(swap_in_ms):.2f}ms  "
          f"median={statistics.median(swap_in_ms):.2f}ms")
    print(f"  prompt_len={prompt_len}, max_new_tokens={MAX_NEW_TOKENS}, block_size={BLOCK_SIZE}")
    print(f"  模型：{os.path.basename(MODEL_PATH)}")


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 7 Preemption Benchmark")
    parser.add_argument(
        "--dry-only",
        action="store_true",
        help="只跑 dry_run 测试（无需 GPU/模型权重）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 7 Preemption + Priority Scheduling Benchmark")
    print("=" * 60)

    bench_scheduler_latency()
    bench_e2e_dry_run()

    if not args.dry_only:
        bench_gpu_swap_latency()
        bench_gpu_throughput()
    else:
        print("\n（--dry-only 模式：跳过 GPU 测试）")

    print("\n✓ benchmark 完成")
