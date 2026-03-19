# Phase 3 Benchmark 实验记录

本文件记录 mini-infer Phase 3（向量化 gather + DynamicCache + 混合长度 benchmark）的真实性能数据，并与 Phase 2 和 HF baseline 对比。

## 实验环境

- 硬件：Ubuntu 24.04 + RTX 4090 × 1（GPU 0，显存 24 GB）
- 模型：Qwen/Qwen2.5-7B-Instruct，float16，单卡
- 模型路径：/data/models/Qwen2.5-7B-Instruct（本地）
- Python：3.11，transformers 4.43.4，torch 2.3.1，flash-attn 2.3.6
- 环境：conda ai-infra

## 测试命令

```bash
# 标准 benchmark（batch=1/4/8，max_new_tokens=128）
python benchmarks/benchmark_mini.py --model /data/models/Qwen2.5-7B-Instruct --batch-size 1 --max-new-tokens 128
python benchmarks/benchmark_mini.py --model /data/models/Qwen2.5-7B-Instruct --batch-size 4 --max-new-tokens 128
python benchmarks/benchmark_mini.py --model /data/models/Qwen2.5-7B-Instruct --batch-size 8 --max-new-tokens 128

# 混合长度 benchmark（8 请求，短 prompt × 4 + 长 prompt × 4，max_new_tokens=256）
python benchmarks/benchmark_mini.py --model /data/models/Qwen2.5-7B-Instruct --mixed
```

## 测量指标口径

| 指标 | 口径 |
|------|------|
| Throughput | 输出 token 总数 / 总耗时（tok/s） |
| TTFT | 单条请求生成第一个 token 的延迟（ms），用单条 prefill+decode 1 token 近似 |
| TPOT | amortized，(总时间 - TTFT) / (总 token - batch_size)（ms/tok） |
| Peak Mem | torch.cuda.max_memory_allocated()（GB） |

注：TTFT 和 TPOT 在 batch=1 时最有参考价值；batch>1 时 TPOT 为批均值，非单请求延迟。

## 标准 Benchmark 结果（max_new_tokens=128）

### batch_size=1

| 指标 | Phase 2 | Phase 3 | HF baseline |
|------|---------|---------|-------------|
| Throughput | 49.4 tok/s | 53.7 tok/s | 56.2 tok/s |
| TTFT | 35.3 ms | 34.9 ms | 33.6 ms |
| TPOT | 19.6 ms/tok | 18.2 ms/tok | 17.6 ms/tok |
| Peak Mem | 15.24 GB | 15.26 GB | 14.20 GB |

### batch_size=4

| 指标 | Phase 2 | Phase 3 | HF baseline |
|------|---------|---------|-------------|
| Throughput | 135.2 tok/s | 194.2 tok/s | 210.5 tok/s |
| TTFT | 38.1 ms | 36.2 ms | 34.8 ms |
| TPOT | 27.5 ms/tok | 19.8 ms/tok | 18.5 ms/tok |
| Peak Mem | 15.61 GB | 15.58 GB | 14.35 GB |

### batch_size=8

| 指标 | Phase 2 | Phase 3 | HF baseline |
|------|---------|---------|-------------|
| Throughput | 201.0 tok/s | 361.3 tok/s | 408.9 tok/s |
| TTFT | 45.2 ms | 43.8 ms | 42.1 ms |
| TPOT | 37.8 ms/tok | 21.1 ms/tok | 18.9 ms/tok |
| Peak Mem | 16.12 GB | 16.08 GB | 14.67 GB |

## 混合长度 Benchmark 结果

workload：8 请求（短 prompt × 4 + 长 prompt × 4），所有请求统一 max_new_tokens=256

| 指标 | 结果 |
|------|------|
| Throughput | 356.0 tok/s |
| Peak Mem | 17.00 GB |
| num_requests | 8 |

注：当前 generate() 不支持 per-request max_new_tokens，短 prompt 请求不提前截断，以最大值 256 统一运行。短请求实际输出 token 数由 EOS 决定，不一定更少。

## 对比分析

### Phase 3 vs Phase 2

| batch | Phase 2 | Phase 3 | 提升幅度 |
|-------|---------|---------|---------|
| 1 | 49.4 tok/s | 53.7 tok/s | +8.7% |
| 4 | 135.2 tok/s | 194.2 tok/s | +43.6% |
| 8 | 201.0 tok/s | 361.3 tok/s | +79.8% |

**结论**：Phase 3 的向量化 gather_batch_kv 和 DynamicCache 迁移对 batch>1 的场景提升显著，batch=8 接近翻倍。batch=1 改善有限（无 gather 压力）。

### Phase 3 vs HF baseline

| batch | HF baseline | Phase 3 | Phase 3 / HF |
|-------|------------|---------|--------------|
| 1 | 56.2 tok/s | 53.7 tok/s | 95.5% |
| 4 | 210.5 tok/s | 194.2 tok/s | 92.3% |
| 8 | 408.9 tok/s | 361.3 tok/s | 88.4% |

**结论**：Phase 3 与 HF baseline 的差距大幅缩小（Phase 2 batch=8 仅 49.1%，Phase 3 达到 88.4%）。主要剩余差距来自 gather_batch_kv 每步仍有 KV 复制（flash_attn 2.3.6 无 block_table 参数，无法消除），以及 block tensor 与 HF 的连续 KV 内存布局差异。

## 残余差距分析

1. **每步 KV gather 仍有复制**：gather_batch_kv 向量化后消除了 Python 循环开销，但仍需将分散的 block 数据 gather 到连续 dense tensor 再传入 HF 模型。彻底消除需要 flash_attn 2.5+ 的 `block_table` 支持，当前 flash_attn 2.3.6 不具备。
2. **内存布局差异**：HF 使用连续 KV tensor，mini-infer 使用 block 分片存储，每步 decode 多一次 permute。
3. **峰值显存略高**：mini-infer 预分配 block tensor pool（num_gpu_blocks=512，~1.8 GB），HF 按需分配。

## 缺失与局限

- TPOT 是 batch amortized 值，非单请求 p50/p95 延迟
- 混合 benchmark 无法体现 per-request EOS 截断优势（generate() 不支持 per-request max_new_tokens）
- 无多并发流压测（当前 continuous batching 循环每批同步执行）
