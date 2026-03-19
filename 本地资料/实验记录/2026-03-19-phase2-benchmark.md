# Phase 2 Benchmark：mini-infer vs HuggingFace Baseline

这个文件记录 Phase 2（Paged KV Cache + Continuous Batching + Batch Decode）的 benchmark 实验，与 HF baseline 对比。

## 环境

| 项目 | 值 |
|------|-----|
| 日期 | 2026-03-19 |
| 机器 | Ubuntu 24.04 |
| GPU | RTX 4090 (GPU 0，24 GB VRAM) |
| Conda 环境 | ai-infra |
| Python | 3.10 |
| transformers | 4.43.4（有 DynamicCache 弃用警告，功能正常） |
| PyTorch | CUDA |
| 模型 | Qwen2.5-7B-Instruct（本地路径） |

## 模型架构参数

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 28 |
| num_key_value_heads | 4 (GQA，28 Q heads : 4 KV heads) |
| head_dim | 128 |
| num_attention_heads | 28 |

## 命令

```bash
# HF Baseline
conda run -n ai-infra python benchmarks/benchmark_hf.py \
  --model /home/shh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct \
  --batch-size 1 --max-new-tokens 128 --device cuda:0

# mini-infer Phase 2
conda run -n ai-infra python benchmarks/benchmark_mini.py \
  --model /home/shh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct \
  --device cuda:0 --batch-size 1 --max-new-tokens 128 --num-gpu-blocks 512
```

## 指标口径

- **Throughput**：所有请求总输出 token 数 / 总耗时（tok/s）
- **TTFT**：用单条请求 max_new_tokens=1 时的端到端时间近似（ms）
- **TPOT（mini-infer）**：`(总耗时 - TTFT) / (总 token - batch_size)`，即 amortized 批次均摊值，非单请求延迟
- **TPOT（HF）**：`(总耗时 - TTFT) / (总 token - 1)`，单请求延迟
- **Peak Mem**：`torch.cuda.max_memory_allocated()` 峰值

> 注意：TPOT 口径不同，mini-infer 是批次均摊，HF 是单请求均摊。直接对比 TPOT 数值不等价。

## 结果

### HuggingFace Baseline（静态 Batching）

| batch_size | Throughput (tok/s) | TTFT (ms) | TPOT (ms/tok) | Peak Mem (GB) |
|-----------|-------------------|-----------|---------------|---------------|
| 1 | 56.2 | 19.0 | 17.78 | 15.78 |
| 4 | 210.5 | 20.8 | 18.99 | 15.81 |
| 8 | 408.9 | 23.6 | 19.53 | 15.88 |

### mini-infer Phase 2（Paged KV Cache + Continuous Batching）

| batch_size | Throughput (tok/s) | TTFT (ms) | TPOT 均摊 (ms/tok) | Peak Mem (GB) |
|-----------|-------------------|-----------|---------------------|---------------|
| 1 | 49.4 | 18.7 | 20.24 | 16.26 |
| 4 | 135.2 | 19.0 | 7.42 | 16.31 |
| 8 | 201.0 | 18.8 | 5.00 | 16.42 |

### Throughput 对比（mini-infer / HF）

| batch_size | HF (tok/s) | mini-infer (tok/s) | 比率 |
|-----------|------------|-------------------|------|
| 1 | 56.2 | 49.4 | 87.9% |
| 4 | 210.5 | 135.2 | 64.2% |
| 8 | 408.9 | 201.0 | 49.1% |

## 分析

### 1. TTFT 基本一致
mini-infer 与 HF 的 TTFT 几乎相同（18.7–19.0ms），说明 prefill 路径完整，不存在额外 prefill 开销。

### 2. Throughput 低于 HF，差距随 batch 扩大
- batch=1 时 mini-infer 达到 HF 的 87.9%，差距主要来自块管理元数据开销
- batch=4/8 差距扩大，核心原因是 `gather_batch_kv()` 每个 decode step 都复制一次所有请求的 KV 到新的 dense tensor，复制量随 batch 线性增长

### 3. mini-infer TPOT 均摊值随 batch 下降
mini-infer 均摊 TPOT: 20.24ms (b=1) → 7.42ms (b=4) → 5.00ms (b=8)，说明 batch decode 的 GPU 计算确实在均摊，batch decode 逻辑正确。

### 4. 显存开销
mini-infer 比 HF 多用约 0.5 GB：
- KV block pool 预分配：512 块 × 16 slots × 28 层 × 2 (K+V) × 4 heads × 128 dim × float16 ≈ 512×16×28×2×4×128×2 bytes ≈ **0.48 GB** ✓

### 5. 吞吐量差距根因
| 原因 | 影响 |
|------|------|
| `gather_batch_kv()` 每步复制 KV 到 dense tensor | 主要瓶颈，O(batch × seq_len) 复制 |
| Python 循环控制开销 | 次要 |
| 无 CUDA graph / FlashAttention | 理论上限低于 HF |

## 结论

Phase 2 批次 decode 架构正确，TTFT 与 HF 对齐，吞吐量在 batch=1 达到 HF 的 87.9%。当前主要瓶颈是 `gather_batch_kv()` 的 KV 复制开销，这是实现简单性的代价。

**Phase 2 没有实现吞吐量超越 HF baseline，这是预期内的结果**：Phase 2 目标是正确实现 Paged KV Cache + Continuous Batching 架构，不是极致性能优化。

## 局限性

1. TPOT 口径不同，mini-infer 是均摊值，HF 是单请求延迟，不可直接比较
2. HF batch=4/8 用的是右填充（`right-padding`）运行时有 warning，对正确性有影响但对吞吐量测量影响有限
3. 未测试长序列（prompt_len > 200）场景下 paged KV cache 的内存节省优势
4. mini-infer 的 continuous batching 优势在混合长度请求场景下更明显，本次 benchmark 用的是同质 prompt，未充分展示该优势
