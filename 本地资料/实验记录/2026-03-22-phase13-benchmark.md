# Phase 13 Tensor Parallelism Benchmark

## 环境

- 硬件：2 × RTX 4090 (24 GB VRAM each)
- 操作系统：Ubuntu 24.04
- PyTorch：2.1.2+cu121
- transformers：4.43.4
- flash_attn：2.5.9.post1
- 模型：Qwen2.5-1.5B-Instruct（本地，HF_HUB_OFFLINE=1）
- dtype：float16
- 运行日期：2026-03-22

## Workload

- prompts：3 条（中英文混合，详见 benchmarks/benchmark_tp.py PROMPTS）
- max_new_tokens：64
- warmup：2 runs（单卡/PP），2 runs（TP）
- measure：5 runs（单卡/PP），5 runs（TP）
- batch 方式：顺序逐条生成（sequential per-prompt generate，非 batched）

## 测量口径说明

- throughput = (∑ output_tokens × runs) / wall_clock_seconds
- peak_vram = torch.cuda.max_memory_allocated() after inference（含模型权重）
- TP 模式：torchrun --nproc_per_node 2 启动，每 rank 独立测量 VRAM，rank 0 收集 all-gather
- TTFT / TPOT：未测量（不接入 LLMEngine，不适用）
- --mode tp（mp.spawn）：计时包含进程启动 + 模型加载，**不可用于吞吐对比**，已从 --mode all 排除

## 结果

| 模式 | 吞吐量 (tok/s) | VRAM GPU0 (GB) | VRAM GPU1 (GB) | 相对单卡 |
|------|---------------|---------------|---------------|---------|
| single (单卡 HF generate) | 98.0 | 3.58 | N/A | 100% (baseline) |
| pp (device_map=balanced) | 82.4 | 2.03 | 1.56 | 84.1% |
| tp=2 (torchrun + NCCL all-reduce) | 76.5 | 3.57 | 3.57 | 78.1% |

## 正确性验证

TP=2 与单卡 greedy 输出完全一致（3/3 prompts text match 100%）：

```
[single] sample: ' 量子计算是一种基于量子力学原理的新型计算方式，它利用了量子比特（qubit）来存储和处理信息。与传统计算机使用的二进制'
[tp=2]   sample: ' 量子计算是一种基于量子力学原理的新型计算方式，它利用了量子比特（qubit）来存储和处理信息。与传统计算机使用的二进制'
```

## 结果分析

### TP=2 为何慢于单卡（符合预期）

1. **NCCL all-reduce 开销主导**：1.5B 模型共 28 层，每层 2 次 all-reduce（self_attn + mlp），decode 时共 56 次 NCCL 通信/step。模型小，通信占比高。

2. **Memory-bound 工作负载**：小 batch（bs=3）decode 是内存带宽瓶颈，非算力瓶颈。TP 能分摊权重计算，但 56 次 all-reduce 的延迟叠加大于权重减半带来的带宽节省。

3. **Load-then-shard**：当前实现各 rank 先加载完整模型权重，再就地替换分片权重。VRAM 未达理想 50% 减半（两张卡均为 3.57 GB ≈ 单卡 3.58 GB），原因是旧权重张量的 CUDA 缓存在 peak 测量窗口内未释放。

4. **TP 适合大 batch / 大模型 / prefill**：在 batch=64 或模型 ≥ 7B 时，单卡 OOM 而 TP 可以继续扩展，这才是 TP 的实际收益场景。

### PP 为何慢于单卡

Pipeline Parallel（device_map=balanced）每条 prompt 顺序通过所有层，层间跨 GPU 传输激活张量，但因是串行 generate（非 pipeline bubble 利用），本质上是增加了跨 GPU 传输开销而无并行收益。VRAM 确实减半（2.03 + 1.56 ≈ 3.59 GB，各自承载约 50% 层的权重）。

## 验收标准对照

| 验收标准 | 状态 |
|---------|------|
| 实现真 TP（含 NCCL all-reduce，非 PP 别名） | ✅ |
| 与 PP / Replica 明确区分 benchmark 口径 | ✅ |
| 正确性：TP=2 greedy 输出与单卡一致 | ✅ |
| 干跑测试（无 GPU/NCCL）全部通过 | ✅ 13/13 passed |

## 局限性

- 未测 TTFT / TPOT（顺序 generate 不适用）
- 1.5B 模型 TP=2 吞吐低于单卡，属正常结果；7B 大 batch 场景下结论可能不同
- VRAM 未测量 shard 后稳态用量（只测 peak，含初始化 + 全量加载阶段）
- 通信时间未单独测量（all-reduce 占 decode step 比例未剖析）
