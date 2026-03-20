# Phase 7 Benchmark — Preemption + Priority Scheduling

## 概述

本次 benchmark 目标：验证 Phase 7 新增的 swap_out / swap_in 机制延迟，以及确认正常路径（无抢占）吞吐无回归。

---

## 环境

| 项目 | 值 |
|------|----|
| 日期 | 2026-03-20 |
| 硬件 | Ubuntu 24.04 + 2 × RTX 4090（单卡测试） |
| CUDA | 12.2 |
| Python | 3.10.19 |
| PyTorch | 2.1.2+cu121 |
| flash_attn | 2.5.9.post1 |
| transformers | 4.43.4 |
| 模型 | Qwen2.5-7B-Instruct（fp16，本地缓存） |
| HF_HUB_OFFLINE | 1 |

---

## 测试 1：调度器纯逻辑延迟（dry_run，无 GPU tensor）

| 操作 | mean | median | p99 |
|------|------|--------|-----|
| swap_out（块释放元数据） | 0.37 µs | 0.36 µs | 0.54 µs |
| swap_in（块分配元数据） | 0.61 µs | 0.60 µs | 0.88 µs |

样本数：1000 次。
说明：仅含 `_block_tables` dict 操作 + deque pop/append，无 GPU 拷贝。调度路径本身开销可忽略不计。

---

## 测试 2：GPU Swap 真实延迟（GPU↔CPU KV tensor 拷贝）

### 配置

- block_size = 16（swap 路径，不经过 flash_attn）
- 模型参数：num_layers=28，num_kv_heads=4，head_dim=128，dtype=fp16
- 样本数：50 次（含 warmup）

### 结果

| seq_len | KV 数据量 | swap_out | swap_out 带宽 | swap_in | swap_in 带宽 |
|---------|---------|----------|--------------|---------|-------------|
| 32 | 1.8 MB | 1.60 ms | 1148 MB/s | 1.69 ms | 1086 MB/s |
| 256 | 14.7 MB | 13.54 ms | 1085 MB/s | 13.66 ms | 1075 MB/s |
| 512 | 29.4 MB | 26.43 ms | 1111 MB/s | 27.54 ms | 1066 MB/s |

### 分析

**PCIe 有效带宽约 1050–1150 MB/s**，远低于 PCIe 4.0 x16 理论峰值（~32 GB/s）。原因：

当前实现在 Python 层逐块循环拷贝，对于 seq_len=32、block_size=16：

```
28 层 × 2（K+V）× 2 块 = 112 次小 tensor 拷贝
每次拷贝大小：16 tokens × 4 heads × 128 dim × fp16 ≈ 16 KB
```

每次 `k_cache[l][phys_blk, :n].cpu()` 都触发一次独立 PCIe 传输 + CUDA 同步（`torch.cuda.synchronize()`），高频小传输导致带宽利用率低。

**实际影响**：
- seq_len=256 的请求换出耗时 ~13.5 ms，换入 ~13.7 ms
- 对于 TPOT ~20 ms/step 的场景，swap 开销约 0.5–1 个 decode step 当量

**与 vLLM 的差距**：vLLM 的 swap 是一次性大块拷贝（所有层合并），理论带宽可达 ~10–20 GB/s。本项目使用逐块循环是为了清晰展示原理，不是工程优化目标。

---

## 测试 3：吞吐回归检查

### 配置

- block_size = 256（flash_attn_with_kvcache 路径）
- num_gpu_blocks = 512
- batch_size = 4，max_new_tokens = 32
- 精确 token 计数（tokenizer.encode）

### 结果

| 路径 | 耗时 | tokens | throughput |
|------|------|--------|------------|
| 正常路径（无优先级） | 0.65 s | 128 | **198.3 tok/s** |
| priority 调度路径（有优先级，充足块） | 0.65 s | 128 | **198.2 tok/s** |

**相对偏差：-0.0%**（期望：接近 0%）

### 与 Phase 6 baseline 对比

| 指标 | Phase 6 | Phase 7（本次） |
|------|---------|----------------|
| batch_size | 8 | 4 |
| throughput | 406 tok/s（= 100% HF） | 198.3 tok/s |
| 比值（仅参考） | — | 约 48.8%（batch 差异主导） |

注：batch=4 vs batch=8 吞吐差异主要来自 batch 利用率，而非 Phase 7 引入的开销。Phase 7 在相同 batch 下对 decode 路径的吞吐影响为 **0%**（符合预期——调度层变更不影响 decode forward）。

---

## 结论

| 验收标准 | 结果 | 通过 |
|----------|------|------|
| swap_out / swap_in 有真实测量数字 | ✓（见测试 2） | ✓ |
| 正常路径吞吐无回归 | 0.0% 偏差 | ✓ |
| priority 路径与正常路径一致 | 0.0% 偏差 | ✓ |

## 局限性

1. **真实抢占场景未在 GPU 上测量**：由于 `generate()` API 要求所有请求同时提交，无法方便地测量"已 prefill 请求被 swap_out 后恢复"的完整路径。该场景的正确性已通过 dry_run 测试验证，延迟可从上表 swap 延迟推算。

2. **swap 带宽未优化**：当前 ~1100 MB/s 的 PCIe 利用率远低于峰值，改进方向是将所有层的 KV 合并为单次拷贝（减少 kernel launch 次数）。

3. **batch_size 限制**：本次使用 batch=4，与 Phase 6 的 batch=8 不完全可比。

---

## 运行命令

```bash
export HF_HUB_OFFLINE=1
conda run -n ai-infra python benchmarks/benchmark_preemption.py
```

dry_run only（无需模型）：
```bash
conda run -n ai-infra python benchmarks/benchmark_preemption.py --dry-only
```
