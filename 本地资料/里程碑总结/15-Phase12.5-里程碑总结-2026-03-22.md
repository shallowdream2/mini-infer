# Phase 12.5 里程碑总结：Flash Decoding（Split-K Attention）

日期：2026-03-22

---

## 背景与目标

Phase 12（CUDA Graph）消除了 Python dispatch 开销，使 1.5B bs=1 的 decode step 延迟从 7.33ms 降到 5.21ms（+28.9%）。但这只解决了 CPU 侧的调度问题，GPU 侧 attention 计算的并行效率未触及。

Phase 6.5 的 Triton kernel（grid = batch × num_q_heads）在 decode 场景 batch=1 时，1.5B 只有 12 个 thread block，RTX 4090 有 128 个 SM，SM 利用率约 9%。序列越长，每个 thread block 处理的 KV 越多，latency 线性增长，但 SM 利用率始终只有 9%。

**目标**：用 Triton 实现 Flash Decoding（split-K），在 KV 序列长度方向切分，grid 扩展为 batch × num_q_heads × num_splits，使 SM 利用率随 seq_len 提升，长序列 decode attention latency 趋于平坦。

**范围**：纯实验性 kernel（dense KV，不接入推理主路径），基于 Phase 6.5 扩展。

---

## 已完成内容

| 产出 | 说明 |
|------|------|
| `mini_infer/triton_flash_decode.py` | split kernel + PyTorch 归约 + `flash_decode_triton()` + `auto_num_splits()` |
| `tests/test_flash_decode.py` | 21 个测试：正确性、GQA、batch>1、非整除 seq_len、空 split、自动 num_splits |
| `benchmarks/benchmark_flash_decode.py` | seq_len sweep（128-4096），4 路径对比，支持 1.5B/7B 配置 |

---

## 关键成就

### 1. split-K 延迟平坦性验证

**1.5B 配置（12Q/2KV heads，batch=1）**：

| seq_len | triton_65_ms | flash_decode_ms | spd_vs_triton65 |
|---------|-------------|----------------|-----------------|
| 128 | 0.021 | 0.064 | 0.32× |
| 512 | 0.033 | 0.064 | 0.51× |
| 1024 | 0.058 | 0.064 | 0.91× |
| 2048 | 0.112 | 0.070 | **1.60×** |
| 4096 | 0.224 | 0.068 | **3.31×** |

flash_decode 延迟：128→4096（32×）增幅仅 +6%（0.064ms→0.068ms）
triton_65 延迟：128→4096 增幅 +967%（0.021ms→0.224ms）

**7B 配置（28Q/4KV heads，batch=1）**：seq_len=4096 时 **2.57×** 于 triton_65

### 2. 盈亏平衡点量化

- 1.5B：约 seq_len≈1200（0.91× @ 1024，1.60× @ 2048）
- 7B：约 seq_len≈850（0.70× @ 512，1.17× @ 1024）
- 短序列（< 512）：split-K 开销（partial buffer 写入、kernel launch）大于并行收益

### 3. 正确性全覆盖

21 个测试全通过（数据来自 `pytest tests/test_flash_decode.py`）：
- num_splits=1 退化 vs reference：max_diff < 1e-3
- 各 (seq_len, num_splits) 组合 vs reference/flash_attn：max_diff < 1e-2
- GQA（28Q/4KV）、batch=4、非 BLOCK_N 整除 seq_len（100/513/1000）、空 split

---

## 问题与失败点

### 未达成的验收标准

原 infer-plan 设定的 flash_attn 对标指标全部未达成：

| 原标准 | 实测 | 原因 |
|--------|------|------|
| seq_len=512 ≤ 1.2× flash_attn latency | ≈ 4.6× | flash_attn 是高度优化的 CUDA C++（warp-level + SMEM tiling），Triton 研究实现无法匹敌 |
| seq_len=2048 ≤ flash_attn | ≈ 6.4× | 同上；split-K 本身不能消除 Triton vs C++ 的实现差距 |
| seq_len=4096 ≤ 0.8× flash_attn | ≈ 5.7× | 同上 |

**根因**：原计划参考路线图 "< 0.5× flash_attn" 的目标过于激进，混淆了"Triton 实现改善"与"超越高度优化 CUDA C++ 实现"两个不同层级的目标。有意义的对比对象是同为 Triton 研究实现的 Phase 6.5 kernel，在此维度 split-K 效果显著。

### 实现过程中的问题

1. **`tl.ones` 在 Triton 2.1.0 不存在**：改为 `tl.full([1], 1.0, dtype=tl.float32)`
2. **标量 pointer + block value 的 `tl.store` 报错**：写 partial_lse 时 pointer 为标量但 lse 为 `[1]` block tensor，用 `pl_base + tl.arange(0, 1)` 解决

### 未接入推理主路径

flash_decode_triton 仍是独立实验 kernel，不接入 engine/model_runner，throughput/TTFT/TPOT 指标不适用。接入主路径需要支持 Paged KV（block_table 格式），留给后续扩展。

---

## 当前状态

Phase 12.5 代码、测试、benchmark 均完成。split-K 的核心概念已验证：SM 利用率从 9% 提升到 ~103%（num_splits=11），使 decode attention latency 在长序列下趋于恒定。

---

## 下一步计划

根据 `本地资料/Claude计划/00-长期路线图.md`：

**Phase 13：Tensor Parallelism（真 TP，NCCL all-reduce）**

- 实现 column parallel（Q/K/V/gate/up projections 按 head 切分）和 row parallel（O/down projections + all-reduce）
- 目标：Qwen2.5-7B 在双卡 TP=2 下运行，与 Phase 4 的 PP/Replica 做 benchmark 口径分离对比
- 最小验收：至少一条主链路（含 NCCL all-reduce）实现并跑通，给出 throughput/latency 对比
