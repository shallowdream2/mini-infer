# Mini-Infer

这个文件说明 mini-infer 项目的目标、实现状态、性能数据和快速上手方式。

## 项目概述

mini-infer 是面向 Qwen2.5 系列 decoder-only 模型的推理系统，从零实现了以下核心机制：

- **Paged KV Cache**：BlockTable + 空闲块池，消除连续 KV 缓存的显存碎片
- **Prefill / Decode 分离**：Prefill 单独 forward，Decode 步合并成 batch
- **Continuous Batching**：每步动态准入，充分复用 GPU batch 带宽
- **双卡扩展**：Replica 数据并行 + Pipeline Parallel（HF device_map）测量

项目面向单机 2 × RTX 4090 环境，模型为 Qwen2.5-7B-Instruct（float16）。

## 性能数据

环境：Ubuntu 24.04，RTX 4090，Qwen2.5-7B-Instruct float16，transformers 4.43.4。

### 单卡吞吐（vs HF baseline，max_new_tokens=128）

| 实现 | batch=1 | batch=4 | batch=8 |
|------|---------|---------|---------|
| HF Transformers baseline | 56 tok/s | ~196 tok/s | ~409 tok/s |
| Phase 1（串行 decode） | 56 tok/s | 53 tok/s | 56 tok/s |
| Phase 2（Paged KV + Batch Decode） | ~49 tok/s | ~131 tok/s | 201 tok/s |
| **Phase 3（向量化 gather + DynamicCache）** | 54 tok/s | 194 tok/s | **361 tok/s（88.4% HF）** |

Phase 1 在 batch=1 时与 HF 持平，但串行 decode 导致 batch=8 时吞吐仅为 HF 的 1/7.3。
Phase 2 引入 Batch Decode 后吞吐大幅提升，主要瓶颈转移到 `gather_batch_kv` 的逐请求 KV 复制。
Phase 3 将 `gather_batch_kv` 改为 PyTorch advanced indexing 向量化，batch=8 吞吐 +79.8%，达到 HF 的 88.4%。

### 双卡扩展（Phase 4，batch=8，max_new_tokens=128）

| 模式 | Throughput | GPU0 峰值显存 | GPU1 峰值显存 |
|------|-----------|------------|------------|
| single（Phase 3 基线） | 361.4 tok/s | 16.42 GB | — |
| replica（双卡 Replica） | 376.1 tok/s | 16.31 GB | 16.31 GB |
| tp2（HF Pipeline Parallel） | 361.5 tok/s | 7.00 GB | 8.97 GB |

Replica batch=8 仅 +4.1%：batch=8 拆成 4+4 后，每卡 batch=4 效率（194 tok/s × 2 = 388）而单卡 batch=8 已达 361（388 的 93%），scaling 空间只有 7%。
PP 吞吐持平，价值在于每卡显存减半（支持装不进单卡的大模型）。

## 架构设计

```
engine.py           LLMEngine：continuous batching 主循环
  ├── scheduler.py      Scheduler：waiting/running 队列管理
  ├── kv_cache.py       KVCacheManager：Paged KV Cache（BlockTable + FreeBlockPool）
  └── model_runner.py   ModelRunner：prefill + batch decode 执行

replica_engine.py   ReplicaEngine：双卡数据并行（ThreadPoolExecutor）
tp_engine.py        TPEngine：HF Pipeline Parallel（测量用，device_map="balanced"）
```

**decode_batch 关键步骤：**
1. `gather_batch_kv`：从 block tensor 聚合各请求的历史 KV（Phase 3 向量化）
2. DynamicCache 构造 + input 准备
3. `model_forward`：一次 batch forward，等价于同时推进所有 active 请求的 decode
4. `write_decode_kv`：将新 token KV 写回 block tensor

## 快速开始

### 环境要求

- Ubuntu 24.04 + CUDA（benchmark 需要 GPU）
- Conda 环境 `ai-infra`（Python 3.10+，transformers 4.40+，PyTorch 2.x）

### 单卡 benchmark

```bash
conda activate ai-infra
export MODEL=/path/to/Qwen2.5-7B-Instruct   # 本地根目录路径（避免 HF snapshot 问题）
export HF_HUB_OFFLINE=1

# HuggingFace baseline
python benchmarks/benchmark_hf.py --model $MODEL --batch-size 8 --max-new-tokens 128

# mini-infer
python benchmarks/benchmark_mini.py --model $MODEL --batch-size 8 --max-new-tokens 128
```

### 双卡 benchmark

```bash
# Replica 模式（双卡数据并行）
python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode replica

# Pipeline Parallel 模式（HF device_map="balanced"）
python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode tp2
```

### decode_batch profiling（Phase 5）

```bash
# 依赖上方已设置的 MODEL 和 HF_HUB_OFFLINE=1
python benchmarks/profile_decode.py --model $MODEL --batch-size 4 --decode-steps 20
```

输出 `gather_batch_kv / model_forward / write_decode_kv` 各自的 CUDA 时间占比。

### 单元测试（无 GPU）

```bash
python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py tests/test_replica_engine.py
```

## 项目结构

```
mini_infer/              核心推理代码
  engine.py              LLMEngine + continuous batching 主循环
  kv_cache.py            Paged KV Cache（BlockTable + FreeBlockPool）
  model_runner.py        ModelRunner（prefill + batch decode，含 profiler 标签）
  scheduler.py           请求调度器（waiting/running 队列）
  config.py              EngineConfig 数据类
  request.py             Request / RequestState / SamplingParams
  replica_engine.py      ReplicaEngine（双卡数据并行）
  tp_engine.py           TPEngine（HF Pipeline Parallel，测量用）

benchmarks/
  benchmark_hf.py        HuggingFace Transformers baseline
  benchmark_mini.py      mini-infer 单卡 benchmark
  benchmark_multi_gpu.py 双卡 benchmark（replica/tp2）
  profile_decode.py      decode_batch 内部 profiling（Phase 5）

tests/                   单元测试（dry_run，无 GPU）
  test_smoke.py
  test_kv_cache.py
  test_scheduler.py
  test_replica_engine.py

.claude/                 Claude Code 协作配置
CLAUDE.md                项目级协作规则
本地资料/                实验记录、博客草稿、知识整理（不上传 Git）
```

## 开发阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 单卡最小推理链路（真实模型加载、串行 decode、HF baseline 对比） | ✅ 完成 |
| Phase 2 | Paged KV Cache + Prefill/Decode 分离 + Continuous Batching | ✅ 完成 |
| Phase 3 | gather_batch_kv 向量化 + DynamicCache 迁移（batch=8 吞吐 +79.8%） | ✅ 完成 |
| Phase 4 | 双卡扩展（Replica + HF Pipeline Parallel） | ✅ 完成 |
| Phase 5 | Profiling + 技术总结 | ✅ 完成 |

## 工作环境说明

- 当前协作默认假设：已位于 Ubuntu 24.04 项目终端内
- 当前开发复用 `ai-infra` Conda 环境，不默认新建环境
- 真实模型推理、benchmark、多卡实验需要 CUDA GPU、模型权重和对应依赖
- 模型加载建议使用 `HF_HUB_OFFLINE=1` + 本地绝对路径（避免 HF snapshot 缺失触发重下载）
