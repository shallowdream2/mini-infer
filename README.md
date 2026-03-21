# Mini-Infer

这个文件说明 mini-infer 项目的目标、实现状态、性能数据和快速上手方式。

## 项目概述

mini-infer 是面向 Qwen2.5 系列 decoder-only 模型的推理系统学习项目，从零实现了以下核心机制：

- **Paged KV Cache**：BlockTable + 空闲块池，消除连续 KV 缓存的显存碎片
- **Prefill / Decode 分离**：Prefill 单独 forward，Decode 步合并成 batch
- **Continuous Batching**：每步动态准入，充分复用 GPU batch 带宽
- **双卡扩展**：Replica 数据并行 + Pipeline Parallel（HF device_map）测量
- **True PagedAttention**：flash_attn_with_kvcache block_table，decode attention 直接从 block tensor 寻址
- **Triton Decode Kernel**：手写 Triton attention kernel（online softmax + GQA），对比 flash_attn
- **Preemption + Priority Scheduling**：GPU KV swap to CPU，优先级调度，低优先级请求主动让出 GPU 块
- **OpenAI-compatible HTTP API**：FastAPI + SSE streaming，AsyncEngine 实现跨 HTTP 请求的 continuous batching

项目面向单机 2 × RTX 4090 环境，模型为 Qwen2.5-7B-Instruct（float16）。

## 性能数据

环境：Ubuntu 24.04，RTX 4090，Qwen2.5-7B-Instruct float16，transformers 4.43.4。

### 单卡吞吐（vs HF baseline，max_new_tokens=128）

| 实现 | batch=1 | batch=4 | batch=8 |
|------|---------|---------|---------|
| HF Transformers baseline | 56 tok/s | ~196 tok/s | ~406 tok/s |
| Phase 1（串行 decode） | 56 tok/s | 53 tok/s | 56 tok/s |
| Phase 2（Paged KV + Batch Decode） | ~49 tok/s | ~131 tok/s | 201 tok/s |
| Phase 3（向量化 gather + DynamicCache） | 54 tok/s | 194 tok/s | 361 tok/s（88.4% HF）|
| **Phase 6（True PagedAttention，flash_attn block_table）** | — | — | **406 tok/s（100.0% HF）** |

Phase 1 在 batch=1 时与 HF 持平，但串行 decode 导致 batch=8 时吞吐仅为 HF 的 1/7.3。
Phase 2 引入 Batch Decode 后吞吐大幅提升，主要瓶颈转移到 `gather_batch_kv` 的逐请求 KV 复制。
Phase 3 将 `gather_batch_kv` 改为 PyTorch advanced indexing 向量化，batch=8 吞吐 +79.8%，达到 HF 的 88.4%。
Phase 6 用 flash_attn_with_kvcache 的 block_table 接口替换 gather→DynamicCache→write_kv 三段，batch=8 吞吐达到 HF 的 100.0%。

### 双卡扩展（Phase 4，batch=8，max_new_tokens=128）

| 模式 | Throughput | GPU0 峰值显存 | GPU1 峰值显存 |
|------|-----------|------------|------------|
| single（历史 Phase 3 基线） | 361.4 tok/s | 16.42 GB | — |
| replica（双卡 Replica） | 376.1 tok/s | 16.31 GB | 16.31 GB |
| pp（HF Pipeline Parallel） | 361.5 tok/s | 7.00 GB | 8.97 GB |

Replica batch=8 仅 +4.1%：batch=8 拆成 4+4 后，每卡 batch=4 效率（194 tok/s × 2 = 388）而单卡 batch=8 已达 361（388 的 93%），scaling 空间只有 7%。
PP 吞吐持平，价值在于每卡显存减半（支持装不进单卡的大模型）。
注：此处 PP = Pipeline Parallel（HF device_map="balanced"，不同层在不同 GPU），不是 Tensor Parallel（同层 all-reduce）。
说明：上表保留 Phase 4 的历史测量结果；当前 `benchmarks/benchmark_multi_gpu.py` 为了对齐主线 paged decode 路径，single/replica 默认使用 `block_size=256`、`num_gpu_blocks=200`。

## 架构设计

```
engine.py           LLMEngine：continuous batching 主循环 + Phase 8 add_request/step 接口
  ├── scheduler.py      Scheduler：waiting/running/swapped 队列管理（Phase 7 preemption）
  ├── kv_cache.py       KVCacheManager：Paged KV Cache（BlockTable + FreeBlockPool + swap_out/in）
  ├── attention.py      PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
  └── model_runner.py   ModelRunner：prefill + batch decode 执行

async_engine.py     AsyncEngine：后台线程 step loop + asyncio.Queue，供 HTTP server 使用
server.py           FastAPI HTTP server：GET /v1/models，POST /v1/chat/completions
openai_schema.py    OpenAI Chat Completions API Pydantic 模型
serve.py            CLI 启动脚本（argparse + uvicorn）

replica_engine.py   ReplicaEngine：双卡数据并行（ThreadPoolExecutor）
tp_engine.py        TPEngine：HF Pipeline Parallel（测量用，device_map="balanced"）
```

**decode_batch 关键步骤（Phase 6，True PagedAttention）：**
1. `ensure_next_slot`：确保下一个 decode 位置有物理块
2. `build_block_tables`：构建 block_table / cache_seqlens
3. `paged_ctx.set`：注入共享状态（含预计算的 max_kv_len，避免 28 层各做一次 `.item()`）
4. `model_forward`：28 层 patched attention 直接从 block tensor 寻址（flash_attn_with_kvcache）
5. `advance_seq_lens`：递增各请求的逻辑序列长度计数器

**Preemption 调度流程（Phase 7）：**
1. 高优先级请求准入时若 KV 块不足，换出最低优先级的 running 请求
2. 被换出的请求若已 prefill（KV 有效）：`swap_out` → CPU 存储，加入 swapped 队列
3. 被换出的请求若未 prefill：直接释放 KV 块，放回 waiting 队尾（避免保存无效 KV）
4. 有空闲块时：从 swapped 队列 `swap_in`，重新加入 running

**HTTP API 并发架构（Phase 8）：**
```
HTTP 请求 A ─┐
HTTP 请求 B ─┤─→ LLMEngine.add_request() → 共享 Scheduler
HTTP 请求 C ─┘        ↓
                 后台线程 step loop（LLMEngine.step()）
                       ↓ decode_batch([A, B, C])
                 asyncio.Queue (per request)
                       ↓ call_soon_threadsafe
                 async generator（per HTTP connection）
```

## 快速开始

### 环境要求

- Ubuntu 24.04 + CUDA（benchmark 需要 GPU）
- Conda 环境 `ai-infra`（Python 3.10+，transformers 4.40+，PyTorch 2.x）

说明：
- AI/代理的非交互 shell 默认优先使用 `conda run -n ai-infra ...`
- 交互 shell 如果已经完成 `conda init`，也可以继续使用 `conda activate ai-infra`
- **真实模型（非 dry_run）的 `block_size` 必须是 256 的倍数**（flash_attn_with_kvcache 对齐要求），推荐使用默认值 `block_size=256`

### 单卡 benchmark

```bash
export MODEL=/path/to/Qwen2.5-7B-Instruct   # 本地根目录路径（避免 HF snapshot 问题）
export HF_HUB_OFFLINE=1

# HuggingFace baseline
conda run -n ai-infra python benchmarks/benchmark_hf.py --model $MODEL --batch-size 8 --max-new-tokens 128

# mini-infer Phase 3（向量化 gather + DynamicCache）
conda run -n ai-infra python benchmarks/benchmark_mini.py --model $MODEL --batch-size 8 --max-new-tokens 128

# mini-infer Phase 6（True PagedAttention，flash_attn block_table，双 GPU 对比）
conda run -n ai-infra python benchmarks/benchmark_flash.py --model $MODEL --batch-size 8 --max-new-tokens 128 \
    --device cuda:0 --num-gpu-blocks 200 --compare --hf-device cuda:1
```

### Triton kernel benchmark（Phase 6.5，无需模型权重）

```bash
# Triton decode attention kernel vs flash_attn_with_kvcache latency 对比
conda run -n ai-infra python benchmarks/benchmark_triton.py                    # 跑所有配置（batch=1/8，seq_len=128/512/1024/2048）
conda run -n ai-infra python benchmarks/benchmark_triton.py --batch_size 1 --seq_len 512

# Triton kernel 数值正确性测试
conda run -n ai-infra python -m pytest tests/test_triton_attn.py -v
```

### 双卡 benchmark

```bash
# Replica 模式（双卡数据并行）
conda run -n ai-infra python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode replica

# Pipeline Parallel 模式（HF device_map="balanced"）
# 注：这是 Pipeline Parallel（PP），不是 Tensor Parallel（TP）
# PP：不同层在不同 GPU；TP：同一层按 head 切分 + all-reduce
conda run -n ai-infra python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode pp
```

说明：当前 `benchmark_multi_gpu.py` 中 single/replica 默认走主线 `LLMEngine` 配置，即 `block_size=256`、`num_gpu_blocks=200`。

### decode_batch profiling（Phase 6）

```bash
# 依赖上方已设置的 MODEL 和 HF_HUB_OFFLINE=1
conda run -n ai-infra python benchmarks/profile_decode.py --model $MODEL --batch-size 8 --decode-steps 30
```

Phase 6 输出只有 `model_forward` 标签，`gather_batch_kv` 和 `write_decode_kv` 已消除。

### Preemption benchmark（Phase 7，支持 dry_run）

```bash
# 完整测试（需要 Qwen2.5-7B-Instruct 权重）
export HF_HUB_OFFLINE=1
conda run -n ai-infra python benchmarks/benchmark_preemption.py

# dry_run 模式（无需模型权重，测试调度器延迟）
conda run -n ai-infra python benchmarks/benchmark_preemption.py --dry-only
```

### Phase 8 HTTP server

```bash
# dry_run 模式（无需模型权重，快速验证 API 结构）
conda run -n ai-infra python serve.py --dry-run --port 8000

# 真实模型
export MODEL=/path/to/Qwen2.5-7B-Instruct
conda run -n ai-infra python serve.py --model $MODEL --port 8000

# 测试 API（另一个终端）
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-infer","messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":64}'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-infer","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'
```

### 单元测试（无 GPU）

```bash
# 基础测试
conda run -n ai-infra python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py tests/test_replica_engine.py tests/test_engine.py

# Phase 7 preemption 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_preemption.py -v

# Phase 8 HTTP server 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_server.py -v

# Phase 6 GPU 测试（需要 2× RTX 4090）
conda run -n ai-infra python -m pytest tests/test_paged_attention.py -v
```

## 项目结构

```
mini_infer/              核心推理代码
  config.py              EngineConfig 数据类
  request.py             Request / RequestState / SamplingParams
  scheduler.py           请求调度器（waiting/running/swapped 队列，Phase 7 preemption）
  kv_cache.py            Paged KV Cache（BlockTable + FreeBlockPool + swap_out/in）
  attention.py           PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
  model_runner.py        ModelRunner（prefill + batch decode，含 profiler 标签）
  engine.py              LLMEngine：continuous batching 主循环 + Phase 8 step 接口
  async_engine.py        AsyncEngine：后台线程 step loop + asyncio.Queue（Phase 8）
  openai_schema.py       OpenAI Chat Completions API Pydantic 模型（Phase 8）
  server.py              FastAPI HTTP server（Phase 8）
  replica_engine.py      ReplicaEngine（双卡数据并行）
  pp_engine.py           PPEngine（HF Pipeline Parallel，测量用）
  tp_engine.py           向后兼容别名（TPEngine = PPEngine）
  triton_attn.py         Triton decode attention kernel（Phase 6.5，实验性）

serve.py                 HTTP server CLI 启动脚本（Phase 8）

benchmarks/
  benchmark_hf.py        HuggingFace Transformers baseline
  benchmark_mini.py      mini-infer 单卡 benchmark（Phase 3）
  benchmark_flash.py     mini-infer Phase 6 benchmark（flash_attn block_table，含 --compare）
  benchmark_multi_gpu.py 双卡 benchmark（replica/pp）
  benchmark_triton.py    Triton kernel latency 对比 benchmark（Phase 6.5）
  benchmark_preemption.py Phase 7 preemption swap latency + 吞吐回归测试
  profile_decode.py      decode_batch 内部 profiling（Phase 6）

tests/
  test_smoke.py
  test_kv_cache.py
  test_scheduler.py
  test_replica_engine.py
  test_engine.py           LLMEngine 主循环集成测试（dry_run，含 KV 耗尽、块回收、顺序保证）
  test_preemption.py       Phase 7 preemption + priority scheduling 测试（dry_run）
  test_server.py           Phase 8 HTTP server 测试（ASGI TestClient，dry_run）
  test_paged_attention.py  Phase 6 GPU 测试（需要真实 GPU）
  test_triton_attn.py      Phase 6.5 Triton kernel 正确性测试

.claude/                 Claude Code 协作配置（原始来源）
.codex/                  Codex repo-local 技能源文件与安装说明
CLAUDE.md                Claude 项目级协作规则
CODEX.md                 Codex 项目级协作规则
本地资料/                实验记录、博客草稿、知识整理（当前位于仓库内，以个人长期记录为主）
```

## 开发阶段

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 单卡最小推理链路（真实模型加载、串行 decode、HF baseline 对比） | ✅ 完成 |
| Phase 2 | Paged KV Cache + Prefill/Decode 分离 + Continuous Batching | ✅ 完成 |
| Phase 3 | gather_batch_kv 向量化 + DynamicCache 迁移（batch=8 吞吐 +79.8%） | ✅ 完成 |
| Phase 4 | 双卡扩展（Replica + HF Pipeline Parallel） | ✅ 完成 |
| Phase 5 | Profiling + 技术总结 | ✅ 完成 |
| Phase 6 | True PagedAttention（flash_attn block_table，消除 gather/write_kv，batch=8 达到 100% HF）| ✅ 完成 |
| Phase 6.5 | Triton decode attention kernel（online softmax，GQA，roofline 分析）| ✅ 完成 |
| Phase 7 | Preemption + Priority Scheduling（GPU↔CPU KV swap，优先级调度）| ✅ 完成 |
| Phase 8 | OpenAI-compatible HTTP API（FastAPI + SSE streaming，AsyncEngine）| 🔄 进行中 |

## 工作环境说明

- 当前协作默认假设：已位于 Ubuntu 24.04 项目终端内
- 当前开发复用 `ai-infra` Conda 环境，不默认新建环境
- 真实模型推理、benchmark、多卡实验需要 CUDA GPU、模型权重和对应依赖
- 模型加载建议使用 `HF_HUB_OFFLINE=1` + 本地绝对路径（避免 HF snapshot 缺失触发重下载）
- Phase 8 HTTP server 测试（`test_server.py`）使用 dry_run 模式，无需 GPU 或模型权重
- Codex 迁移资产位于 `CODEX.md` 和 `.codex/`；真正安装到 `~/.codex/skills/` 仍然是本机手动步骤
