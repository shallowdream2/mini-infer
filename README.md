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
- **OpenAI Chat Completions 子集兼容 HTTP API**：FastAPI + SSE streaming，AsyncEngine 实现跨 HTTP 请求的 continuous batching
- **Chunked Prefill**：长 prompt 拆分 chunk 投送，decode 请求不被长 prefill 饿死（ITL spike −57%~−67%）

项目面向单机 2 × RTX 4090 环境，模型为 Qwen2.5-7B-Instruct（float16）。

**权威来源**：`CLAUDE.md` 是项目规则和当前状态的权威来源；详细技术规划见 `本地资料/Claude计划/00-长期路线图.md`；本文件仅作快速索引。

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

## 架构概览

```
engine.py           LLMEngine：continuous batching 主循环（Phase 8 HTTP 接口，Phase 9 chunked prefill）
  ├── scheduler.py      Scheduler：waiting/running/swapped/prefilling 队列（Phase 7 preemption，Phase 9）
  ├── kv_cache.py       KVCacheManager：Paged KV Cache（BlockTable + FreeBlockPool + swap_out/in）
  ├── attention.py      PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
  └── model_runner.py   ModelRunner：prefill + batch decode 执行

async_engine.py     AsyncEngine：后台线程 step loop + asyncio.Queue（Phase 8）
server.py / serve.py  FastAPI HTTP server + CLI 启动（Phase 8/9）
replica_engine.py   ReplicaEngine：双卡数据并行
tp_engine.py        TPEngine：HF Pipeline Parallel（测量用）
```

详细架构说明、关键步骤和状态机参见 `CLAUDE.md`。

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

# mini-infer 当前主线单卡 benchmark（Phase 6 路径，含 TTFT/TPOT）
conda run -n ai-infra python benchmarks/benchmark_mini.py --model $MODEL --batch-size 8 --max-new-tokens 128

# mini-infer vs HF baseline（True PagedAttention，flash_attn block_table，双 GPU 对比）
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

### Phase 8/9 HTTP server

```bash
# dry_run 模式（无需模型权重，快速验证 API 结构）
conda run -n ai-infra python serve.py --dry-run --port 8000

# 真实模型
export MODEL=/path/to/Qwen2.5-7B-Instruct
conda run -n ai-infra python serve.py --model $MODEL --port 8000

# 真实模型 + Phase 9 chunked prefill
conda run -n ai-infra python serve.py --model $MODEL --chunk-prefill-size 256 --port 8000

# 测试 API（另一个终端）
curl http://localhost:8000/v1/models

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-infer","messages":[{"role":"user","content":"你好"}],"stream":false,"max_tokens":64}'

# 流式
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-infer","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'

# OpenAI Python SDK（如 shell 中存在代理变量，先清掉）
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u http_proxy -u https_proxy \
  conda run -n ai-infra python -c "from openai import OpenAI; client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='none'); resp = client.chat.completions.create(model='mini-infer', messages=[{'role':'user','content':'hello'}], max_tokens=8); print(resp.choices[0].message.content)"

# 直接 uvicorn 启动（未注入 config 时默认回退到 dry_run）
uvicorn mini_infer.server:app --host 0.0.0.0 --port 8000

# 真实模型也可通过环境变量启动
MINI_INFER_MODEL=$MODEL uvicorn mini_infer.server:app --host 0.0.0.0 --port 8000

# 真实模型 + Phase 9 chunked prefill（直接 uvicorn）
MINI_INFER_MODEL=$MODEL MINI_INFER_CHUNK_PREFILL_SIZE=256 uvicorn mini_infer.server:app --host 0.0.0.0 --port 8000

# HTTP benchmark（Phase 8）
conda run -n ai-infra python benchmarks/benchmark_server.py --model $MODEL

# Chunked Prefill benchmark（Phase 9）
conda run -n ai-infra python benchmarks/benchmark_chunked_prefill.py --model $MODEL --chunk-size 256
```

### 单元测试（无 GPU）

```bash
# 基础测试
conda run -n ai-infra python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py tests/test_replica_engine.py tests/test_engine.py

# Phase 7 preemption 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_preemption.py -v

# Phase 8 HTTP server 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_server.py -v

# Phase 9 Chunked Prefill 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_chunked_prefill.py -v

# Phase 6 GPU 测试（需要 2× RTX 4090）
conda run -n ai-infra python -m pytest tests/test_paged_attention.py -v
```

## 项目结构

```
mini_infer/              核心推理代码
  config.py              EngineConfig 数据类
  request.py             Request / RequestState / SamplingParams
  scheduler.py           请求调度器（waiting/running/swapped/prefilling 队列，Phase 7/9）
  kv_cache.py            Paged KV Cache（BlockTable + FreeBlockPool + swap_out/in）
  attention.py           PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
  model_runner.py        ModelRunner（prefill + batch decode，含 profiler 标签）
  engine.py              LLMEngine：continuous batching 主循环（Phase 8 HTTP 接口，Phase 9 chunked prefill）
  async_engine.py        AsyncEngine：后台线程 step loop + asyncio.Queue（Phase 8）
  openai_schema.py       OpenAI Chat Completions API Pydantic 模型（Phase 8）
  server.py              FastAPI HTTP server（Phase 8/9）
  replica_engine.py      ReplicaEngine（双卡数据并行）
  pp_engine.py           PPEngine（HF Pipeline Parallel，测量用）
  tp_engine.py           向后兼容别名（TPEngine = PPEngine）
  triton_attn.py         Triton decode attention kernel（Phase 6.5，实验性）

serve.py                 HTTP server CLI 启动脚本（Phase 8/9）

benchmarks/
  benchmark_hf.py        HuggingFace Transformers baseline
  benchmark_mini.py      mini-infer 当前主线单卡 benchmark（Phase 6 路径，含 TTFT/TPOT）
  benchmark_flash.py     mini-infer vs HF baseline benchmark（Phase 6，含 --compare）
  benchmark_multi_gpu.py 双卡 benchmark（replica/pp）
  benchmark_triton.py    Triton kernel latency 对比 benchmark（Phase 6.5）
  benchmark_preemption.py Phase 7 preemption swap latency + 吞吐回归测试
  benchmark_server.py    Phase 8 HTTP API benchmark（TTFT/TPOT/并发吞吐）
  benchmark_chunked_prefill.py  Phase 9 Chunked Prefill benchmark（ITL spike / TTFT 对比）
  profile_decode.py      decode_batch 内部 profiling（Phase 6）

tests/
  test_smoke.py
  test_kv_cache.py
  test_scheduler.py
  test_replica_engine.py
  test_engine.py           LLMEngine 主循环集成测试（dry_run，含 KV 耗尽、块回收、顺序保证）
  test_preemption.py       Phase 7 preemption + priority scheduling 测试（dry_run）
  test_server.py           Phase 8 HTTP server 测试（ASGI TestClient，dry_run）
  test_chunked_prefill.py  Phase 9 Chunked Prefill 测试（dry_run，状态机 + 端到端 + KV 无泄漏）
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
| Phase 8 | OpenAI Chat Completions 子集兼容 HTTP API（FastAPI + SSE streaming，AsyncEngine）| ✅ 完成 |
| Phase 9 | Chunked Prefill（长 prefill 不阻塞 decode，ITL spike −57% @ chunk=256）| ✅ 完成 |
| Phase 10 | Prefix Caching（RadixAttention，KV 前缀共享）| ⬜ 计划中 |
| Phase 11 | Speculative Decoding（draft+target 双模型推理加速）| ⬜ 计划中 |
| Phase 12 | CUDA Graph（decode_batch 静态捕获，消除 Python dispatch 开销）| ⬜ 计划中 |
| Phase 12.5 | Flash Decoding（Split-K，长序列 attention 并行化）| ⬜ 计划中 |
| Phase 13 | Tensor Parallelism（真 TP，NCCL all-reduce）| ⬜ 计划中 |
| Phase 14 | MLA（Multi-head Latent Attention，DeepSeek 架构）| ⬜ 计划中 |
| Phase 15 | PD 解耦（Disaggregated Prefill/Decode）| ⬜ 计划中 |

后续阶段验收口径详见 `CLAUDE.md`。
