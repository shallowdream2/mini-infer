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
- **Prefix Caching**：block-level prefix hash + LRU eviction，共享前缀命中时 TTFT −22%
- **Speculative Decoding**：0.5B draft + 7B target，modified rejection sampling，acceptance_rate 55.85%
- **CUDA Graph**：decode_batch 静态捕获 + replay，1.5B 模型 bs=1 延迟 −28.9%
- **Flash Decoding / Split-K**：Triton split-K attention kernel，1.5B seq=4096 延迟 3.31× vs 标准 Triton kernel，SM 利用率 9% → 103%
- **Tensor Parallelism**：Megatron-LM 风格，column/row parallel 权重切分 + NCCL all-reduce forward hook，TP=2 greedy 输出与单卡完全一致
- **MLA（Multi-head Latent Attention）**：DeepSeek-V2/V3 架构，latent cache 压缩 56.25% vs GQA，矩阵吸收优化
- **PD 解耦（Disaggregated Prefill/Decode）**：同机双进程原型，KV 序列化传输，TTFT 三段分解（prefill/transfer/decode）
- **量化推理**（Phase 16）：int8 权重存储 + W8A8 / mixed fallback，1.5B 权重显存 −32.4%，greedy token match 71.8%，decode 全量 fallback
- **MoE + Expert Parallelism**（Phase 17）：synthetic MoE + 2-GPU EP，dense oracle 对齐，正式 benchmark `EP / dense = 1.891x`
- **True Expert Sharding**（Phase 18）：2 卡 `per-rank local expert shard`，正式 benchmark `shard_ratio = 0.5002`，`EP / dense = 1.916x`
- **Non-Padded Expert Dispatch / EP 通信闭环**（Phase 19）：`ep_packed_bytes_per_layer = ep_ideal_bytes_per_layer`，正式 benchmark `EP packed / dense = 2.323x`，`EP packed / EP padded = 1.217x`

项目面向单机 2 × RTX 4090 环境；dense 主线 benchmark 以 Qwen2.5-7B-Instruct（float16）为主，Phase 16 量化 benchmark 以 Qwen2.5-1.5B-Instruct 为主，Phase 17-19 的 MoE / EP benchmark 为 synthetic layer-level workload。当前主线实现已完成到 Phase 19；下一步待进入 Phase 20 的 infer-plan。

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
engine.py           LLMEngine：continuous batching 主循环（Phase 8 HTTP 接口，Phase 9 chunked prefill，Phase 10 prefix cache，Phase 12 CUDA Graph）
  ├── scheduler.py      Scheduler：waiting/running/swapped/prefilling 队列（Phase 7 preemption，Phase 9）
  ├── kv_cache.py       KVCacheManager：Paged KV Cache（BlockTable + FreeBlockPool + swap_out/in + Prefix Cache）
  ├── attention.py      PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
  └── model_runner.py   ModelRunner：prefill + batch decode 执行（Phase 12 graph capture/replay）
quantization.py     QuantLinear + quantize_model（Phase 16，int8 权重存储 + W8A8 / mixed fallback）
moe_layer.py        TopKRouter / MoELayer / EPMoELayer（Phase 17-18，synthetic MoE + dispatch/gather + true expert shard）
moe_model.py        SyntheticMoEConfig / SyntheticMoEModel（Phase 17-18）
ep_engine.py        2-GPU EPEngine（Phase 17-18，mp.spawn + NCCL all-to-all + rank-local shard handoff）

async_engine.py     AsyncEngine：后台线程 step loop + asyncio.Queue（Phase 8）
server.py / serve.py  FastAPI HTTP server + CLI 启动（Phase 8/9）
spec_engine.py      SpecEngine：draft + target 双模型 speculative decoding（Phase 11）
replica_engine.py   ReplicaEngine：双卡数据并行
tp_engine.py        TPEngine：真 Tensor Parallel（Phase 13，NCCL all-reduce）
triton_flash_decode.py  Flash Decoding split-K kernel（Phase 12.5）
mla_attention.py    MLA 三种实现（Phase 14）
pd_engine.py / pd_worker.py  PD 解耦入口与 worker（Phase 15）
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

# 最简单的功能验证：单命令自动起临时 dry-run 服务并进入聊天
conda run --no-capture-output -n ai-infra python quick_chat.py

# 最简单的真实模型聊天：默认尝试本机 Qwen2.5-7B-Instruct 本地目录
conda run --no-capture-output -n ai-infra python quick_chat.py --real

# 如果默认目录不同，可显式指定
conda run --no-capture-output -n ai-infra python quick_chat.py --real --model-path /path/to/Qwen2.5-7B-Instruct

# 如果 cuda:0 忙，可以显式切到另一张卡
conda run --no-capture-output -n ai-infra python quick_chat.py --real --device cuda:1

# 等价写法
conda run --no-capture-output -n ai-infra python chat.py --quick-dry-run

# 如果你已经手动起好了服务，再连现有服务聊天
# 注意：chat.py 是交互程序；conda run 需要加 --no-capture-output 才能正常读 stdin
conda run --no-capture-output -n ai-infra python chat.py
# 支持 /clear、/history、/help、exit

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

### Phase 10-12 benchmark

```bash
# Prefix Caching（Phase 10）
# dry_run：只验证 hit/miss 路径与 cache 行为
conda run -n ai-infra python benchmarks/benchmark_prefix_cache.py --dry_run

# 真实 GPU：建议先用 0.5B 做开发/验证
conda run -n ai-infra python benchmarks/benchmark_prefix_cache.py \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775 \
    --batch_size 8 --max_new_tokens 64

# Speculative Decoding（Phase 11）
# dry_run：验证 rollback / rejection sampling / 主循环
conda run -n ai-infra python benchmarks/benchmark_spec.py --dry_run

# 真实 GPU：draft=0.5B(cu0), target=7B(cu1)，同时跑 target-only baseline
conda run -n ai-infra python benchmarks/benchmark_spec.py \
    --draft auto --target auto --K 4 --max_new_tokens 64 --target_only

# CUDA Graph（Phase 12）
# 1.5B 是当前主要开发/验证模型；7B 可作为最终确认
conda run -n ai-infra python benchmarks/benchmark_cuda_graph.py \
    --model ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306 \
    --num-kv-heads 2 --head-dim 128 --num-layers 28

# Flash Decoding（Phase 12.5，无需模型权重）
# 1.5B 配置（12Q/2KV heads）
conda run -n ai-infra python benchmarks/benchmark_flash_decode.py

# 7B 配置（28Q/4KV heads）
conda run -n ai-infra python benchmarks/benchmark_flash_decode.py \
    --num-q-heads 28 --num-kv-heads 4 --skip-reference

# 正确性测试
conda run -n ai-infra python -m pytest tests/test_flash_decode.py -v
```

# Tensor Parallel（Phase 13）
```bash
# TP benchmark 主要开发模型是 1.5B；不要复用上文 7B 的 MODEL 变量
export TP_MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
export HF_HUB_OFFLINE=1

# 单卡 baseline
conda run -n ai-infra python benchmarks/benchmark_tp.py --model $TP_MODEL --mode single

# Pipeline Parallel（HF device_map=balanced）
conda run -n ai-infra python benchmarks/benchmark_tp.py --model $TP_MODEL --mode pp

# TP 功能验证（mp.spawn，计时包含进程启动 + 模型加载，不作为最终吞吐口径）
conda run -n ai-infra python benchmarks/benchmark_tp.py --model $TP_MODEL --mode tp

# Tensor Parallel TP=2（torchrun，真实 NCCL 通信）
conda run -n ai-infra torchrun --nproc_per_node 2 benchmarks/benchmark_tp.py --model $TP_MODEL --mode torchrun_tp

# TP 正确性测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_tp_engine.py -v
```

# MLA（Phase 14）
```bash
# 理论 KV cache 大小对比（无需模型权重）
conda run -n ai-infra python benchmarks/benchmark_mla.py --section 1

# 真实模型 GPU 显存测量（需要 DeepSeek-V2-Lite）
conda run -n ai-infra python benchmarks/benchmark_mla.py --section 2

# 三种实现单步 decode 延迟对比（需要 DeepSeek-V2-Lite）
conda run -n ai-infra python benchmarks/benchmark_mla.py --section 3

# MLA 测试（CPU 部分无需 GPU，GPU 测试需要 DeepSeek-V2-Lite）
conda run -n ai-infra python -m pytest tests/test_mla_attention.py -v
```

# PD 解耦（Phase 15）
```bash
# 理论 KV 传输大小（无需模型权重、无需 GPU）
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_pd_disagg.py --section 1

# 端到端正确性验证（需要 Qwen2.5-1.5B）
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_pd_disagg.py --section 2

# TTFT 分解（prefill / transfer / decode）
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_pd_disagg.py --section 3

# PD 解耦测试（7 个 dry_run 测试，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_pd_disagg.py -v
```

# 量化推理（Phase 16）
```bash
# 正式对照 benchmark（Qwen2.5-1.5B）
export QUANT_MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
export HF_HUB_OFFLINE=1

TOKENIZERS_PARALLELISM=false conda run -n ai-infra python benchmarks/benchmark_quant.py \
    --model $QUANT_MODEL --compare --batch-size 4 --max-new-tokens 50 \
    --warmup-iters 2 --bench-iters 3 --linear-warmup-iters 5 --linear-bench-iters 20

# 量化主线相关测试
conda run -n ai-infra python -m pytest \
    tests/test_quantization.py \
    tests/test_benchmark_quant.py \
    tests/test_model_runner_quant.py \
    tests/test_smoke.py \
    tests/test_engine.py -q
```

# MoE + Expert Parallelism / True Expert Sharding / Non-Padded Communication（Phase 17-19）
```bash
# 正式对照 benchmark（synthetic MoE，1-GPU dense vs 2-GPU ep_padded / ep_packed）
conda run -n ai-infra python benchmarks/benchmark_moe.py \
    --compare \
    --batch-size 4 \
    --seq-len 16 \
    --hidden-size 512 \
    --intermediate-size 1024 \
    --num-experts 8 \
    --top-k 2 \
    --dtype float16 \
    --warmup 2 \
    --runs 5 \
    --src-rank 1

# dry_run：验证 benchmark 主流程、参数量统计和通信公式
conda run -n ai-infra python benchmarks/benchmark_moe.py --mode dense --dry-run
conda run -n ai-infra python benchmarks/benchmark_moe.py --mode ep --dry-run --src-rank 1
```

Phase 17 正式结果：dense `22622.14 tok/s`，EP `42778.41 tok/s`，`EP / dense = 1.891x`。  
Phase 18 正式结果：dense `22462.79 tok/s`，EP `43036.02 tok/s`，`EP / dense = 1.916x`，`shard_ratio = 0.5002`，`max_abs_diff = 0.000000`。
Phase 19 正式结果：dense `22552.86 tok/s`，`ep_padded` `43046.93 tok/s`，`ep_packed` `52400.26 tok/s`，`EP packed / dense = 2.323x`，`EP packed / EP padded = 1.217x`，`ep_packed_bytes_per_layer = ep_ideal_bytes_per_layer = 262144`。

### 测试命令（多数无需 GPU；最后一项需要真实 GPU）

```bash
# 基础测试
conda run -n ai-infra python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py tests/test_replica_engine.py tests/test_engine.py

# Phase 7 preemption 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_preemption.py -v

# Phase 8 HTTP server 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_server.py -v

# Phase 9 Chunked Prefill 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_chunked_prefill.py -v

# Phase 10 Prefix Cache 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_prefix_cache.py -v

# Phase 11 Speculative Decoding 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_spec_engine.py -v

# Phase 12 CUDA Graph 测试（dry_run，无需 GPU）
conda run -n ai-infra python -m pytest tests/test_cuda_graph.py -v

# Phase 16 量化测试（CPU + GPU 混合；部分真实路径需要 GPU）
conda run -n ai-infra python -m pytest tests/test_quantization.py tests/test_benchmark_quant.py tests/test_model_runner_quant.py -v

# Phase 17-18 MoE / EP / true-sharding 测试（含 2-GPU 路径）
conda run -n ai-infra python -m pytest tests/test_benchmark_moe.py tests/test_moe.py tests/test_tp_engine.py -q

# Phase 6 GPU 测试（需要真实 GPU）
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
  quantization.py        QuantLinear / quantize_model（Phase 16，int8 权重存储 + W8A8 / mixed fallback）
  moe_layer.py           Phase 17-18 synthetic MoE 层（TopKRouter / MoELayer / EPMoELayer + true expert shard）
  moe_model.py           Phase 17-18 synthetic MoE 模型与配置
  ep_engine.py           Phase 17-18 2-GPU EPEngine（mp.spawn + NCCL all-to-all + rank-local shard handoff）
  engine.py              LLMEngine：continuous batching 主循环（Phase 8 HTTP 接口，Phase 9 chunked prefill）
  async_engine.py        AsyncEngine：后台线程 step loop + asyncio.Queue（Phase 8）
  openai_schema.py       OpenAI Chat Completions API Pydantic 模型（Phase 8）
  server.py              FastAPI HTTP server（Phase 8/9）
  replica_engine.py      ReplicaEngine（双卡数据并行）
  pp_engine.py           PPEngine（HF Pipeline Parallel，测量用）
  tp_engine.py           TPEngine：真 Tensor Parallel 引擎（Phase 13，mp.spawn，重写自旧 PP 别名）
  tp_model_runner.py     TensorParallelModelRunner：权重切分 + all-reduce hook（Phase 13）
  mla_attention.py       MLA 注意力三种实现（Phase 14：MLAAttentionNaive / MLAAttentionLatentCache / MLAAttentionAbsorbed）
  pd_engine.py           PDEngine：PD 解耦入口（Phase 15）
  pd_worker.py           PrefillWorker / DecodeWorker + KV 传输辅助（Phase 15）
  triton_attn.py         Triton decode attention kernel（Phase 6.5，实验性）
  triton_flash_decode.py Flash Decoding split-K kernel（Phase 12.5，实验性，dense KV）
  spec_engine.py         SpecEngine：Speculative Decoding 引擎（Phase 11，draft+target 双模型）

serve.py                 HTTP server CLI 启动脚本（Phase 8/9）
chat.py                  高级聊天入口（薄包装，实际实现位于 mini_infer/clients/chat_client.py）
quick_chat.py            一键快速聊天入口（薄包装，默认自动起临时 dry-run 服务）
mini_infer/clients/chat_client.py  聊天客户端实现（HTTP 调用、SSE、quick mode）

benchmarks/
  benchmark_hf.py        HuggingFace Transformers baseline
  benchmark_mini.py      mini-infer 当前主线单卡 benchmark（Phase 6 路径，含 TTFT/TPOT）
  benchmark_flash.py     mini-infer vs HF baseline benchmark（Phase 6，含 --compare）
  benchmark_multi_gpu.py 双卡 benchmark（replica/pp）
  benchmark_triton.py    Triton kernel latency 对比 benchmark（Phase 6.5）
  benchmark_preemption.py Phase 7 preemption swap latency + 吞吐回归测试
  benchmark_server.py    Phase 8 HTTP API benchmark（TTFT/TPOT/并发吞吐）
  benchmark_chunked_prefill.py  Phase 9 Chunked Prefill benchmark（ITL spike / TTFT 对比）
  benchmark_prefix_cache.py    Phase 10 Prefix Cache benchmark（miss/hit TTFT 对比，支持 --dry_run）
  benchmark_spec.py      Phase 11 Speculative Decoding benchmark（acceptance_rate / speedup 对比）
  benchmark_cuda_graph.py  Phase 12 CUDA Graph benchmark（eager vs graph，逐 batch_size 延迟对比）
  benchmark_flash_decode.py Phase 12.5 Flash Decoding benchmark（seq_len sweep，split-K vs triton_65 vs flash_attn）
  benchmark_tp.py        Phase 13 Tensor Parallel benchmark（single/pp/tp/torchrun_tp；其中 tp 仅用于功能验证）
  benchmark_mla.py       Phase 14 MLA benchmark（理论 KV cache 对比 / 真实显存 / 三种实现延迟）
  benchmark_pd_disagg.py Phase 15 PD 解耦 benchmark（理论 KV 大小 / 端到端正确性 / TTFT 三段分解）
  benchmark_quant.py     Phase 16 量化 benchmark（prefill / decode / e2e，对照 fp16，含 `_int_mm` / fallback 统计）
  benchmark_moe.py       Phase 17-19 synthetic MoE / EP benchmark（dense vs ep_padded vs ep_packed，对照吞吐、通信公式、shard_ratio）
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
  test_prefix_cache.py     Phase 10 Prefix Cache 测试（dry_run，miss/hit/evict/preemption）
  test_spec_engine.py      Phase 11 Speculative Decoding 测试（rollback_to / rejection sampling / dry_run，13 tests）
  test_cuda_graph.py       Phase 12 CUDA Graph 测试（dry_run：graph pool 为空时降级、_find_padded_bs 边界）
  test_flash_decode.py     Phase 12.5 Flash Decoding 正确性测试（21 tests：GQA、empty split、非整除 seq_len）
  test_tp_engine.py        Phase 13 Tensor Parallel 测试（dry_run，13 tests：col/row shard、数学等价、attn 属性、mock all-reduce）
  test_mla_attention.py    Phase 14 MLA 测试（CPU + GPU，latent cache 大小和数学等价）
  test_pd_disagg.py        Phase 15 PD 解耦测试（dry_run + GPU，7 tests：KVPayload/Queue/extract/rebuild/engine）
  test_quantization.py     Phase 16 量化 contract / 数值路径测试
  test_benchmark_quant.py  Phase 16 benchmark 口径与输出结构测试
  test_model_runner_quant.py Phase 16 ModelRunner 量化接入顺序测试
  test_moe.py             Phase 17-18 MoE / EP / true-sharding 数学路径与 2-GPU 边界测试
  test_benchmark_moe.py   Phase 17-19 benchmark 口径、compare 路径、参数统计与计时窗口测试
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
| Phase 10 | Prefix Caching（block-level hash + LRU，TTFT −22% @ 1-block prefix）| ✅ 完成 |
| Phase 11 | Speculative Decoding（Qwen2.5-0.5B draft + 7B target，rejection sampling，acceptance_rate=55.85%）| ✅ 完成 |
| Phase 12 | CUDA Graph（decode_batch 静态捕获，消除 Python dispatch 开销；1.5B bs=1 +28.9%）| ✅ 完成 |
| Phase 12.5 | Flash Decoding（Split-K，Triton 实现；1.5B seq=4096 3.31× vs triton_65）| ✅ 完成 |
| Phase 13 | Tensor Parallelism（真 TP，NCCL all-reduce；1.5B TP=2 正确性验证通过，tp=2 76.5 tok/s vs single 98.0）| ✅ 完成 |
| Phase 14 | MLA（Multi-head Latent Attention，DeepSeek 架构；latent cache 56.25% vs GQA，10 tests pass）| ✅ 完成 |
| Phase 15 | PD 解耦（同机双进程原型；greedy 输出一致，TTFT 三段分解：prefill 12.3ms/transfer≈14.7ms/decode 519ms；1.19× overhead vs unified）| ✅ 完成 |
| Phase 16 | 量化推理（int8 权重存储 + W8A8 / mixed fallback；1.5B 权重显存 −32.4%，greedy token match 71.8%，decode 100% fallback）| ✅ 完成 |
| Phase 17 | MoE + Expert Parallelism（synthetic MoE + 2-GPU EP；dense 22622.14 tok/s，EP 42778.41 tok/s，`EP / dense = 1.891x`，`max_abs_diff = 0.000000`）| ✅ 完成 |
| Phase 18 | True Expert Sharding（2 卡 `per-rank local expert shard`；dense 22462.79 tok/s，EP 43036.02 tok/s，`EP / dense = 1.916x`，`shard_ratio = 0.5002`）| ✅ 完成 |
| Phase 19 | Non-Padded Expert Dispatch / EP 通信闭环（`ep_packed_bytes_per_layer = ep_ideal_bytes_per_layer = 262144`；dense 22552.86 tok/s，`ep_padded` 43046.93 tok/s，`ep_packed` 52400.26 tok/s）| ✅ 完成 |

下一阶段：

- Phase 20：待 infer-plan

后续阶段验收口径详见 `CLAUDE.md`。
