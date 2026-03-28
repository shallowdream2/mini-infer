# mini-infer

> 一个**从零实现 LLM 推理系统关键机制**的学习型推理引擎项目。
> 不停留在"会调用模型"，而是亲手实现并验证 **Paged KV Cache、Continuous Batching、PagedAttention、Chunked Prefill、Prefix Caching、Speculative Decoding、CUDA Graph、Flash Decoding、Tensor Parallelism、MLA、MoE Expert Parallelism** 等核心能力，并通过 benchmark 分析它们的收益、代价与适用边界。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-12.1-green)
![Tests](https://img.shields.io/badge/tests-207%20dry--run%20pass-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 项目定位

`mini-infer` 不是一个只追求"能跑"的 demo，也不是直接对标生产级 vLLM 的工业 serving 框架。
它的定位是：

- **学习导向**：把主流推理系统中的关键优化逐步拆开、实现、验证
- **工程导向**：提供统一的引擎接口、服务接口、测试和 benchmark
- **研究导向**：回答"某个优化为什么有效、在什么条件下有效、代价是什么"

因此，它既适合：

- AI Infra / 推理方向学习与面试准备
- 作为个人推理引擎作品集项目
- 作为后续扩展成更完整 serving runtime 的基础骨架

---

## 你能在这个项目中看到什么

这个项目围绕一条清晰主线展开：

**1. 运行时基础能力**
从最小推理链路开始，逐步构建 continuous batching、KV cache、调度器和批量 decode。

**2. 关键性能优化机制**
实现真实的 PagedAttention、Chunked Prefill、Prefix Caching、CUDA Graph、Flash Decoding、Speculative Decoding 等核心能力。

**3. 扩展能力**
逐步走向 Tensor Parallelism、MLA、Prefill/Decode 解耦、量化、MoE Expert Parallelism。

换句话说，这不是"把一堆论文点拼在一起"，而是沿着**现代 LLM 推理系统演化路径**，一步一步把骨架搭出来。

---

## 核心成果

| 技术 | 关键数据 |
|------|---------|
| **True PagedAttention**（flash_attn block_table） | batch=8 吞吐达到 HF Transformers **100%**（406 tok/s） |
| **Chunked Prefill** | ITL spike 降低 **57%–67%** |
| **Prefix Caching**（block-level hash + LRU） | 共享前缀 TTFT **−22%** |
| **Speculative Decoding**（0.5B draft + 7B target） | acceptance rate **55.85%** |
| **CUDA Graph**（decode_batch 静态捕获） | 1.5B bs=1 decode 延迟 **−28.9%** |
| **Flash Decoding**（Triton split-K） | seq=4096 延迟 **3.31×** vs 标准 Triton，SM 利用率 9%→103% |
| **Tensor Parallelism**（NCCL all-reduce，Megatron-LM 风格） | TP=2 greedy 输出与单卡**完全一致** |
| **MLA**（DeepSeek-V2/V3 架构） | latent cache 体积 **−56.25%** vs GQA |
| **W8A8 量化**（per-channel int8 + mixed fallback） | 权重显存 **−32.4%**，greedy match 71.8% |
| **MoE Expert Parallelism**（Grouped Local Execution） | EP grouped / dense = **2.500×** |

完整 benchmark 数据与复现命令见 [docs/benchmarks.md](docs/benchmarks.md)。

---

## 快速开始

### 安装

```bash
git clone https://github.com/psmarter/mini-infer
cd mini-infer
pip install -e ".[serve]"        # 基础依赖 + HTTP serving
pip install -e ".[serve,dev]"    # 含测试工具
pip install -e ".[all]"          # 安装全部可选依赖
```

如需 True PagedAttention（flash_attn block_table）路径：

```bash
pip install "flash-attn>=2.5.0" --no-build-isolation
```

安装后可直接使用 CLI 命令：`mini-infer-serve` / `mini-infer-chat` / `mini-infer-demo`

---

## 先跑一个最小可用演示

### 方式 A — 无需模型权重，30 秒验证

```bash
python serve.py --dry-run --port 8000   # 或：mini-infer-serve --dry-run --port 8000
python quick_chat.py
```

### 方式 B — 功能对比演示（需要 Qwen2.5-1.5B，约 3 GB VRAM）

```bash
export MODEL=/path/to/Qwen2.5-1.5B-Instruct && export HF_HUB_OFFLINE=1

python demo.py --model $MODEL --mode quant         # FP16 vs W8A8：显存 + 文本质量
python demo.py --model $MODEL --mode cuda-graph    # Eager vs CUDA Graph：decode 延迟
python demo.py --model $MODEL --mode prefix-cache  # 冷启动 vs 前缀命中：TTFT
python demo.py --model $MODEL --mode all
```

### 方式 C — 主线 benchmark（需要 Qwen2.5-7B-Instruct）

```bash
export MODEL=/path/to/Qwen2.5-7B-Instruct && export HF_HUB_OFFLINE=1
python benchmarks/benchmark_flash.py --model $MODEL --batch-size 8 --compare
```

---

## 服务化能力

`mini-infer` 提供 OpenAI Chat Completions 子集兼容接口：

- `GET /v1/models`
- `POST /v1/chat/completions`（streaming + non-streaming）
- 通过 `AsyncEngine` 实现后台 step loop
- 多并发请求自动合并进同一 decode batch

**启动服务：**

```bash
mini-infer-serve --dry-run --port 8000                           # 无需模型权重
mini-infer-serve --model /path/to/model --port 8000              # 真实模型
mini-infer-serve --model /path/to/model --chunk-prefill-size 256 # 开启 Chunked Prefill
```

**curl 快速验证：**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-infer","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

Python 完整示例（non-streaming / streaming / 多轮对话）见 [`examples/openai_client.py`](examples/openai_client.py)。

---

## 架构总览

```mermaid
graph TD
    A["HTTP / CLI 请求"] --> B["AsyncEngine\n后台 step loop"]
    B --> C["LLMEngine\nContinuous Batching 主循环"]
    C --> D["Scheduler\nwaiting / running / swapped / prefilling"]
    C --> E["KVCacheManager\nBlockTable + FreeBlockPool + Prefix Cache"]
    C --> F["ModelRunner\nprefill + decode_batch"]
    F --> G["PagedAttention\nflash_attn block_table"]
    F --> H["CUDA Graph\ndecode replay"]
    F --> I["QuantLinear\nW8A8 / mixed fallback"]

    subgraph dist ["分布式扩展"]
        J["TPEngine\nNCCL all-reduce (Phase 13)"]
        K["EPEngine\nMoE all-to-all (Phase 17–21)"]
    end

    subgraph algo ["算法扩展"]
        L["SpecEngine\ndraft + target (Phase 11)"]
        M["PDEngine\nPrefill/Decode split (Phase 15)"]
    end

    C --> dist
    C --> algo
```

详细模块说明见 [docs/architecture.md](docs/architecture.md)。

---

## 当前实现路线

**Runtime 基础** — Paged KV Cache、Continuous Batching、Scheduler（四队列）、Preemption

**性能优化** — True PagedAttention（100% HF baseline）、Chunked Prefill（ITL −57%）、Prefix Caching（TTFT −22%）、Speculative Decoding（acceptance 55.85%）、CUDA Graph（延迟 −28.9%）、Flash Decoding（3.31×）

**扩展能力** — HTTP serving（OpenAI 兼容）、Tensor Parallelism（NCCL）、MLA（DeepSeek-V2/V3）、PD 解耦、W8A8 量化（显存 −32.4%）、MoE EP Grouped Execution（2.500×）

<details>
<summary>展开完整 21 阶段列表</summary>

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 单卡最小推理链路 | ✅ |
| Phase 2 | Paged KV Cache + Prefill/Decode 分离 + Continuous Batching | ✅ |
| Phase 3 | gather_batch_kv 向量化 + DynamicCache（batch=8 吞吐 88.4% HF） | ✅ |
| Phase 4 | 双卡扩展（Replica + HF Pipeline Parallel） | ✅ |
| Phase 5 | Profiling + 技术总结 | ✅ |
| Phase 6 | True PagedAttention（flash_attn block_table，100% HF） | ✅ |
| Phase 6.5 | Triton decode attention kernel | ✅ |
| Phase 7 | Preemption + Priority Scheduling | ✅ |
| Phase 8 | OpenAI Chat Completions 兼容 HTTP API | ✅ |
| Phase 9 | Chunked Prefill（ITL spike −57%） | ✅ |
| Phase 10 | Prefix Caching（TTFT −22%） | ✅ |
| Phase 11 | Speculative Decoding（acceptance_rate 55.85%） | ✅ |
| Phase 12 | CUDA Graph（decode 延迟 −28.9%） | ✅ |
| Phase 12.5 | Flash Decoding（3.31× vs triton_65） | ✅ |
| Phase 13 | Tensor Parallelism（NCCL，Megatron-LM 风格） | ✅ |
| Phase 14 | MLA（DeepSeek-V2/V3，latent cache −56.25%） | ✅ |
| Phase 15 | PD 解耦（同机双进程，KV 传输） | ✅ |
| Phase 16 | W8A8 量化（权重显存 −32.4%） | ✅ |
| Phase 17 | MoE + Expert Parallelism（1.891×） | ✅ |
| Phase 18 | True Expert Sharding（shard_ratio 0.5002） | ✅ |
| Phase 19 | Non-Padded EP Dispatch（2.323×） | ✅ |
| Phase 20 | EP Control Plane 收敛（2.310×） | ✅ |
| Phase 21 | Grouped Expert Execution（2.500×） | ✅ |

详细阶段说明见 [docs/phases.md](docs/phases.md)。

</details>

---

## 推荐使用方式

### 1. 当作学习型推理引擎阅读

按 phase 或模块阅读代码，理解每个优化机制的实现逻辑与设计权衡。

推荐阅读顺序：`runtime/engine.py` → `runtime/scheduler.py` → `cache/kv_cache.py` → `kernels/attention.py` → `modeling/model_runner.py`

详细模块指南见 [docs/architecture.md](docs/architecture.md)。

### 2. 当作 benchmark playground

运行 benchmark 脚本，对比某个机制前后的吞吐、TTFT、ITL、显存和正确性。

每个 phase 对应一个独立的 benchmark 脚本，大多数支持 `--dry_run` / `--compare` 参数。完整索引见 [docs/benchmarks.md](docs/benchmarks.md)。

### 3. 当作后续完整引擎的基础骨架

在现有代码上继续扩展：接入新模型、修改调度策略、添加量化方案、扩展 serving 接口或并行策略。

---

## 目录结构

```text
mini-infer/
├─ mini_infer/
│  ├─ core/            # EngineConfig、Request、SamplingParams
│  ├─ runtime/         # LLMEngine、Scheduler、AsyncEngine、SpecEngine、PDEngine
│  ├─ cache/           # KVCacheManager（BlockTable + Prefix Cache）、KV 传输
│  ├─ modeling/        # ModelRunner、量化、MLA、MoE
│  ├─ kernels/         # PagedAttention、Triton decode、Flash Decoding
│  ├─ parallel/        # TP、EP、Replica、PP
│  ├─ serving/         # FastAPI server、OpenAI schema
│  ├─ clients/         # 交互式聊天客户端
│  └─ cli/             # console scripts（mini-infer-serve/chat/demo）
├─ benchmarks/         # 每个 phase 对应一个 benchmark 脚本（21 个）
├─ tests/              # 测试套件（35+ 模块，287 tests）
├─ examples/           # API 使用示例（openai_client.py / local_chat.py）
├─ docs/               # 架构、阶段与 benchmark 说明
├─ serve.py            # HTTP server 入口（委托给 cli/serve.py）
├─ quick_chat.py       # 一键聊天（委托给 cli/chat.py）
├─ demo.py             # 功能对比演示（委托给 cli/demo.py）
└─ pyproject.toml
```

---

## 设计边界

这个项目当前更接近：

- **高质量学习型推理系统**
- **研究/面试/作品集友好的工程项目**
- **可继续向完整 serving runtime 演进的基础版引擎**

它当前**还不是**：

- 完整生产级 serving 框架
- 支持多模型、多租户、复杂调度策略的成熟系统
- 面向线上 SLA 的工业级部署方案

这不是缺点，反而是它的价值所在：
你可以清晰看到每个模块为什么存在、如何演进，而不是被成熟框架的复杂度淹没。

---

## 扩展方向

后续扩展方向与和生产级框架的 gap 说明见 [docs/roadmap.md](docs/roadmap.md)。

---

## 测试与质量保证

```bash
make test-fast    # 不需要 GPU，207 tests，约 7s（推荐作为最小验证门槛）
make test         # 全量测试，需要 GPU，约 50s
make test-gpu     # 仅 GPU 专项（paged_attention / Triton / Flash Decoding）
```

建议：
- `test-fast` 作为 PR / commit 最小门槛
- `test` 作为本地全量验证
- 外部环境可使用 `PYTHON=python make test-fast` 覆盖默认 conda 路径

大多数测试支持 `dry_run` 模式，不依赖真实模型权重。GPU 专项测试在 RTX 4090（CUDA 12.1）上验证通过。

---

## 适合谁

这个项目特别适合：

- 想找 **AI Infra / 推理 / 系统优化** 相关岗位的人
- 想真正弄懂 **vLLM / TensorRT-LLM / Megatron-LM** 背后机制的人
- 想做一个"有深度、有代码、有结果"的**个人代表项目**的人

---

## 环境

| 依赖 | 版本 |
|------|------|
| Python | 3.10+ |
| PyTorch | 2.1.2+cu121 |
| transformers | 4.43.4 |
| flash-attn | 2.5.9.post1 |
| CUDA | 12.1 |

真实模型推理时 `block_size` 必须是 256 的倍数（`flash_attn_with_kvcache` 对齐要求）。

---

## 文档

| 文档 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 包结构、模块说明、请求生命周期、设计选择 |
| [docs/phases.md](docs/phases.md) | 21 个阶段的实现内容与演化路径 |
| [docs/benchmarks.md](docs/benchmarks.md) | 所有阶段的 benchmark 数据、命令与复现说明 |
| [docs/faq.md](docs/faq.md) | 常见问题（环境、安装、性能、与 vLLM 的区别） |
| [docs/roadmap.md](docs/roadmap.md) | 后续扩展方向与已知 gap |

---

## License

MIT
