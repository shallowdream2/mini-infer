# CLAUDE.md

这个文件定义 Claude Code 在 mini-infer 仓库中的长期项目级规则、边界和工作方式。

## 项目定位

这是一个面向 `Qwen2.5` 类 `decoder-only` 模型的推理系统学习项目，实现并验证了：

- `Paged KV Cache`（BlockTable + FreeBlockPool + block tensor 池）
- `Prefill / Decode` 分离
- `Continuous Batching`（动态准入调度）
- 向量化 KV Gather（Phase 2-5，PyTorch advanced indexing）
- **True PagedAttention**（Phase 6，flash_attn_with_kvcache + block_table，decode attention kernel 直接从 block tensor 寻址）
- Triton decode attention kernel（Phase 6.5，对比 flash_attn）
- Preemption + Priority Scheduling（Phase 7，swap to CPU，优先级调度）
- 双卡扩展（Replica 数据并行 + HF Pipeline Parallel 测量）
- 真实 benchmark（对照 HF Transformers baseline，batch=8 达到 100% HF）
- **OpenAI Chat Completions 子集兼容 HTTP API**（Phase 8，FastAPI + SSE streaming，AsyncEngine continuous batching）
- **Chunked Prefill**（Phase 9，长 prefill 拆分 chunk 投送，decode 请求不被长 prefill 饿死，ITL spike −57%~−67%）
- **Prefix Caching**（Phase 10，block-level SHA-256 链式 hash + LRU eviction + ref_count，共享前缀 TTFT −22%）
- **Speculative Decoding**（Phase 11，Qwen2.5-0.5B draft + 7B target，modified rejection sampling，acceptance_rate 55.85%）
- **CUDA Graph**（Phase 12，decode_batch 静态捕获 + graph pool，1.5B bs=1 延迟 −28.9%）
- **Flash Decoding / Split-K Attention**（Phase 12.5，Triton split-K kernel，1.5B seq=4096 延迟 3.31× vs triton_65，SM 利用率 9% → 103%）
- **Tensor Parallelism**（Phase 13，真 TP，NCCL all-reduce，column/row parallel，Megatron-LM 风格权重切分）
- **MLA（Multi-head Latent Attention）**（Phase 14，DeepSeek-V2/V3 架构，latent cache 压缩 56.25% vs GQA，矩阵吸收优化）
- **PD 解耦（Disaggregated Prefill/Decode）**（Phase 15，同机双进程原型，KV 序列化传输，TTFT 三段分解：prefill 12.3ms / transfer≈14.7ms / decode 519ms）

## 当前状态

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 单卡最小推理链路 | ✅ |
| Phase 2 | Paged KV Cache + Batch Decode + Continuous Batching | ✅ |
| Phase 3 | 向量化 gather_batch_kv + DynamicCache（batch=8 达到 HF 的 88.4%）| ✅ |
| Phase 4 | 双卡扩展（Replica + HF PP）| ✅ |
| Phase 5 | Profiling + 项目收尾 | ✅ |
| Phase 6 | True PagedAttention（flash_attn block_tables，batch=8 达到 100% HF）| ✅ |
| Phase 6.5 | Triton kernel（decode attention kernel，对比 flash_attn，大厂核心路线主线）| ✅ |
| Phase 7 | Preemption + Priority Scheduling（swap to CPU，优先级调度）| ✅ |
| Phase 8 | OpenAI Chat Completions 子集兼容 HTTP API（FastAPI + streaming）| ✅ |
| Phase 9 | Chunked Prefill（长 prefill 不阻塞 decode，调度器改造）| ✅ |
| Phase 10 | Prefix Caching（block-level hash + LRU，TTFT −22% @ 1-block prefix）| ✅ |
| Phase 11 | Speculative Decoding（draft+target 双模型，rejection sampling）| ✅ |
| Phase 12 | CUDA Graph（decode_batch 静态捕获，消除 Python dispatch 开销，1.5B +28.9%）| ✅ |
| Phase 12.5 | Flash Decoding（Split-K attention，长序列并行，Triton 实现，1.5B seq=4096 3.31× vs triton_65）| ✅ |
| Phase 13 | Tensor Parallelism（真 TP，NCCL all-reduce，column/row parallel）| ✅ |
| Phase 14 | MLA（Multi-head Latent Attention，DeepSeek-V2/V3 架构）| ✅ |
| Phase 15 | PD 解耦（Disaggregated Prefill/Decode，KV 网络传输）| ✅ |

**权威来源说明**：当前状态与未来计划以本文件（`CLAUDE.md`）为权威来源；详细技术规划参考 `本地资料/Claude计划/00-长期路线图.md`；`README.md` 仅作快速索引；带日期的里程碑总结、开发日志和实验记录默认视为历史快照。

## 环境事实

- 当前默认工作前提：已经位于 Ubuntu 24.04 项目目录内
- 当前项目默认直接复用 `本地资料/环境配置/AI-Infra学习之旅-服务器环境配置.md` 中已经配置好的 `ai-infra` 环境
- Phase 8 HTTP/SDK 相关的补充环境事实优先参考 `本地资料/环境配置/phase8-env-notes.md`
- 主开发环境是 Ubuntu 24.04 + bash + Python 3.10+
- 真实运行和 benchmark 环境是 Ubuntu 24.04 + 2 × RTX 4090

**当前已安装关键版本（2026-03-22）：**

| 包 | 版本 | 备注 |
|----|------|------|
| PyTorch | 2.1.2+cu121 | CUDA 12.1 |
| transformers | 4.43.4 | >= 4.40.0 ✓ |
| flash_attn | **2.5.9.post1** | ✅ block_table 可用（Phase 6 已用） |
| Python | 3.10 | — |

**关键约束：**
- PyTorch SDPA flash_sdp 已启用 → HF 模型推理时已在用 FlashAttention kernel
- flash_attn 2.5.9.post1 的 `flash_attn_with_kvcache` 已有 `block_table` 参数，Phase 6 已使用

**本地已下载模型（2026-03-22）：**

| 模型 | 路径 | VRAM | 架构参数 | 用途 |
|------|------|------|---------|------|
| Qwen2.5-0.5B-Instruct | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775` | ~1.9 GB | 24层, 14Q/2KV heads, head_dim=64, vocab=151936 | spec draft 模型；快速 dev/test |
| Qwen2.5-1.5B-Instruct | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | ~3 GB | 28层, 12Q/2KV heads, head_dim=128, vocab=151936 | Phase 12+ 主要开发用模型 |
| Qwen2.5-7B-Instruct | `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`（**根目录**，非 snapshot） | ~18.2 GB | 28层, 28Q/4KV heads, head_dim=128, vocab=152064 | 最终 benchmark 验证；snapshot 子目录不可用 |
| DeepSeek-V2-Lite | `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/snapshots/604d5664dddd88a0433dbae533b7fe9472482de0` | ~15.2 GB（fp16，2×4090） | 27层 MoE, 16Q heads, kv_lora_rank=512, qk_rope_head_dim=64, hidden=2048 | Phase 14 MLA 验证；trust_remote_code=True 必须 |

**模型使用注意事项：**
- 所有本地模型必须配合 `HF_HUB_OFFLINE=1` 使用，防止 transformers 联网检查触发重下载
- 7B 使用根目录路径（snapshot 子目录 shard 2-4 软链接缺失）
- 0.5B 和 1.5B 使用 snapshots/ 子目录路径（完整）
- EngineConfig 参数需按模型明确指定 `num_hidden_layers / num_kv_heads / head_dim`（不可依赖默认值，默认值为 7B 参数）

## 工作方式

- 大任务先探索，再计划，再编码
- 先给可验证的最小闭环，不做大而空抽象
- 每次实现都优先给出验证方式
- 如果无法运行验证，明确说明原因和缺失条件
- 上下文过长或任务切换明显时，建议用户使用 `/clear`
- 阶段性工作完成后，应产出总结、知识沉淀或博客草稿，而不是只停留在代码层

## 代码范围

核心代码位于：

- `mini_infer/config.py` — EngineConfig 数据类
- `mini_infer/request.py` — Request / RequestState / SamplingParams
- `mini_infer/scheduler.py` — 请求调度器（waiting / running / swapped / prefilling 队列，Phase 7 preemption，Phase 9 chunked prefill）
- `mini_infer/kv_cache.py` — Paged KV Cache（BlockTable + FreeBlockPool）
- `mini_infer/attention.py` — PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
- `mini_infer/model_runner.py` — ModelRunner（prefill + batch decode，含 profiler 标签）
- `mini_infer/engine.py` — LLMEngine（continuous batching 主循环，Phase 8 add_request/step 接口，Phase 9 chunked prefill 状态机）
- `mini_infer/async_engine.py` — AsyncEngine（后台线程 step loop + asyncio.Queue，Phase 8）
- `mini_infer/openai_schema.py` — OpenAI Chat Completions API Pydantic 模型（Phase 8）
- `mini_infer/server.py` — FastAPI HTTP server（Phase 8）
- `mini_infer/replica_engine.py` — ReplicaEngine（双卡数据并行）
- `mini_infer/pp_engine.py` — PPEngine（HF Pipeline Parallel，测量用）
- `mini_infer/tp_engine.py` — TPEngine 真 TP 引擎（Phase 13，mp.spawn + 文件锁 rendezvous）
- `mini_infer/tp_model_runner.py` — TensorParallelModelRunner（Phase 13，Megatron-LM 风格权重切分 + all-reduce hook）
- `mini_infer/spec_engine.py` — SpecEngine（Phase 11，draft+target 双模型 speculative decoding）
- `mini_infer/triton_flash_decode.py` — Flash Decoding split-K kernel（Phase 12.5，实验性，密 KV，不接入主路径）
- `mini_infer/mla_attention.py` — MLA 注意力三种实现（Phase 14，MLAAttentionNaive / MLAAttentionLatentCache / MLAAttentionAbsorbed + compute_kv_cache_bytes）
- `serve.py` — HTTP server CLI 启动脚本（argparse + uvicorn，Phase 8）
- `chat.py` / `quick_chat.py` / `mini_infer/clients/chat_client.py` — 本地聊天入口与临时服务 quick mode

测试位于 `tests/`。

benchmark 位于 `benchmarks/`：
- `benchmark_hf.py` — HuggingFace Transformers baseline
- `benchmark_mini.py` — mini-infer 主线单卡 benchmark（Phase 6，含 TTFT/TPOT）
- `benchmark_flash.py` — mini-infer vs HF baseline（Phase 6，含 --compare）
- `benchmark_multi_gpu.py` — 双卡 benchmark（replica / pp）
- `benchmark_triton.py` — Triton kernel latency 对比（Phase 6.5）
- `benchmark_preemption.py` — Phase 7 preemption swap latency + 吞吐回归
- `benchmark_server.py` — Phase 8 HTTP API benchmark（TTFT / TPOT / 并发吞吐）
- `benchmark_chunked_prefill.py` — Phase 9 Chunked Prefill benchmark（ITL spike / TTFT 对比）
- `benchmark_prefix_cache.py` — Phase 10 Prefix Caching benchmark（miss/hit TTFT，对比共享前缀收益）
- `benchmark_spec.py` — Phase 11 Speculative Decoding benchmark（acceptance rate / spec vs target-only）
- `benchmark_cuda_graph.py` — Phase 12 CUDA Graph benchmark（eager vs graph decode step latency）
- `benchmark_flash_decode.py` — Phase 12.5 Flash Decoding benchmark（seq_len sweep，split-K vs triton_65 vs flash_attn）
- `profile_decode.py` — decode_batch 内部 profiling（Phase 6）

skills 位于 `.claude/skills/`。
规则位于 `.claude/rules/`。
长期计划与个人记录位于 `本地资料/`。

## Skills 使用方式

- 优先使用 `.claude/skills/` 中的长期工作流能力，而不是临时重复提示
- 需要显式调用时，在任务里直接点名对应 skill，例如 `infer-plan`、`infer-implement`、`infer-benchmark`
- 对于阶段规划、阶段总结、知识沉淀和博客写作，默认优先复用已有 skill
- 当某个 skill 变得复杂时，应继续在对应目录中补充模板、清单和辅助说明文件，而不是把所有要求堆回一个文档

## 阶段工作流

每个阶段依次执行 7 步：infer-plan → infer-implement → infer-review → infer-benchmark → infer-summarize → infer-blog → infer-archive。步骤细节见 `.claude/rules/workflow.md`。

**行为规则（每次对话必须遵守）**

1. 对话开始时，说明当前步骤并主动引导下一步。
2. 一次只执行一个 skill，完成后等待用户指令。
3. infer-plan 完成（验收标准和步骤已确认）前不得进入 infer-implement；infer-plan 可先运行以输出环境验证命令，用户确认环境就绪后再正式开始 implement。
4. infer-benchmark 要求进度表 infer-implement 和 infer-review 均为 ✓，否则拒绝执行。
5. infer-review 无阻塞问题：将 infer-implement 和 infer-review 均标 ✓；有阻塞问题：仅标 infer-review ✓，infer-implement 保持 ⬜。修复完成后必须再次执行 infer-review，不得跳过 re-review 直接进入 infer-benchmark。
6. infer-review 只输出问题列表，不修改任何文件。
7. infer-archive 完成后：（a）将"当前状态"表中本阶段改为 ✅；（b）**删除**本阶段进度表（不移入 `<details>`）；（c）在下方"下一阶段"行更新为下一 Phase。
8. 发现计划有根本性错误时，停下来修订计划，不得继续实现。
9. 无模型权重时，明确说明，不伪造运行结果。（GPU 始终可用，不需要确认。）
10. Phase 之间的空档期（上一 Phase archive 完成、下一 Phase 尚未 plan）：对话开始时说明"当前在 Phase N 和 Phase N+1 之间，下一步是 Phase N+1 的 infer-plan"，不要误判为 Phase N 仍在进行。

**Phase 9-15 战略原则**：
- 每个 Phase 结束时项目是完整的、可独立展示的，不依赖后续 Phase
- 每个 Phase 有明确的跳过/降级条件，卡点超过合理时间可跳过并文档记录原因
- 整体叙事方向：从调度优化（9-10）→ 算法创新（11）→ 运行时优化（12/12.5）→ 分布式扩展（13）→ 前沿架构（14-15）
- 详细技术规划见 `本地资料/Claude计划/00-长期路线图.md`

**后续阶段最小验收口径**：

| 阶段 | 最小验收标准 |
|------|-------------|
| Phase 13 | 至少实现一层或一条主链路上的真 TP（含 NCCL all-reduce），并与 PP / Replica 明确区分 benchmark 口径 |
| Phase 14 | 跑通 MLA 核心数据流与 cache 组织，解释与标准 MHA/GQA 的差别，给出最小正确性与性能验证 |
| Phase 15 | 拆分 Prefill/Decode 角色或进程，给出解耦后的吞吐、延迟、资源占用变化，说明适用 workload |

## 知识与内容产出

当用户要求阶段总结、里程碑复盘、知识沉淀或博客写作时，默认使用这些目录：

- `本地资料/里程碑总结/`
- `本地资料/知识专题/`
- `本地资料/博客草稿/`
- `本地资料/实验记录/`

博客质量标准见 `infer-blog` skill 及其 `QUALITY_CHECKLIST.md`。

## 开发规则

- 先读相关文件，再动代码
- 优先最小可运行实现，不做无关重构
- 先正确，再测量，再优化
- 没有 benchmark 数据，不要声称性能收益
- 修改项目目标、范围或阶段时，先同步更新 `CLAUDE.md`（权威来源），再更新 `README.md`（简要索引）
- 项目级 `.claude/` 和 `CLAUDE.md` 需要随 Git 同步；如工作区中存在 `.claude/settings.local.json`，仅将其视为机器相关快照，不作为共享项目事实或当前状态来源
- 用户个人研究、实验、环境记录放在 `本地资料/`，不要混进正式项目文档

## 命令与验证

优先使用这些命令：

- AI/代理 shell：`conda run -n ai-infra python ...`
- 交互 shell（已完成 `conda init` 时）：`conda activate ai-infra`
- 环境检查：`pwd`、`conda run -n ai-infra python --version`
- GPU 检查：`nvidia-smi`
- 最小测试：`conda run -n ai-infra python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py`

如果缺少模型权重，必须明确说明，不得伪造运行结果。（GPU 始终可用。）
