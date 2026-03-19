# CLAUDE.md

这个文件定义 Claude Code 在 mini-infer 仓库中的长期项目级规则、边界和工作方式。

## 项目定位

这是一个面向 `Qwen2.5` 类 `decoder-only` 模型的推理系统学习项目，实现并验证了：

- `Paged KV Cache`（BlockTable + FreeBlockPool + block tensor 池）
- `Prefill / Decode` 分离
- `Continuous Batching`（动态准入调度）
- 向量化 KV Gather（PyTorch advanced indexing）
- 双卡扩展（Replica 数据并行 + HF Pipeline Parallel 测量）
- 真实 benchmark（对照 HF Transformers baseline）

**注意**：本项目实现的是 Paged KV Cache + DynamicCache 接口，**不是真正的 PagedAttention**（真正的 PagedAttention 需要 flash_attn 2.5+ 的 block_tables，让 attention kernel 直接从 block tensor 寻址，不需要 gather 到 dense tensor）。

## 当前状态

5 个阶段已全部完成（2026-03-19）：

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 单卡最小推理链路 | ✅ |
| Phase 2 | Paged KV Cache + Batch Decode + Continuous Batching | ✅ |
| Phase 3 | 向量化 gather_batch_kv + DynamicCache（batch=8 达到 HF 的 88.4%）| ✅ |
| Phase 4 | 双卡扩展（Replica + HF PP）| ✅ |
| Phase 5 | Profiling + 项目收尾 | ✅ |

如果继续开发，下一步方向参考 `本地资料/Claude计划/00-长期路线图.md`。

## 环境事实

- 当前默认工作前提：已经位于 Ubuntu 24.04 项目目录内
- 当前项目默认直接复用 `本地资料/环境配置/AI-Infra学习之旅-服务器环境配置.md` 中已经配置好的 `ai-infra` 环境
- 主开发环境是 Ubuntu 24.04 + bash + Python 3.10+
- 真实运行和 benchmark 环境是 Ubuntu 24.04 + 2 × RTX 4090
- 真实模型推理、多卡实验和性能数据默认来自 Ubuntu 实机环境，而不是 Windows 侧远程会话

## 工作方式

- 大任务先探索，再计划，再编码
- 先给可验证的最小闭环，不做大而空抽象
- 每次实现都优先给出验证方式
- 如果无法运行验证，明确说明原因和缺失条件
- 上下文过长或任务切换明显时，建议用户使用 `/clear`
- 阶段性工作完成后，应产出总结、知识沉淀或博客草稿，而不是只停留在代码层

## 文件头规则

- 每个新建或修改的文件，都必须在开头说明当前文件的作用和功能
- 对支持注释或文档字符串的格式，直接使用文件头注释或模块文档字符串
- 对 Markdown 文件，标题下第一段必须说明用途
- 对 JSON 这类不支持注释的格式，使用首个可读字段如 `$comment` 说明用途

## 代码范围

核心代码位于：

- `mini_infer/config.py` — EngineConfig 数据类
- `mini_infer/request.py` — Request / RequestState / SamplingParams
- `mini_infer/scheduler.py` — 请求调度器（waiting/running 队列）
- `mini_infer/kv_cache.py` — Paged KV Cache（BlockTable + FreeBlockPool）
- `mini_infer/model_runner.py` — ModelRunner（prefill + batch decode，含 profiler 标签）
- `mini_infer/engine.py` — LLMEngine（continuous batching 主循环）
- `mini_infer/replica_engine.py` — ReplicaEngine（双卡数据并行）
- `mini_infer/tp_engine.py` — TPEngine（HF Pipeline Parallel，测量用）

测试位于 `tests/`。
benchmark 位于 `benchmarks/`（benchmark_hf.py / benchmark_mini.py / benchmark_multi_gpu.py / profile_decode.py）。
skills 位于 `.claude/skills/`。
规则位于 `.claude/rules/`。
长期计划与个人记录位于 `本地资料/`。

## Skills 使用方式

- 优先使用 `.claude/skills/` 中的长期工作流能力，而不是临时重复提示
- 需要显式调用时，在任务里直接点名对应 skill，例如 `infer-plan`、`infer-implement`、`infer-benchmark`
- 对于阶段规划、阶段总结、知识沉淀和博客写作，默认优先复用已有 skill
- 当某个 skill 变得复杂时，应继续在对应目录中补充模板、清单和辅助说明文件，而不是把所有要求堆回一个文档

## 阶段工作流

每个开发阶段必须按以下顺序至少执行一次每个 skill，不得跳过或乱序：

```
1. infer-plan       → 规划目标、范围、验收标准
2. infer-implement  → 实现代码，最小验证（可多轮）
3. infer-review     → 审查代码，发现问题回到 implement
4. infer-benchmark  → 跑真实数据，记录结果
5. infer-summarize  → 阶段总结，落盘里程碑文档
6. infer-blog       → 博客草稿（可攒多阶段后集中写）
7. infer-archive    → 核查并补全本地资料所有子目录，阶段正式收尾
```

**Claude 的行为要求：**
- 每次对话开始时，如果用户在推进某个阶段，主动说明当前阶段处于哪一步
- 用户完成某一步后，主动提示下一步应该执行哪个 skill
- 如果用户跳步（如跳过 review 直接 benchmark），必须提醒缺失了哪一步
- 不得在 infer-plan 完成前开始 infer-implement，不得在 infer-implement 完成前开始 infer-benchmark
- `infer-archive` 是阶段收尾门控，只有 archive 完成后才能进入下一阶段的 infer-plan
- **一次只执行一个 skill**：用户调用某个 skill 时，完成该 skill 后停止，不得自动串联下一步
- **infer-review 只列问题，不改代码**：review 阶段的任何问题，留给用户调用 infer-implement 处理

**Phase 1 已完成（2026-03-19）**：所有 7 步 ✓，串行推理链路跑通，benchmark 与 HF baseline 对比完成。

**Phase 2 已完成（2026-03-19）**：所有 7 步 ✓，Paged KV Cache + Batch Decode + Continuous Batching 跑通，benchmark 与 HF baseline 对比完成。

**Phase 3 已完成（2026-03-19）**：所有 7 步 ✓，向量化 gather_batch_kv + DynamicCache，batch=8 throughput 从 49.1% → 88.4% HF baseline。

**Phase 4 已完成（2026-03-19）**：所有 7 步 ✓，Replica + HF PP 双卡扩展，Replica batch=8 +4.1%，PP 显存减半。

**当前阶段进度（Phase 5）：**

| 步骤 | skill | 状态 |
|------|-------|------|
| 1 | infer-plan | ✓ 完成 |
| 2 | infer-implement | ✓ 完成 |
| 3 | infer-review | ✓ 完成 |
| 4 | infer-benchmark | ✓ 完成 |
| 5 | infer-summarize | ✓ 完成 |
| 6 | infer-blog | ✓ 完成 |
| 7 | infer-archive | ✓ 完成 |

## 知识与内容产出

当用户要求阶段总结、里程碑复盘、知识沉淀或博客写作时，默认使用这些目录：

- `本地资料/里程碑总结/`
- `本地资料/知识专题/`
- `本地资料/博客草稿/`
- `本地资料/实验记录/`

技术博客必须满足：

- 有明确问题背景，而不是空泛概述
- 有真实设计取舍，而不是纯概念堆砌
- 有实验、代码或运行证据支撑
- 有失败点、坑点和反思
- 语言专业、克制、可发表

## 开发规则

- 先读相关文件，再动代码
- 优先最小可运行实现，不做无关重构
- 先正确，再测量，再优化
- 没有 benchmark 数据，不要声称性能收益
- 修改项目目标、范围或阶段时，先同步更新 `README.md`
- 项目级 `.claude/` 和 `CLAUDE.md` 需要随 Git 同步，`.claude/settings.local.json` 保持本地
- 用户个人研究、实验、环境记录放在 `本地资料/`，不要混进正式项目文档

## 命令与验证

优先使用这些命令：

- 环境激活：`conda activate ai-infra`
- 环境检查：`pwd`、`python3 --version`
- GPU 检查：`nvidia-smi`
- 最小测试：`python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py`

如果当前环境没有 Python、GPU、模型权重或 CUDA 条件，必须明确说明，不得伪造运行结果。
