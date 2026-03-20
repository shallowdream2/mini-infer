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

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 单卡最小推理链路 | ✅ |
| Phase 2 | Paged KV Cache + Batch Decode + Continuous Batching | ✅ |
| Phase 3 | 向量化 gather_batch_kv + DynamicCache（batch=8 达到 HF 的 88.4%）| ✅ |
| Phase 4 | 双卡扩展（Replica + HF PP）| ✅ |
| Phase 5 | Profiling + 项目收尾 | ✅ |
| Phase 6 | True PagedAttention（flash_attn block_tables，batch=8 达到 100% HF）| ✅ |
| Phase 6.5 | Triton kernel（decode attention kernel，对比 flash_attn，大厂核心路线主线）| ✅ |
| Phase 7 | Preemption + Priority Scheduling（swap to CPU，优先级调度）| 🔄 进行中 |
| Phase 8 | OpenAI-compatible HTTP API（FastAPI + streaming）| 🔜 规划中 |

详细阶段计划参考 `本地资料/Claude计划/00-长期路线图.md`。

## 环境事实

- 当前默认工作前提：已经位于 Ubuntu 24.04 项目目录内
- 当前项目默认直接复用 `本地资料/环境配置/AI-Infra学习之旅-服务器环境配置.md` 中已经配置好的 `ai-infra` 环境
- 主开发环境是 Ubuntu 24.04 + bash + Python 3.10+
- 真实运行和 benchmark 环境是 Ubuntu 24.04 + 2 × RTX 4090

**当前已安装关键版本（2026-03-20）：**

| 包 | 版本 | 备注 |
|----|------|------|
| PyTorch | 2.1.2+cu121 | CUDA 12.1 |
| transformers | 4.43.4 | >= 4.40.0 ✓ |
| flash_attn | **2.5.9.post1** | ✅ block_table 可用（Phase 6 已用） |
| Python | 3.10 | — |

**关键约束：**
- PyTorch SDPA flash_sdp 已启用 → HF 模型推理时已在用 FlashAttention kernel
- flash_attn 2.5.9.post1 的 `flash_attn_with_kvcache` 已有 `block_table` 参数，Phase 6 已使用

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
- `mini_infer/scheduler.py` — 请求调度器（waiting/running 队列）
- `mini_infer/kv_cache.py` — Paged KV Cache（BlockTable + FreeBlockPool）
- `mini_infer/attention.py` — PagedDecodeContext + patch_model_for_paged_decode（Phase 6）
- `mini_infer/model_runner.py` — ModelRunner（prefill + batch decode，含 profiler 标签）
- `mini_infer/engine.py` — LLMEngine（continuous batching 主循环）
- `mini_infer/replica_engine.py` — ReplicaEngine（双卡数据并行）
- `mini_infer/pp_engine.py` — PPEngine（HF Pipeline Parallel，测量用）
- `mini_infer/tp_engine.py` — 向后兼容别名（TPEngine = PPEngine）

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

每个阶段依次执行 7 步：infer-plan → infer-implement → infer-review → infer-benchmark → infer-summarize → infer-blog → infer-archive。步骤细节见 `.claude/rules/workflow.md`。

**行为规则（每次对话必须遵守）**

1. 对话开始时，说明当前步骤并主动引导下一步。
2. 一次只执行一个 skill，完成后等待用户指令。
3. Step 0 ✓ 前不得进入 infer-implement；infer-plan 可在 Step 0 前运行以输出验证命令。
4. infer-benchmark 要求进度表 infer-implement 和 infer-review 均为 ✓，否则拒绝执行。
5. infer-review 无阻塞问题：将 infer-implement 和 infer-review 均标 ✓；有阻塞问题：仅标 infer-review ✓，infer-implement 保持 ⬜。
6. infer-review 只输出问题列表，不修改任何文件。
7. infer-archive 完成后，将 CLAUDE.md "当前状态"表中本阶段改为 ✅。
8. 发现计划有根本性错误时，停下来修订计划，不得继续实现。
9. 无 GPU 或无模型权重时，明确说明，不伪造运行结果。

**当前阶段进度（Phase 7）：**

> 说明：每个 skill 完成后应立即将对应步骤标为 ✓。进度表在 Phase 开始和 archive 完成时精确；中途如果 /clear 了对话，以此表为重建上下文的起点。
>
> **Step 0 标记规则**：用户执行前置条件验证命令后，若反馈"OK"或"通过"，Claude 应立即将 Step 0 标为 ✓，**不需要用户手动修改进度表**。

| 步骤 | skill | 状态 |
|------|-------|------|
| 0 | 前置条件验证（mini_infer 可导入 + 现有测试通过）| ✓ |
| 1 | infer-plan | ✓ |
| 2 | infer-implement | ⬜ |
| 3 | infer-review | ✓ |
| 4 | infer-benchmark | ⬜ |
| 5 | infer-summarize | ⬜ |
| 6 | infer-blog | ⬜ |
| 7 | infer-archive | ⬜ |

Phase 7 前置条件验证命令：
```bash
python -c "from mini_infer import LLMEngine, EngineConfig; print('ok')"
python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py tests/test_engine.py -q
```

<details>
<summary>Phase 6.5 历史进度（已完成 ✓）</summary>

| 步骤 | skill | 状态 |
|------|-------|------|
| 0 | 前置条件验证（triton 可用 + GPU JIT 执行）| ✓ |
| 1 | infer-plan | ✓ |
| 2 | infer-implement | ✓ |
| 3 | infer-review | ✓ |
| 4 | infer-benchmark | ✓ |
| 5 | infer-summarize | ✓ |
| 6 | infer-blog | ✓ |
| 7 | infer-archive | ✓ |

</details>

<details>
<summary>Phase 6 历史进度（已完成 ✓）</summary>

| 步骤 | skill | 状态 |
|------|-------|------|
| 0 | 前置条件验证（flash_attn 升级 + block_table 参数验证）| ✓ |
| 1 | infer-plan | ✓ |
| 2 | infer-implement | ✓ |
| 3 | infer-review | ✓ |
| 4 | infer-benchmark | ✓ |
| 5 | infer-summarize | ✓ |
| 6 | infer-blog | ✓ |
| 7 | infer-archive | ✓ |

</details>

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
