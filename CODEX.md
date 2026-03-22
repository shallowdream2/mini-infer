# CODEX.md

这个文件说明 Codex 在 mini-infer 仓库中的项目级工作方式。

与 `README.md` 一起阅读；repo-local Codex skills 源文件位于 `.codex/skills/`。

## 权威来源

针对当前项目，Claude 侧文件是工作流、阶段状态和清单规则的权威来源：

- `CLAUDE.md`
- `.claude/rules/`
- `.claude/skills/`

Codex 侧 `infer-*` skills 与 `mini-infer-*` skills 是适配层，不再单独定义一套不同的阶段口径。

如果 `CODEX.md`、`.codex/skills/` 与 Claude 侧文件冲突，以 Claude 侧文件为准。

## 适用范围

- 仅适用于当前 `mini-infer` 仓库
- 适用对象包括：代码实现、代码审查、benchmark、阶段总结、知识沉淀、博客草稿
- 不包含 `~/.codex` 用户级设置；本机安装步骤见 `本地资料/环境配置/Codex 使用与迁移说明.md`
- 对于当前阶段、进度表、Step 0-7 门控和完成后动作，直接遵守 `CLAUDE.md` 的同名章节

## 先读哪些文件

- `README.md`：项目目标、架构、命令、当前状态
- `本地资料/Claude计划/00-长期路线图.md`：阶段路线图、未来阶段、验收口径
- `本地资料/环境配置/服务器开发与运行流程.md`：日常开发方式
- `本地资料/环境配置/phase1-env-notes.md`、`phase4-env-notes.md`、`phase6-env-notes.md`、`phase8-env-notes.md`：真实环境坑点与版本事实
- `本地资料/实验记录/`：真实 benchmark 和 profiling 数据
- `本地资料/里程碑总结/`：阶段复盘、问题和下一步

带日期的开发日志、里程碑总结和实验记录默认按历史快照处理；判断当前状态和未来计划时，优先以 `README.md`、`CLAUDE.md` 和 `本地资料/Claude计划/00-长期路线图.md` 为准。

## 项目事实

mini-infer 是一个面向 `Qwen2.5` 类 `decoder-only` 模型的推理系统学习项目，已经实现并验证了：

- `Paged KV Cache`
- `Prefill / Decode` 分离
- `Continuous Batching`
- 向量化 KV gather
- `True PagedAttention`
- Triton decode attention kernel
- `Preemption + Priority Scheduling`
- 双卡扩展（Replica + HF Pipeline Parallel 测量）
- `OpenAI Chat Completions 子集兼容 HTTP API`
- `Chunked Prefill`

项目代码主入口：

- `mini_infer/config.py`
- `mini_infer/request.py`
- `mini_infer/scheduler.py`
- `mini_infer/kv_cache.py`
- `mini_infer/attention.py`
- `mini_infer/model_runner.py`
- `mini_infer/engine.py`
- `mini_infer/async_engine.py`
- `mini_infer/server.py`
- `mini_infer/openai_schema.py`
- `mini_infer/replica_engine.py`
- `mini_infer/pp_engine.py`
- `mini_infer/tp_engine.py`
- `benchmarks/benchmark_chunked_prefill.py`
- `tests/test_chunked_prefill.py`

测试位于 `tests/`，benchmark 位于 `benchmarks/`。

## 环境默认值

- 默认工作环境：Ubuntu 24.04 + bash + Python 3.10+
- 默认复用现有 Conda 环境：`ai-infra`
- 真实 benchmark / 多卡实验环境：Ubuntu 24.04 + 2 x RTX 4090
- 默认假设 GPU 可用
- 如果缺少模型权重，必须明确说明缺的是哪个模型，不得伪造运行结果

已验证的关键版本快照（2026-03-21）：

- `torch 2.1.2+cu121`
- `transformers 4.43.4`
- `flash_attn 2.5.9.post1`

默认命令口径跟随 `CLAUDE.md`、`README.md` 和环境文档。

当前已验证：在本机 agent shell 中，`conda activate ai-infra` 会因为缺少 `conda init` 而失败；因此 Codex 默认使用 `conda run -n ai-infra ...`，只有在用户交互 shell 已验证可用时才使用 `conda activate ai-infra`。

模型加载默认约定：

- `MODEL=/home/shh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`
- `HF_HUB_OFFLINE=1`

优先使用本地模型根目录路径，不使用 HF Hub ID，避免 snapshot 检查触发重下载。

## 工作方式

- 先读相关代码和文档，再动代码
- 先给最小可验证闭环，不顺手做无关重构
- 先正确，再测量，再优化
- 修改后优先跑最小相关测试；无法验证时，明确说明边界
- 没有真实 benchmark 数据时，不要宣称性能提升
- 任务切换明显或上下文过长时，建议用户主动开启新会话

## 阶段工作流

Codex 对当前项目默认使用与 Claude 相同的 7 步阶段工作流：

1. `infer-plan`
2. `infer-implement`
3. `infer-review`
4. `infer-benchmark`
5. `infer-summarize`
6. `infer-blog`
7. `infer-archive`

兼容保留的 Codex 别名：

- `mini-infer-plan`
- `mini-infer-implement`
- `mini-infer-review`
- `mini-infer-benchmark`
- `mini-infer-summarize`
- `mini-infer-blog`
- `mini-infer-archive`
- `mini-infer-repo`

如果只是进入仓库做通用分析、实现或排查，而不是明确阶段工作，可先使用 `mini-infer-repo` 作为入口 skill，再路由到同名 `infer-*` 流程 skill。

关键门控、当前阶段进度、Step 0 规则和 archive 后的状态更新，都按 `CLAUDE.md` 原样执行。

## 验证与表达规则

- 修改 Python 代码后，先跑最小相关测试，不默认全量 `pytest`
- GPU 路径修改，至少说明真实验证方式；如受模型权重限制，明确写出
- profiling / benchmark 报告至少说明：环境、模型、batch size、prompt/output 长度、并发数
- 核心指标默认包括：`throughput`、`TTFT`、`TPOT`、`peak memory`
- 如果某个 benchmark 指标当前不可得、只近似可得或与其他脚本口径不同，必须显式写明 `N/A`、`近似` 或差异说明
- benchmark 异常时先诊断时间分布和根因，不继续堆结论

## 文档和产出位置

阶段性内容默认写入这些目录：

- `本地资料/实验记录/`
- `本地资料/里程碑总结/`
- `本地资料/知识笔记/`
- `本地资料/知识专题/`
- `本地资料/博客草稿/`
- `本地资料/Claude计划/`
- `本地资料/环境配置/`

如果项目目标、架构、命令或当前状态变化，先同步更新 `README.md`。

## Repo-local Codex 资产

仓库内已经整理出一套面向 Codex 的本地资产：

- `CODEX.md`：项目级指引
- `.codex/skills/infer-*`：与 Claude 同名、同口径的 Codex 入口 skills
- `.codex/skills/mini-infer-*`：兼容别名与仓库入口 skill
- `本地资料/环境配置/Codex 使用与迁移说明.md`：本机安装和本地设置说明

这些文件是共享版本控制的一部分；真正的用户级安装动作仍然需要在本机 `~/.codex/` 下手动完成。
