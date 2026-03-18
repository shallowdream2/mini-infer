# CLAUDE.md

这个文件定义 Claude Code 在 mini-infer 仓库中的长期项目级规则、边界和工作方式。

## 项目定位

这是一个面向 `Qwen2.5` 类 `decoder-only` 模型的推理系统项目，最终目标是实现：

- `Paged KV Cache`
- `Block Table`
- `Prefill / Decode` 分离
- `Continuous Batching`
- 真实 benchmark

## 当前状态

当前仓库仍处于初级骨架阶段。

- 核心代码大多是桩实现
- benchmark 仍是骨架
- 测试只覆盖最小 smoke 逻辑
- 真实模型推理、真实 KV cache 和真实调度器尚未开始正式实现

## 当前优先级

1. 单卡真实模型推理链路
2. `HuggingFace` baseline benchmark
3. `Paged KV Cache`
4. `Continuous Batching`
5. 双卡扩展

未被明确要求时，不要提前进入后续阶段。

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

- `mini_infer/config.py`
- `mini_infer/request.py`
- `mini_infer/scheduler.py`
- `mini_infer/kv_cache.py`
- `mini_infer/model_runner.py`
- `mini_infer/engine.py`

测试位于 `tests/`。
benchmark 位于 `benchmarks/`。
skills 位于 `.claude/skills/`。
规则位于 `.claude/rules/`。
长期计划与个人记录位于 `本地资料/`。

## Skills 使用方式

- 优先使用 `.claude/skills/` 中的长期工作流能力，而不是临时重复提示
- 需要显式调用时，在任务里直接点名对应 skill，例如 `infer-plan`、`infer-implement`、`infer-benchmark`
- 对于阶段规划、阶段总结、知识沉淀和博客写作，默认优先复用已有 skill
- 当某个 skill 变得复杂时，应继续在对应目录中补充模板、清单和辅助说明文件，而不是把所有要求堆回一个文档

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
