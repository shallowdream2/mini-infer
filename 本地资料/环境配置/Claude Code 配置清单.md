# Claude Code 配置清单

这个文件用于说明当前仓库中已经准备好的 Claude Code 配置，以及在 Ubuntu 项目环境中应如何组织用户级、项目级和本地项目级配置。

## 当前仓库内已经准备好的内容

- `CLAUDE.md`
- `.claude/settings.json`
- `.claude/settings.local.json`
- `.claude/rules/`
- `.claude/skills/`

## 配置原则

### 三层配置

1. 用户级：`~/.claude/settings.json`、`~/.claude/CLAUDE.md`
2. 项目级：当前仓库的 `CLAUDE.md`、`.claude/settings.json`、`.claude/rules/`、`.claude/skills/`
3. 本地项目级：`.claude/settings.local.json`

### 你应该如何使用这三层

- 用户级：放你所有项目都适用的偏好
- 项目级：放 mini-infer 的目标、规则、skills 和边界
- 本地项目级：放这台机器上的长期记忆目录、个人覆盖项

## 当前默认前提

- 当前项目默认已经位于 Ubuntu 24.04 项目终端内
- 当前项目直接复用 `AI-Infra学习之旅-服务器环境配置.md` 中已经配置完成的 `ai-infra` 环境
- 项目文档和规则不再以 Windows Remote SSH 作为默认工作方式
- 真实 benchmark 与 GPU 结论仍然需要 Ubuntu + CUDA 实机环境

## 长期产出链路

这个项目不只是编码，还要持续产出：

- 阶段总结
- 里程碑复盘
- 深度知识专题
- 高质量技术博客

## Skills 使用方式

- 优先复用 `.claude/skills/` 中的长期工作流能力，而不是每次手写临时提示
- 需要显式使用时，直接在任务里点名对应 skill
- `infer-benchmark`、`infer-summarize`、`infer-blog` 还带有同目录支持文件，可直接复用指标定义、模板和清单

### 规划类

```text
请使用 infer-plan，为第一阶段的真实单卡推理链路做任务拆解。
```

### 实现类

```text
请使用 infer-implement，把 model_runner.py 从桩实现替换成 HuggingFace 单卡推理。
```

### benchmark 类

```text
请使用 infer-benchmark，为 Qwen2.5-7B 设计单卡 baseline benchmark 方案。
```

### review 类

```text
请使用 infer-review，审查最近对 scheduler 和 kv_cache 的修改。
```

### 总结类

```text
请使用 infer-summarize，总结当前第一阶段进度，并生成下一阶段待办。
```

### 博客类

```text
请使用 infer-blog，基于 Paged KV Cache 的设计与实现，写一篇高质量技术博客草稿。
```

## 高质量技术博客的要求

- 必须基于真实项目经历
- 必须有真实设计与取舍
- 必须有真实代码或实验支撑
- 必须写出坑点和反思
- 不接受空泛、模板化、流水账式文章
