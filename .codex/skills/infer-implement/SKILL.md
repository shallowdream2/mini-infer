---
name: infer-implement
description: 在 mini-infer 中实现具体功能，并按项目规则完成最小验证。Use this in Codex when you want the same implementation workflow and validation gate defined for Claude.
---

# infer-implement

在 `mini-infer` 仓库里，`infer-implement` 与 Claude 侧同名 skill 保持一致。

## 必读

1. `CLAUDE.md`
2. `.claude/rules/workflow.md`
3. `.claude/skills/infer-implement/SKILL.md`

## 执行原则

- 按 Claude 版本的计划依赖、前置条件验证、最小实现范围和验证要求执行
- 保留 Claude 版本中对 `dry_run`、GPU 路径和多轮 implement/review 循环的约束
- 如果 Codex 侧旧文档与 Claude 侧规则冲突，以 Claude 文件为准
