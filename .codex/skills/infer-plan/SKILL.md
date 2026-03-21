---
name: infer-plan
description: 为 mini-infer 规划阶段任务、范围和验收标准。Use this in Codex when you want the same planning workflow and output contract defined for Claude in this repository.
---

# infer-plan

在 `mini-infer` 仓库里，`infer-plan` 的项目权威来源是 Claude 侧文件，而不是 Codex 的独立改写版。

## 必读

1. `CLAUDE.md`
2. `.claude/rules/workflow.md`
3. `.claude/skills/infer-plan/SKILL.md`
4. `.claude/skills/infer-plan/CHECKLIST.md`

## 执行原则

- 按 Claude 版本的输出结构、门控规则、自查清单和完成后动作原样执行
- 如果 `CODEX.md`、旧版 `.codex/skills/mini-infer-plan/` 或其他文档与以上 Claude 文件冲突，以 Claude 文件为准
- 不要简化验收标准、风险表、前置条件验证命令或自查输出
