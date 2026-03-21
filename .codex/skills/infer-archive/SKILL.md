---
name: infer-archive
description: 阶段结束时检查并补全本地资料文件夹，确保每个子目录都有本阶段产出。Use this in Codex when you want the same phase-closeout workflow defined for Claude.
---

# infer-archive

在 `mini-infer` 仓库里，`infer-archive` 以 Claude 侧归档流程为权威来源。

## 必读

1. `CLAUDE.md`
2. `.claude/skills/infer-archive/SKILL.md`
3. `.claude/skills/infer-archive/CHECKLIST.md`

## 执行原则

- 补缺不重写、检查所有本地资料子目录、同步 README 和阶段状态时，按 Claude 版本执行
- 归档报告和逐项自查也按 Claude 版本执行
- 如果 Codex 侧旧版 archive 说明更简化，以 Claude 文件为准
