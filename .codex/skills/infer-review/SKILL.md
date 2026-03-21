---
name: infer-review
description: 对 mini-infer 的改动做代码审查，优先指出真实问题和风险。Use this in Codex when you want the same findings-first review workflow defined for Claude.
---

# infer-review

在 `mini-infer` 仓库里，`infer-review` 以 Claude 侧同名 skill 和 checklist 为权威来源。

## 必读

1. `CLAUDE.md`
2. `.claude/skills/infer-review/SKILL.md`
3. `.claude/skills/infer-review/CHECKLIST.md`

## 执行原则

- 输出结构、阻塞问题定义、review 与 implement 的分离、以及完成后状态更新逻辑都按 Claude 版本执行
- findings-first，不要降级成泛泛建议
- 如果 Codex 侧旧版 review 说明更宽松，以 Claude 文件为准
