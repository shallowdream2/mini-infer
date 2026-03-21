---
name: infer-benchmark
description: 为 mini-infer 设计或执行 benchmark，并输出统一口径的结果说明。Use this in Codex when you want the same benchmark gates, metrics, and reporting rules defined for Claude.
---

# infer-benchmark

在 `mini-infer` 仓库里，`infer-benchmark` 以 Claude 侧 benchmark 规则为权威来源。

## 必读

1. `CLAUDE.md`
2. `.claude/skills/infer-benchmark/SKILL.md`
3. `.claude/skills/infer-benchmark/METRICS_SPEC.md`

## 执行原则

- 保留 Claude 版本的硬门控，不把 benchmark 降级成“最好在 review 后执行”
- 保留专项 profiling、attention 替换验证、异常结果处理规程和落盘要求
- 如果 Codex 侧旧版 benchmark 说明更宽松，以 Claude 文件为准
