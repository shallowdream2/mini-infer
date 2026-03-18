---
name: infer-review
description: 对 mini-infer 的改动做代码审查，优先指出真实问题和风险。
argument-hint: [审查范围]
---

# infer-review

这个文件用于对当前改动做高质量代码审查。

把附加文本视为审查范围，并遵守这些要求：

- 先列问题，再给总结
- 优先关注行为回归、状态流转错误、KV cache 风险、benchmark 口径错误、缺失测试
- 不要把风格建议放在真正问题前面
- 说明仍未验证的部分
