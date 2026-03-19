# 工作流规则

这个文件约束 Claude Code 在当前项目中的默认工作流和推进顺序。

- 大任务默认采用三段式：探索、计划、实现
- 如果需求跨多个模块，先给计划，再改代码
- 任务切换明显时，建议使用 `/clear`
- 先解决主链路，不要顺手扩张范围
- 每次实现完成后，先做最小验证，再汇报结果
- 如果发现项目目标与当前代码不一致，先更新 `README.md` 再继续

## Skill 调用约束（硬性）

- **一次只执行一个 skill**：用户调用某个 skill 时，只执行该 skill 的内容，不得自动串联下一个
- 比如调用 `infer-implement` 后，不得在同一次响应中自动开始 `infer-review` 或 `infer-benchmark`
- 等用户明确调用下一个 skill 再继续，即使上一步已完成

## 阶段 Skill 执行顺序（强制）

每个开发阶段必须依次完成以下 7 个步骤：

1. `infer-plan` — 确定目标、范围和验收标准，不得跳过直接写代码
2. `infer-implement` — 最小可运行实现，每轮都要有验证结果；**可多轮迭代**，直到 review 无阻塞性问题
3. `infer-review` — **只列问题，不改代码**；发现阻塞性问题必须回 implement 修复后再继续
4. `infer-benchmark` — 在真实 GPU 环境跑数据，不得用估算代替
5. `infer-summarize` — 总结落盘，写入 `本地资料/里程碑总结/`；明确对照阶段初的验收标准
6. `infer-blog` — 博客草稿，写入 `本地资料/博客草稿/`（可攒多阶段后集中写）
7. `infer-archive` — 核查并补全 `本地资料/` 所有子目录，确保每个文件夹都有本阶段产出

违反顺序时，必须向用户说明当前缺少了哪一步，并建议补上后再继续。
`infer-archive` 是阶段的收尾门控：只有 archive 完成后，本阶段才算真正结束，才可以进入下一阶段的 `infer-plan`。

## implement → review 的迭代说明

implement 和 review 是一对迭代循环，不是线性一次性步骤：

```
infer-implement → infer-review → (有问题) → infer-implement → infer-review → (无阻塞) → infer-benchmark
```

每轮 implement 修复 review 发现的问题；每轮 review 验证修复是否正确。
轮次不限，直到 review 报告中无"严重/必须修复"问题为止。