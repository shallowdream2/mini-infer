# 工作流说明

这个文件说明 mini-infer 项目的阶段工作流和推进顺序。**行为规则见 `CLAUDE.md` 的"行为规则"节**，此文件为流程细节参考。

- 大任务默认采用三段式：探索、计划、实现
- 如果需求跨多个模块，先给计划，再改代码
- 任务切换明显时，建议使用 `/clear`
- 先解决主链路，不要顺手扩张范围
- 每次实现完成后，先做最小验证，再汇报结果
- 如果发现项目目标与当前代码不一致，先更新 `README.md` 再继续

## 阶段 Skill 执行顺序

每个开发阶段依次完成以下 7 个步骤：

1. `infer-plan` — 确定目标、范围和验收标准；**不得在此之前讨论具体实现细节（技术路线可以讨论）**
2. `infer-implement` — 最小可运行实现，每轮都要有验证结果；可多轮迭代，直到 review 无阻塞性问题
3. `infer-review` — 只列问题，不改代码；发现阻塞性问题必须回 implement 修复后再继续
4. `infer-benchmark` — 在真实 GPU 环境跑数据，不得用估算代替
5. `infer-summarize` — 总结落盘，写入 `本地资料/里程碑总结/`；明确对照阶段初的验收标准
6. `infer-blog` — 博客草稿，写入 `本地资料/博客草稿/`（可攒多阶段后集中写）
7. `infer-archive` — 核查并补全 `本地资料/` 所有子目录，确保每个文件夹都有本阶段产出

`infer-archive` 是阶段的收尾门控：只有 archive 完成后，本阶段才算真正结束，才可以进入下一阶段。

## implement → review 的迭代

implement 和 review 是一对迭代循环，不是线性一次性步骤：

```
infer-implement → infer-review → (有问题) → infer-implement → infer-review → (无阻塞) → infer-benchmark
```

每轮 implement 修复 review 发现的问题；每轮 review 验证修复是否正确。轮次不限，直到 review 报告中无阻塞性问题为止。

## 阶段规模控制

- 单个阶段目标不超过 2 个主要技术变更（flash_attn 集成 + KV cache 格式变更算一个；flash_attn + preemption 算两个，应拆分）
- 如果规划超出这个范围，拆成两个阶段
- 每阶段产出应有独立的 benchmark 数据支撑，不能只依赖上一阶段的数字
