# Skills

这个文件用于说明 mini-infer 仓库中 Claude Code skills 的组织方式、用途和使用约定。

## 当前 skills

- `infer-plan`：阶段规划、范围收敛和验收标准拆解
- `infer-implement`：具体实现与最小验证
- `infer-benchmark`：benchmark 设计、执行和结果整理
- `infer-review`：代码审查与风险检查
- `infer-summarize`：阶段总结、里程碑复盘和下一步计划
- `infer-blog`：高质量技术博客草稿与写作清单

## 使用约定

- 需要显式使用时，直接在任务里说明使用哪个 skill
- 优先复用现有 skill，不重复手写一大段临时要求
- skill 的支持文件放在各自目录内，例如模板、指标说明和检查清单
- 当项目规则变化时，先同步更新 `CLAUDE.md` 和对应 skill