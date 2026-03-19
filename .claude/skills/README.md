# Skills

这个文件用于说明 mini-infer 仓库中 Claude Code skills 的组织方式、用途和使用约定。

## 当前 skills

- `infer-plan`：阶段规划、范围收敛和验收标准拆解
- `infer-implement`：具体实现与最小验证
- `infer-benchmark`：benchmark 设计、执行和结果整理
- `infer-review`：代码审查与风险检查
- `infer-summarize`：阶段总结、里程碑复盘和下一步计划
- `infer-blog`：高质量技术博客草稿与写作清单
- `infer-archive`：阶段收尾，核查并补全本地资料所有子目录

## 每阶段必须执行的顺序

每个开发阶段按以下顺序执行，每步至少一次，不得跳过：

```
1. infer-plan        规划目标、范围、验收标准
         ↓
2. infer-implement   实现代码 + 最小验证（可多轮）
         ↓
3. infer-review      代码审查，有问题回 implement
         ↓
4. infer-benchmark   真实 GPU 数据，记录结果
         ↓
5. infer-summarize   阶段总结，落盘里程碑文档
         ↓
6. infer-blog        博客草稿（可攒多阶段集中写）
         ↓
7. infer-archive     核查补全本地资料，阶段正式收尾
```

## 各阶段进度追踪

| 阶段 | plan | implement | review | benchmark | summarize | blog | archive |
|------|------|-----------|--------|-----------|-----------|------|---------|
| Phase 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Phase 2 | ✓ | — | — | — | — | — | — |
| Phase 3 | — | — | — | — | — | — | — |

## 使用约定

- 需要显式使用时，直接在任务里说明使用哪个 skill
- 优先复用现有 skill，不重复手写一大段临时要求
- skill 的支持文件放在各自目录内，例如模板、指标说明和检查清单
- 当项目规则变化时，先同步更新 `CLAUDE.md` 和对应 skill