# Repo-local Codex Assets

这个目录保存 mini-infer 的 Codex 迁移资产，但不会自动安装到 `~/.codex/`。

当前策略已经切换为：**Codex 侧按照 Claude 侧工作流执行**。

## 当前内容

- `../CODEX.md`：项目级 Codex 指引
- `skills/mini-infer-plan`
- `skills/mini-infer-implement`
- `skills/mini-infer-review`
- `skills/mini-infer-benchmark`
- `skills/mini-infer-summarize`
- `skills/mini-infer-blog`
- `skills/mini-infer-archive`
- `skills/mini-infer-repo`
- `skills/infer-plan`
- `skills/infer-implement`
- `skills/infer-review`
- `skills/infer-benchmark`
- `skills/infer-summarize`
- `skills/infer-blog`
- `skills/infer-archive`

## Claude -> Codex 映射

| Claude 资产 | Codex 资产 | 说明 |
|------|------|------|
| `CLAUDE.md` | `CODEX.md` | Codex 项目说明，但冲突时以 Claude 侧为准 |
| `.claude/skills/infer-plan` | `.codex/skills/infer-plan` | 同名同口径入口 |
| `.claude/skills/infer-implement` | `.codex/skills/infer-implement` | 同名同口径入口 |
| `.claude/skills/infer-review` | `.codex/skills/infer-review` | 同名同口径入口 |
| `.claude/skills/infer-benchmark` | `.codex/skills/infer-benchmark` | 同名同口径入口 |
| `.claude/skills/infer-summarize` | `.codex/skills/infer-summarize` | 同名同口径入口 |
| `.claude/skills/infer-blog` | `.codex/skills/infer-blog` | 同名同口径入口 |
| `.claude/skills/infer-archive` | `.codex/skills/infer-archive` | 同名同口径入口 |
| Codex 兼容别名 | `.codex/skills/mini-infer-*` | 历史兼容名，内部也应遵守 Claude 侧规则 |
| `无直接等价物` | `.codex/skills/mini-infer-repo` | Codex 仓库入口 skill，仅用于加载上下文并路由 |

## 手动安装到 `~/.codex/skills`

如果要让 Codex 真正发现这些 skills，需要在本机执行复制或软链接。

推荐方式是软链接，便于仓库内更新后自动生效：

```bash
mkdir -p ~/.codex/skills

ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-plan ~/.codex/skills/mini-infer-plan
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-implement ~/.codex/skills/mini-infer-implement
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-review ~/.codex/skills/mini-infer-review
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-benchmark ~/.codex/skills/mini-infer-benchmark
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-summarize ~/.codex/skills/mini-infer-summarize
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-blog ~/.codex/skills/mini-infer-blog
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-archive ~/.codex/skills/mini-infer-archive
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-repo ~/.codex/skills/mini-infer-repo
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-plan ~/.codex/skills/infer-plan
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-implement ~/.codex/skills/infer-implement
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-review ~/.codex/skills/infer-review
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-benchmark ~/.codex/skills/infer-benchmark
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-summarize ~/.codex/skills/infer-summarize
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-blog ~/.codex/skills/infer-blog
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-archive ~/.codex/skills/infer-archive
```

如果不想用软链接，也可以手动复制目录。

## 建议保留在本机、不进 Git 的内容

- `~/.codex/config.toml` 里的个人模型偏好和 trust 配置
- 模型绝对路径、代理设置、私有别名
- 任何账号、token、密钥或个人记忆目录

## 不做 1:1 迁移的 Claude 专有字段

这些内容不建议直接机械搬运到 Codex：

- `.claude/settings.json` 的 `permissions.allow / deny`
- `.claude/settings.local.json` 的 `remote.defaultEnvironmentId`
- `.claude/settings.local.json` 的 `autoMemoryDirectory`

这些字段只能按语义重写，不能假定 Codex 存在完全等价字段。

## 当前本机推荐约定

- 优先用 `conda run -n ai-infra ...`
- 模型路径优先用本地绝对目录：`/home/shh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct`
- 默认配合 `HF_HUB_OFFLINE=1`
- 在 repo 内工作时，优先使用与 Claude 同名的 `infer-*` skills
- 如果只是进入仓库做通用分析或路由，可先用 `mini-infer-repo`
