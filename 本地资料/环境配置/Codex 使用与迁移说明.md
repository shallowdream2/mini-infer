# Codex 使用与迁移说明

这个文件用于说明：仓库里已经为 Codex 整理了哪些资产、还剩哪些本机动作需要手动完成，以及 Codex 现在如何按 Claude 工作流运行 mini-infer。

## 仓库内已经完成的部分

当前仓库已经新增这些 Codex 资产：

- `CODEX.md`：项目级 Codex 指引
- `.codex/skills/infer-*`：与 Claude 同名、同口径的 Codex 入口 skills
- `.codex/skills/mini-infer-*`：可安装到 `~/.codex/skills/` 的项目技能源文件
- `.codex/README.md`：技能映射和安装说明

这些文件都位于仓库内，可以和项目内容一起同步。

## 还没有自动完成的本机动作

以下动作仍然需要在你自己的机器上手动做：

1. 把 `.codex/skills/infer-*` 安装到 `~/.codex/skills/`；如需兼容旧入口，再额外安装 `.codex/skills/mini-infer-*`
2. 如有需要，手动调整 `~/.codex/config.toml` 的模型偏好、默认推理强度和 trust 配置
3. 根据本机环境设置模型路径、代理或 shell alias

原因：

- 这些路径位于仓库外，不能假定每台机器都一样
- Claude 的 `settings.json` / `settings.local.json` 不是 Codex 的等价配置格式，不能机械迁移

## 推荐安装方式

推荐用软链接，把仓库里的 skill 源目录映射到 `~/.codex/skills/`：

```bash
mkdir -p ~/.codex/skills

ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-plan ~/.codex/skills/infer-plan
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-implement ~/.codex/skills/infer-implement
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-review ~/.codex/skills/infer-review
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-benchmark ~/.codex/skills/infer-benchmark
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-summarize ~/.codex/skills/infer-summarize
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-blog ~/.codex/skills/infer-blog
ln -sfn /home/shh/projects/mini-infer/.codex/skills/infer-archive ~/.codex/skills/infer-archive

ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-plan ~/.codex/skills/mini-infer-plan
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-implement ~/.codex/skills/mini-infer-implement
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-review ~/.codex/skills/mini-infer-review
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-benchmark ~/.codex/skills/mini-infer-benchmark
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-summarize ~/.codex/skills/mini-infer-summarize
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-blog ~/.codex/skills/mini-infer-blog
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-archive ~/.codex/skills/mini-infer-archive
ln -sfn /home/shh/projects/mini-infer/.codex/skills/mini-infer-repo ~/.codex/skills/mini-infer-repo
```

这样仓库里的更新可以直接反映到 `~/.codex/skills/`。

## 当前这台机器的推荐本地约定

### 1. Python / Conda

优先使用：

```bash
conda run -n ai-infra python ...
```

原因：在代理 shell 里，`conda activate ai-infra` 不一定稳定。

### 2. 模型路径

优先使用本地绝对路径，而不是 HF Hub ID：

```bash
export MODEL=/home/shh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1
```

原因：当前 HF cache 的 snapshot 目录不完整，直接传 Hub ID 可能触发重下载。

### 3. GPU / benchmark

- 默认假设 GPU 可用
- 真实 benchmark、多卡实验、性能结论只基于 Ubuntu + CUDA 实测
- 如果缺少模型权重，要明确说明，不要伪造运行结果

## 不建议直接照搬的 Claude 配置

这些内容不要做 1:1 搬运：

- `.claude/settings.json` 的 `permissions.allow / deny`
- `.claude/settings.local.json` 的 `remote.defaultEnvironmentId`
- `.claude/settings.local.json` 的 `autoMemoryDirectory`

更稳妥的做法是：

- 把项目工作流的权威定义继续留在 `CLAUDE.md`、`.claude/rules/` 和 `.claude/skills/`
- 让 Codex 通过 `.codex/skills/infer-*` 去执行同一套流程
- 把模型路径、代理和个人偏好留在 `~/.codex/` 或 shell 本地配置
