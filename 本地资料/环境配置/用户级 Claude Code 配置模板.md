# 用户级 Claude Code 配置模板

这个文件用于给当前 Ubuntu 开发环境提供用户级 Claude Code 配置模板，并说明哪些内容应该留在用户级、哪些内容应该放回项目级。

## 用户级设置文件

路径：`~/.claude/settings.json`

推荐模板：

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "effortLevel": "high",
  "cleanupPeriodDays": 90,
  "autoMemoryDirectory": "~/ClaudeMemory",
  "permissions": {
    "deny": [
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)",
      "Read(~/.env)",
      "Read(~/.env.*)"
    ]
  }
}
```

说明：

- `effortLevel`：复杂项目建议用 `high`
- `cleanupPeriodDays`：长期项目建议拉长会话保留周期
- `autoMemoryDirectory`：把长期记忆放到固定目录，便于管理和备份
- 这里不再默认写 `CLAUDE_CODE_GIT_BASH_PATH`，因为当前假设已经在 Ubuntu 环境内

### 仅在 Windows 下才需要的可选项

如果你以后回到 Windows，并且 Claude 找不到 Git Bash，再补这段：

```json
{
  "env": {
    "CLAUDE_CODE_GIT_BASH_PATH": "C:\\Program Files\\Git\\bin\\bash.exe"
  }
}
```

## 用户级 CLAUDE.md

路径：`~/.claude/CLAUDE.md`

推荐模板：

```md
# 个人开发偏好

- 默认使用中文交流，代码除外
- 先读代码和文档，再开始实现
- 大任务先计划，后实现
- 优先最小可运行方案
- 修改完成后优先做最小验证
- 如果无法验证，必须明确说明原因
```

## 用户级 rules

路径：`~/.claude/rules/`

适合放这里的内容：

- 你所有项目通用的 code review 风格
- 你所有项目通用的输出格式
- 你所有项目通用的安全约束

## 不要放在用户级的内容

- mini-infer 的目标与范围
- 本项目的模型选择
- 本项目 benchmark 口径
- 本项目阶段计划
