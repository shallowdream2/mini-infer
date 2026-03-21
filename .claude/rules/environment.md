# 环境规则

这个文件说明本项目当前默认的 Ubuntu 工作环境、运行约束和包变更规范。

## 运行环境

- 当前项目默认假设：已经位于 Ubuntu 24.04 项目目录内
- 当前项目默认直接复用已配置完成的 `ai-infra` Conda 环境，不默认建议新建环境
- 当前项目的主开发环境是 Ubuntu 24.04 + Python 3.10+
- 当前项目的真实运行环境是 Ubuntu 24.04 + 2 × RTX 4090
- 真实模型推理、GPU 测试、性能 benchmark 和多卡实验默认在 Ubuntu 实机环境完成
- 非 Ubuntu 环境不用于给出最终性能结论
- **GPU 始终可用**：用户只在 GPU 服务器上工作，不要询问 GPU 是否可用，直接假设有
- 如果当前会话缺少**模型权重**，明确说明缺少的是哪个模型，不得伪造运行结果
- 需要确认环境是否可用时，优先在当前环境执行最小命令或临时测试文件，而不是先建议重建环境
- AI/代理的非交互 shell 可能没有完成 `conda init`；默认优先使用 `conda run -n ai-infra ...`，只有在交互 shell 已验证可用时才使用 `conda activate ai-infra`

## 包版本变更规范

需要修改 Python 包版本或安装新包时，防止升级破坏现有阶段的可复现性。

### 升级前必做

1. **记录当前版本**：`pip show <package> | head -3`
2. **评估影响范围**：哪些已有测试依赖此包？已有 benchmark 结果是否会因升级而无法复现？
3. **制定回退方案**：明确回退命令（`pip install package==旧版本`）

### 升级编译型库（如 flash_attn）

```bash
# 优先尝试（--no-build-isolation 使用已有 PyTorch，避免重新编译）
pip install "flash-attn==X.Y.Z" --no-build-isolation
```

升级后必须验证：
1. 目标 API/参数存在（`python -c "import inspect; ..."`)
2. 现有测试仍全部通过（`python -m pytest tests/ -v`）
3. 基本 import 无异常（`python -c "from mini_infer import LLMEngine, EngineConfig; print('ok')"`)

### 添加新 optional 依赖

如果某个包只在特定阶段才需要，加入 `[project.optional-dependencies]` 而不是 `dependencies`：

```toml
[project.optional-dependencies]
flash = ["flash-attn>=2.5.0"]   # Phase 6+
serve = ["fastapi>=0.100", "uvicorn"]  # Phase 8+
```

### 变更后的文档同步

升级成功后，必须同步更新：
1. `CLAUDE.md` 的"当前已安装关键版本"表格（**同时更新括号中的日期**）
2. `本地资料/环境配置/` 下记录变更日志（日期 + 变更内容 + 验证结果）
