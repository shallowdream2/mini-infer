# Phase 6.5 环境配置笔记（2026-03-20）

## 新增依赖

Phase 6.5 使用 Triton，PyTorch 2.1.2 已内置，无需单独安装。

| 包 | 版本 | 状态 |
|----|------|------|
| triton | 2.1.0（PyTorch 内置） | ✅ 已验证 |

## 验证命令

```bash
# 验证 triton 版本
python -c "import triton; print(triton.__version__)"
# 输出：2.1.0

# 验证 GPU JIT 编译可用（必须写到文件，不能用 python -c）
# 见 CLAUDE.md Phase 6.5 前置条件验证命令
```

## 注意事项

**@triton.jit 不能用 python -c 测试**：`inspect.getsource()` 需要实际源文件，内联代码无源文件会报 `OSError`。测试 kernel 必须写到 `.py` 文件运行。

## 无需变更

- PyTorch 版本：2.1.2+cu121（未变更）
- flash_attn 版本：2.5.9.post1（未变更，Phase 6 已验证）
- transformers 版本：4.43.4（未变更）
- Conda 环境：ai-infra（未变更）
