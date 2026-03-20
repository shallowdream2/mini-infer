# infer-review 检查清单

这个文件列出代码审查时的标准检查项，用于确保每次 review 覆盖关键风险点。

## 必须检查

### 正确性
- [ ] 新增 API 调用是否使用了正确的字段名 / 参数名（对照数据类定义核实）
- [ ] 新增函数是否可能在 dry_run=True 路径下被意外调用
- [ ] KV cache 操作：分配的 block 是否在所有路径（包括异常）下都能被释放
- [ ] 多卡操作：device 参数是否正确传递，是否存在隐式 `.cuda()` 或 `.cpu()` 跨设备

### 性能关键路径（必查，标为阻塞性问题）
- [ ] **被多层（N 层）各调用一次的函数**（如 patched_forward、per-layer hook）里，是否存在以下 CPU-GPU sync 操作：
  - `.item()`：从 GPU tensor 读标量到 CPU
  - `.numpy()`：GPU tensor 转 numpy
  - `bool(gpu_tensor)` / `int(gpu_tensor)`：Python 隐式类型转换 GPU tensor
  - `print(gpu_tensor)`：打印 GPU tensor（强制 sync）
  - `if gpu_tensor`：对 CUDA tensor 做 Python 条件判断

  → **后果**：N 层 × sync_cost 的额外延迟。Qwen2.5-7B 实测：28 层 × 18ms/sync = 504ms/step，足以把正确实现变成 3.7% HF baseline（Phase 6 教训）。

### 测试覆盖盲区
- [ ] 新增的 benchmark 脚本或 profiling 脚本：是否能在 dry_run=True 模式下完整构造 EngineConfig（验证参数名正确），不需要真实 GPU
- [ ] 新增的 GPU 路径代码：是否有对应的 dry_run 桩代码可供单元测试覆盖
- [ ] 如果没有测试覆盖，在 review 报告中明确标注"仅能在 GPU 实机验证"

### Benchmark 口径
- [ ] 新增 benchmark 文件是否说明了环境、模型、batch size、max_new_tokens
- [ ] 性能数字是否来自真实运行（不是估算）
- [ ] TTFT / TPOT / Peak Mem 口径是否与 `../infer-benchmark/METRICS_SPEC.md` 一致

### 文档
- [ ] 修改的文件是否已更新文件头说明
- [ ] 如果 API 行为发生变化，调用方的注释是否同步更新

## 不需要检查（留给 implement）
- 具体的修复方式（review 只定位问题，不提供实现）
- 风格偏好（命名、缩进等，只在严重影响可读性时才提）
- 未来可能的优化方向（除非是已知的正确性风险）
