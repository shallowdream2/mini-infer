# Phase 12 知识笔记：CUDA Graph

日期：2026-03-22

---

## CUDA Graph 是什么

CUDA Graph 把一段 CUDA kernel 序列录制成一张"图"，后续只需 `g.replay()` 就能重放整个序列，跳过 Python→CUDA dispatcher 的逐 kernel 调度开销。

录制期间，PyTorch 照常执行所有 Python 代码（条件分支、for 循环），但 CUDA 侧的 kernel launch 被拦截并录制。回放时只执行录制的 kernel，不再经过 Python。

---

## 为什么能用在 decode_batch

decode_batch 的结构：同一批请求，每步 forward 的 tensor 形状相同（input_ids=[bs,1]，position_ids=[bs,1]），只有数据不同。只要 batch size 固定，就满足 CUDA Graph 的静态形状要求。

动态性来源有三处，全部可解决：
1. **batch size 变化** → Graph Pool（为 bs={1,2,4,8} 各捕一张图，actual pad 到 padded）
2. **block_table 内容变化** → 静态 buffer + copy_()（地址不变，数据变）
3. **max_kv_len 随 seq_len 增长** → 固定为 config.max_model_len（RoPE cos/sin 形状恒定，position_ids 索引到正确行）

---

## 核心 API

```python
# 捕获
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = model(...)     # 录制 CUDA kernel 序列

# 更新输入（in-place，不改变 shape）
static_input.copy_(new_data)

# 回放
g.replay()
# 输出张量 out 已更新（和捕获时同一块内存）
```

---

## warmup 为什么必须在捕获前做

CUDA 首次执行某些 kernel 时会分配 workspace（如 cudnn、cublas 的临时缓冲区）。如果在捕获期间触发了新的内存分配，这个分配地址会被录制进 graph，下次 replay 时该地址可能已被其他 tensor 占用，导致数据错乱（不崩溃，但输出错误）。

warmup 后加 `torch.cuda.synchronize()` 确保所有 kernel 完成，再进入捕获。

---

## 关键限制

- **形状必须完全相同**：不能在 graph 内做动态判断、条件分支或形状变化的操作
- **不能录制 Python 操作**：`_paged_ctx.set()` / `.clear()` 是 Python 属性赋值，在 graph context 外执行
- **CPU-GPU sync 会破坏 graph 效率**：graph 回放期间如果有 `.item()` 等 sync 操作，会强制等待，部分抵消收益

---

## mini-infer 实测

1.5B（dispatch 占 29%）：+20–29% speedup
7B（dispatch 占 5%）：+4–5% speedup（接近理论上限）
