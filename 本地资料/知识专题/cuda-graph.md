# 知识专题：CUDA Graph 在 LLM 推理中的应用

---

## 一、问题定义

LLM 自回归 decode 的每个 step，PyTorch 需要：
1. 遍历模型所有层
2. 为每层每个算子构造 kernel 启动参数
3. 依次提交到 CUDA stream

这个"Python→CUDA dispatcher"的调度过程不涉及 GPU 计算，但在 CPU 侧消耗固定时间。对于 1.5B 这类 forward 极快的模型，这个开销可占总 step 时间的 20–30%，是可观的瓶颈。

CUDA Graph 的核心思路：把 forward 录制一次，后续每步直接回放录制好的 kernel 序列，彻底跳过 Python dispatch。

---

## 二、核心原理

### CUDA Graph 的工作机制

```
第一次（捕获）：
  Python 代码正常运行，但 CUDA kernel launch 被拦截并记录
  每个 kernel 的地址、参数、依赖关系被存入 CUDAGraph 对象

后续每步（回放）：
  直接调用 g.replay()，CUDA driver 重放所有 kernel
  完全跳过 Python dispatcher、PyTorch autograd、dispatch 表查找
```

回放时 kernel 读写的 GPU 内存地址与捕获时相同。因此，通过 `tensor.copy_()` 更新这些地址的内容，回放时就能使用最新数据。

### 静态形状约束

CUDA Graph 要求录制和回放时 tensor 的：
- **shape**：完全相同（否则某些 kernel 的 block size 配置失效）
- **dtype**：相同
- **设备**：相同

值可以变（通过 `copy_()`），形状不能变。

---

## 三、工程实现方式

### 接入 LLM decode 的标准模式

```python
# 1. 分配静态 buffer（形状固定，与录制时一致）
static_input_ids      = torch.zeros(bs, 1, dtype=torch.long, device=device)
static_position_ids   = torch.zeros(bs, 1, dtype=torch.long, device=device)
static_cache_seqlens  = torch.zeros(bs, dtype=torch.int32, device=device)
static_block_table    = torch.zeros(bs, max_blocks, dtype=torch.int32, device=device)

# 2. Warmup（让 CUDA 分配好 workspace，避免捕获时触发动态分配）
for _ in range(3):
    out = model(input_ids=static_input_ids, ...)
torch.cuda.synchronize()

# 3. 捕获
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    static_logits = model(input_ids=static_input_ids, ...)[:, 0, :]

# 4. 每步回放
static_input_ids.copy_(actual_input_ids)   # 更新数据
static_block_table.copy_(actual_block_table)
g.replay()                                  # 回放（不经过 Python）
result = static_logits[:actual_bs].clone() # 取结果
```

### 处理动态 batch size：Graph Pool

```python
graph_pool = {}  # batch_size → CUDAGraph

for bs in [1, 2, 4, 8]:
    graph_pool[bs] = capture_graph(bs)

def decode_with_graph(actual_bs, ...):
    padded_bs = min(s for s in graph_pool if s >= actual_bs)
    # copy_() 把 actual_bs 行数据填入 padded_bs 的静态 buffer
    # pad 行填零
    graph_pool[padded_bs].replay()
    return static_logits[padded_bs][:actual_bs]
```

### 处理 RoPE 的 max_kv_len 问题

RoPE 实现通常用 `seq_len` 参数控制 cos/sin 缓存的大小：

```python
cos, sin = rotary_emb(v, seq_len=max_kv_len)  # 输出 [max_kv_len, head_dim]
```

若 `max_kv_len` 每步增长，`cos/sin` 的形状就变，graph 失效。

解决方案：**固定 `max_kv_len = config.max_model_len`**。cos/sin 始终生成 `[max_model_len, head_dim]`，形状恒定。实际使用的位置通过 `position_ids`（可 copy_() 的静态 buffer）索引，正确性不受影响。

代价：每步生成完整长度的 cos/sin 表（通常是 slice 操作，开销极小）。

---

## 四、设计取舍

### Graph Pool 大小

捕获更多 batch size → 更少 padding overhead，但 warmup + 捕获时间增加（每张图约 1–2s），显存占用略增。

实践中 {1, 2, 4, 8} 是合理选择，覆盖大多数 online serving 场景。超出最大值时降级 eager，不影响正确性。

### 启动时间 vs 推理效率

启动捕获 4 张图约需 5–8s 额外时间。对于长期运行的服务来说是合算的；对于一次性脚本或 batch 离线场景则不值，这也是 `use_cuda_graph` 默认 `False` 的原因。

### 哪些操作不能进 graph

以下操作必须在 graph 外执行：
- `ensure_next_slot()`：分配物理 KV 块（修改 Python dict）
- `advance_seq_lens()`：递增 seq_len（修改 Python dict）
- token 采样：`_sample_token()`（依赖 CPU 随机数生成）
- `_paged_ctx.set()` / `clear()`：Python 属性赋值

graph 内只有纯 CUDA 计算（attention + 线性层）。

### 错误处理

`g.replay()` 应用 `try/finally` 包裹，确保 `_paged_ctx.clear()` 在任何情况下都执行。如果 replay 失败后不 clear，下次 prefill 调用会看到 `ctx.block_table is not None`，误走 decode 路径，静默输出错误。

---

## 五、常见误区

**误区 1：graph 捕获会提高运算精度或改变算法**
不会。graph 只改变 kernel 调度方式，不改变计算内容。输出与 eager 模式完全一致。

**误区 2：graph 对大模型加速效果也很好**
取决于 dispatch overhead 占 step 总时间的比例。7B 模型的 model forward 约 15ms，dispatch 约 1ms（占 5%）；1.5B 的 forward 约 3ms，dispatch 约 2ms（占 40%）。大模型收益有限。

**误区 3：warmup 可以省略**
不能。首次运行会触发 CUDA workspace 分配，如果在捕获期间分配，会导致 replay 时数据错乱（不报错，输出错误）。warmup 至少跑 3 次，且需要 `cuda.synchronize()`。

**误区 4：graph 内可以做 `.item()` 或 CPU 操作**
`.item()` 触发 GPU→CPU sync，会破坏 graph 的异步执行。如果必须有 scalar，应在 graph 外提前计算（如 max_kv_len 在 graph 外计算后通过 Python 变量传入）。

---

## 六、和 mini-infer 的关系

Phase 12 的具体实现见 `mini_infer/model_runner.py`：
- `warmup_cuda_graphs()`：启动时捕获，bs={1,2,4,8}，固定 max_kv_len=config.max_model_len
- `_graph_decode_forward()`：copy_() + replay + try/finally
- `decode_batch()`：通过 `_find_padded_bs()` 分路，graph pool 为空时降级 eager

实测（RTX 4090，Qwen2.5-1.5B）：bs=1 时 dispatch overhead 从 ~29% 降到 ~4%，step 延迟从 7.33ms 降到 5.21ms（+28.9%）。

---

## 七、进一步阅读

- PyTorch 官方文档：[CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)
- vLLM 实现参考：`vllm/worker/model_runner.py` 的 `capture_model()`
- Leviathan et al. 2023（Speculative Decoding）——类似的"一次录制，多次使用"思路在算法层面的体现
