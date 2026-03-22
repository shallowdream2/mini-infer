# Tensor Parallelism：从权重切分到 NCCL 通信

## 主题

Tensor Parallelism（TP）在大语言模型推理中的原理、Megatron-LM 实现方案、在 Qwen2.5 上的工程实践，以及它与 Pipeline Parallel 的本质区别。

---

## 一、问题定义

**核心问题**：单卡 24 GB VRAM 装不下 70B 模型（fp16 需要 ~140 GB）。扩展服务能力需要把同一个请求的计算分摊到多张卡上。

Pipeline Parallel 是一种方案，但它是层间串行：每一步只有一张卡在做计算（其他卡在等激活张量传过来）。吞吐不提升，只解决了内存问题的一半。

Tensor Parallelism 是另一种方案：把同一层的 attention heads 和 FFN neurons 切分到多张卡，每一步所有卡同时做计算，forward 结束时通过 NCCL all-reduce 合并结果。这才是真正的层内并行。

---

## 二、核心原理

### 2.1 线性层的 TP 分解

线性层 `y = x @ W.T` 可以按两种方式切分：

**Column Parallel（沿输出维度切分，无通信）**：

```
W 按行切成两半：W0（前一半行），W1（后一半行）
rank 0: y0 = x @ W0.T    shape: (batch, seq, out_dim/2)
rank 1: y1 = x @ W1.T    shape: (batch, seq, out_dim/2)
合并: y = concat(y0, y1, dim=-1)   不需要通信
```

**Row Parallel（沿输入维度切分，需要 all-reduce）**：

```
W 按列切成两半：W0（前一半列），W1（后一半列）
x 也对应切分：x0（前一半），x1（后一半）—— 由 column parallel 输出
rank 0: y0 = x0 @ W0.T   shape: (batch, seq, out_dim)
rank 1: y1 = x1 @ W1.T   shape: (batch, seq, out_dim)
合并: y = y0 + y1         需要 all-reduce(SUM)
```

数学等价性验证（无需 GPU）：

```python
full = x @ W_col.T @ W_row.T

col0, col1 = W_col[:mid], W_col[mid:]          # column shard
row0, row1 = W_row[:, :mid], W_row[:, mid:]    # row shard
tp = x @ col0.T @ row0.T + x @ col1.T @ row1.T  # all-reduce
assert torch.allclose(full, tp, atol=1e-5)    # ✅
```

### 2.2 Transformer 层的切分模式

Column Parallel（Q/K/V/gate/up）+ Row Parallel（O/down）组合使用，只需在 row parallel 后各做一次 all-reduce：

```
prefill/decode:
  Q = x @ W_q.T    ← column parallel（无通信）
  K = x @ W_k.T    ← column parallel
  V = x @ W_v.T    ← column parallel
  attn_out = attention(Q, K, V)
  y_attn = attn_out @ W_o.T    ← row parallel → all-reduce ①

  gate = silu(x @ W_gate.T)    ← column parallel
  up   = x @ W_up.T            ← column parallel
  ffn_out = gate * up
  y_ffn = ffn_out @ W_down.T   ← row parallel → all-reduce ②
```

每层 2 次 all-reduce，N 层共 2N 次/forward pass。

### 2.3 GQA 下的 KV head 切分

GQA（Grouped Query Attention）中 KV heads 数量少于 Q heads：

```
Qwen2.5-7B: 28 Q heads，4 KV heads（TP=2 时，每卡 14 Q/2 KV）
Qwen2.5-1.5B: 12 Q heads，2 KV heads（TP=2 时，每卡 6 Q/1 KV）
```

约束：`num_key_value_heads % tp_size == 0` 必须成立，否则无法等分。

---

## 三、工程实现方式

### 3.1 Megatron-LM 风格（mini-infer Phase 13 的选择）

优点：不修改模型代码，用 PyTorch forward hook 注入 all-reduce；实现简单。

```python
# 权重切分（就地替换）
def _shard_qwen2_weights(model, rank, tp_size):
    for layer in model.model.layers:
        # column parallel
        for proj in (attn.q_proj, attn.k_proj, attn.v_proj):
            proj.weight = nn.Parameter(col_shard(proj.weight.data, rank, tp_size))
        # row parallel
        for proj in (attn.o_proj, mlp.down_proj):
            proj.weight = nn.Parameter(row_shard(proj.weight.data, rank, tp_size))
        # 更新 attn 元数据（必须！）
        attn.num_heads //= tp_size
        attn.num_key_value_heads //= tp_size
        attn.hidden_size = attn.num_heads * attn.head_dim

# forward hook 注入 all-reduce
def make_allreduce_hook():
    def hook(module, inputs, output):
        tensor = output[0] if isinstance(output, tuple) else output
        dist.all_reduce(tensor.contiguous(), op=dist.ReduceOp.SUM)
        return (tensor,) + output[1:] if isinstance(output, tuple) else tensor
    return hook

for layer in model.model.layers:
    layer.self_attn.register_forward_hook(make_allreduce_hook())
    layer.mlp.register_forward_hook(make_allreduce_hook())
```

### 3.2 多进程初始化

```python
# torchrun 模式（推荐，进程常驻，适合性能测量）
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
torch.cuda.set_device(rank)   # 必须在任何 CUDA API 前调用

# mp.spawn 模式（进程每次重建，仅适合功能验证）
init_method = f"file://{rendezvous_file}"
dist.init_process_group("nccl", init_method=init_method, rank=rank, world_size=tp_size)
```

### 3.3 generate 的同步要求

```python
output_ids = model.generate(
    **inputs,
    synced_gpus=(tp_size > 1),   # 必须！防止 EOS 时间不同导致 NCCL hang
)
```

---

## 四、设计取舍

### 4.1 load-then-shard vs shard-during-load

**mini-infer 选择**：load-then-shard（先加载全量，再就地替换分片权重）

- 优点：实现简单，不需要修改 `from_pretrained` 的内部逻辑
- 缺点：peak VRAM = 全量权重（不是 50%）；每张卡都加载了一次用不到的参数

**Megatron-LM / vLLM 选择**：shard-during-load（加载时按 rank 只读对应切片）

- 优点：peak VRAM ≈ 50%（真正减半）
- 缺点：需要了解 checkpoint 格式和 tensor 存储方式，工程量大

### 4.2 hook 注入 vs 修改模型代码

hook 注入不需要 fork 或修改 transformers 源码，但每层执行路径中会有额外的 Python 函数调用开销（每步 56 个 hook）。修改模型代码可以消除这一开销，但维护代价更高。对小规模实验，hook 是更好的选择。

### 4.3 mp.spawn vs torchrun

| | mp.spawn | torchrun |
|---|---|---|
| 进程生命周期 | 每次 generate 重建 | 常驻 |
| 模型加载 | 每次重新加载 | 一次 |
| 适合场景 | 功能验证 | 性能测量 |
| 通信初始化 | 文件锁 rendezvous | 环境变量 |

---

## 五、常见误区

**误区 1：TP=2 一定比单卡快**

不一定。小模型 + 小 batch decode 是 memory-bound，TP 的算力分摊无效，反而增加 NCCL 通信开销。TP 在 compute-bound 场景（大 batch prefill、70B+ 模型）才有加速效果。

**误区 2：PP 和 TP 是同一件事**

完全不同。PP 把不同层放到不同 GPU（层间串行），TP 把同一层切分到不同 GPU（层内并行）。PP 的 VRAM 减半但吞吐不变；TP 在合适场景下同时降低 VRAM 和提升吞吐。

**误区 3：切分权重就够了，不需要更新元数据**

错。Qwen2.5 的 attention forward 里有 `attn_output.reshape(bsz, q_len, self.hidden_size)`。切分后 `self.hidden_size` 必须更新为 `num_heads_per_rank × head_dim`，否则 reshape 报错。

**误区 4：eager 模式能用**

在 transformers 4.43.4 上，Qwen2.5 的 eager attention 实现有 attention mask bug，会产生乱码输出。TP 模式下必须用 `flash_attention_2`。

---

## 六、和 mini-infer 的关系

Phase 13 在以下位置实现了 TP：

- `mini_infer/tp_model_runner.py` — 核心：col/row shard + hook 注入 + TensorParallelModelRunner
- `mini_infer/tp_engine.py` — 外层：mp.spawn 启动多进程 worker，主进程 API
- `benchmarks/benchmark_tp.py` — 对比 single/pp/torchrun_tp 三种模式的吞吐和 VRAM
- `tests/test_tp_engine.py` — 13 个 dry_run 测试，无需 GPU

**实测结果**（1.5B，bs=3，decode，2 × RTX 4090）：

| 模式 | 吞吐 | VRAM/卡 |
|------|------|---------|
| single | 98.0 tok/s | 3.58 GB |
| pp | 82.4 tok/s | ~1.8 GB |
| tp=2 | 76.5 tok/s | 3.57 GB（peak，含全量加载） |

tp=2 低于单卡符合预期：小模型 decode memory-bound + 56 次/step NCCL 开销。

---

## 七、进一步阅读

- Megatron-LM 论文：*Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM*（2021）
- vLLM TP 实现：`vllm/model_executor/layers/linear.py`（ColumnParallelLinear / RowParallelLinear）
- Megatron-LM 源码：`megatron/core/tensor_parallel/layers.py`
- PyTorch distributed：`torch.distributed` NCCL backend 文档
