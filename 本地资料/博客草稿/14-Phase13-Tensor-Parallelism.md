# 从 Pipeline Parallel 到真正的 Tensor Parallel：一次权重切分实验

## 起点：一个骗人的 TPEngine

在 Phase 4 实现"双卡扩展"时，我写了一个叫 `TPEngine` 的类。它实际上是这样的：

```python
# 旧 mini_infer/tp_engine.py
from .pp_engine import PPEngine as TPEngine
```

一行别名。背后是 `device_map="balanced"`，把模型的不同层放到不同 GPU 上——这是 Pipeline Parallel（PP），不是 Tensor Parallel（TP）。两者的区别是本质性的：PP 是层间串行，每一步只有一张卡在工作；TP 是层内并行，每个 attention head 的计算同时分摊到所有卡上。

Phase 13 的目标是把这个骗局终结，实现真正的 Tensor Parallelism。

---

## TP 的核心思想

以 Qwen2.5-1.5B 的一层 Transformer 为例：

```
attention:
  q_proj: (1536, 1536) — 12 heads × 128 head_dim = 1536
  k_proj: (256, 1536)  — 2 KV heads × 128 = 256  (GQA)
  v_proj: (256, 1536)
  o_proj: (1536, 1536)

FFN:
  gate_proj: (8960, 1536)
  up_proj:   (8960, 1536)
  down_proj: (1536, 8960)
```

Megatron-LM 的切分方案是：

**Column Parallel（无通信）**：Q/K/V/gate/up 沿输出维度（dim=0）切分。TP=2 时，rank 0 和 rank 1 各取一半的行。每张卡独立做 `x @ W_col_shard.T`，不需要通信，因为两块卡的计算互不依赖。

**Row Parallel（需要 all-reduce）**：O/down 沿输入维度（dim=1）切分。每张卡算出的是**部分和**：`x_partial @ W_row_shard.T`。两张卡加起来才是完整结果。所以 forward 后必须做 NCCL all-reduce（SUM）。

数学上，这个等价性是：

```
完整计算：y = x @ W_col.T @ W_row.T

TP=2 切分：
  rank 0: y0 = x @ col0.T @ row0.T
  rank 1: y1 = x @ col1.T @ row1.T
  all-reduce: y = y0 + y1 ← 数学等价
```

验证这一点很简单，不需要 GPU：

```python
W_col = torch.randn(mid_d, in_d)
W_row = torch.randn(out_d, mid_d)

full_out = x @ W_col.T @ W_row.T

col0, col1 = col_shard(W_col, 0, 2), col_shard(W_col, 1, 2)
row0, row1 = row_shard(W_row, 0, 2), row_shard(W_row, 1, 2)

tp_out = x @ col0.T @ row0.T + x @ col1.T @ row1.T  # all-reduce(SUM)
assert torch.allclose(full_out, tp_out, atol=1e-5)
```

---

## 实现细节：三个非显然的地方

### 1. 权重切分之后还要更新 attn 的元数据

`_shard_qwen2_weights` 不只是替换权重矩阵，还要更新注意力模块的属性：

```python
attn.num_heads = attn.num_heads // tp_size           # 12 → 6
attn.num_key_value_heads = attn.num_key_value_heads // tp_size  # 2 → 1
attn.num_key_value_groups = attn.num_heads // attn.num_key_value_heads  # 6//1=6 (不变)
attn.hidden_size = attn.num_heads * attn.head_dim    # 6×128=768（原 1536）
```

最后一行最关键。Qwen2.5 的 attention forward 里有这一行：

```python
attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
```

切分后每张卡只有一半的 attention heads，输出 shape 是 `(bsz, q_len, num_heads_per_rank × head_dim)`。如果 `self.hidden_size` 还是原始值 1536，`reshape` 会因为 shape 不匹配而报错或（更糟糕地）产生错误输出。

### 2. all-reduce 不需要修改模型代码，用 forward hook 注入

不改 transformers 源码，用 PyTorch 的 forward hook 机制：

```python
def make_allreduce_hook():
    def hook(module, inputs, output):
        if isinstance(output, tuple):
            # Qwen2Attention 返回 (hidden_states, attn_weights, past_key_value)
            tensor = output[0].contiguous()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return (tensor,) + output[1:]
        else:
            output = output.contiguous()
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            return output
    return hook

for layer in model.model.layers:
    layer.self_attn.register_forward_hook(make_allreduce_hook())
    layer.mlp.register_forward_hook(make_allreduce_hook())
```

self_attn 和 mlp 各一个 hook，1.5B 的 28 层共注入 56 个 hook，每个 forward 步触发 56 次 all-reduce。

### 3. GQA 的切分边界

Qwen2.5-1.5B 是 GQA（12 Q heads，2 KV heads，TP=2）。切分后每张卡有 6 Q heads 和 1 KV head。这满足整除条件，但 Qwen2.5-0.5B 只有 2 KV heads（TP=2 时每卡 1 head）。如果 KV heads 数量不能被 tp_size 整除，切分会失败。在 `_shard_qwen2_weights` 里加了 assert：

```python
assert attn.num_key_value_heads % tp_size == 0, \
    f"num_key_value_heads={attn.num_key_value_heads} 不能被 tp_size={tp_size} 整除"
```

---

## 踩了哪些坑

### 坑 1：eager 模式输出全是感叹号

第一次跑通 TP forward 时，输出是这样的：

```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
```

不是因为 TP 切分出了问题，而是 `attn_implementation="eager"` 本身在 transformers 4.43.4 上有 attention mask bug。单卡也会复现，只是加了 TP 才第一次注意到这个问题（单卡测试用的是别的路径）。

换成 `attn_implementation="flash_attention_2"` 后，输出恢复正常，且 TP=2 与单卡的 greedy 输出完全一致（3/3 prompts 完全匹配）。flash_attention_2 使用相同的 `self.num_heads` / `self.hidden_size` reshape 模式，权重切分策略不变。

### 坑 2：VRAM 没有减半

直觉上，TP=2 每张卡只需要保存一半的权重，VRAM 应该接近单卡的 50%。实际测量：

| 模式 | GPU0 | GPU1 |
|------|------|------|
| single | 3.58 GB | — |
| tp=2 | 3.57 GB | 3.57 GB |

每张卡的 peak VRAM 和单卡几乎一样。原因是实现方式：先用 `from_pretrained(device_map="cuda:0")` 把完整模型加载到每张卡，再就地替换为分片权重。旧权重张量还没被 GC 释放时，peak memory 已经被记录了。

模型初始化完成后（`torch.cuda.memory_allocated()`）显示每卡 3.11 GB，但 peak 在 3.57 GB。正确的做法是 shard-during-load（加载时按 rank 只读取对应切片），Megatron-LM 的 `from_pretrained` 就是这样做的，工程量较大，Phase 13 暂时不实现。

### 坑 3：torchrun 下 CUDA 设备未初始化

用 `torchrun --nproc_per_node 2` 启动时，`dist.init_process_group("nccl")` 之后调用 `torch.cuda.reset_peak_memory_stats("cuda:0")` 会报：

```
RuntimeError: Invalid device argument 0: did you call init?
```

NCCL backend 初始化不会自动初始化 CUDA 设备。需要显式调用：

```python
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
device = f"cuda:{rank}"
torch.cuda.set_device(rank)   # ← 这行必须在任何 CUDA API 之前
torch.cuda.reset_peak_memory_stats(device)
```

### 坑 4：mp.spawn 模式不能用来测吞吐

`TPEngine.generate()` 内部用 `mp.spawn` 启动 worker，每次调用都会重新 fork 进程、初始化 NCCL、加载完整模型权重。benchmark 里的 warmup 和 measure 循环每次都在做这些事，计时包含了 5-10 秒的模型加载时间，吞吐数字完全没有参考价值。

真正的吞吐测量必须用 `torchrun` 模式：进程常驻，模型只加载一次，warmup 之后再计时。

---

## Benchmark 结果

环境：2 × RTX 4090，Qwen2.5-1.5B-Instruct，float16，max_new_tokens=64，3 prompts 顺序生成。

| 模式 | 吞吐 (tok/s) | GPU0 VRAM | GPU1 VRAM | 相对单卡 |
|------|-------------|-----------|-----------|---------|
| single（单卡 HF generate） | 98.0 | 3.58 GB | — | 100% |
| pp（device_map=balanced） | 82.4 | 2.03 GB | 1.56 GB | 84.1% |
| tp=2（torchrun + NCCL） | 76.5 | 3.57 GB | 3.57 GB | 78.1% |

TP=2 比单卡慢，符合预期。原因不复杂：

**小模型 + 小 batch decode 是 memory-bound**。1.5B 的参数量只有约 3 GB，每个 decode step 的瓶颈是把这 3 GB 读进 SM 的内存带宽，不是 FLOPS。TP=2 把每张卡的权重读取量减半，但同时引入了 56 次/step NCCL all-reduce（28 层 × 2 hook）。在单机双卡、PCIe 连接的环境下，all-reduce 的延迟叠加超过了带宽节省的收益。

TP 的真实价值在这个实验里看不到，因为 1.5B 不需要多卡。它的收益场景是：

1. **模型 > 单卡显存**：70B+ 模型单卡 OOM，TP 是唯一选项
2. **大 batch prefill**：compute-bound 场景，TP 的算力分摊才有意义
3. **有 NVLink 的高速互联**：all-reduce 延迟降到亚毫秒级，通信开销可忽略

---

## 正确性验证

TP=2 生成结果与单卡完全一致（greedy decode，3/3 prompts 文本完全匹配）：

```
single: 量子计算是一种基于量子力学原理的新型计算方式，它利用了量子比特（qubit）来存储和处理信息。
tp=2:   量子计算是一种基于量子力学原理的新型计算方式，它利用了量子比特（qubit）来存储和处理信息。
```

数值等价性由 13 个 dry_run 测试覆盖（无需 GPU/NCCL），通过 mock all-reduce（rank 0 和 rank 1 的 partial 结果手动加和）验证了完整 TP forward 与单卡 forward 的等价性。

---

## 没做的事

- **7B 模型验证**：路线图要求用 7B，实际用了 1.5B。7B 的 snapshot shard 2-4 软链接缺失，改用 1.5B。切分逻辑相同。
- **all-reduce 开销量化**：没有用 CUDA event 测量每次 hook 的通信时间，不知道 56 次 all-reduce 各占多少 decode step 时间。
- **shard-during-load**：每张卡加载完整权重再切片，VRAM peak 未达理想 50%。

---

## 总结

三句话：

1. Pipeline Parallel 不是 Tensor Parallel。PP 是层间串行，TP 才是层内并行。

2. TP 的 Megatron-LM 实现要点：column parallel（无通信，dim=0 切分）+ row parallel（forward 后 all-reduce，dim=1 切分），以及切分后更新 `attn.num_heads` / `hidden_size` 等元数据。

3. 小模型 + 小 batch decode 不是 TP 的适用场景。TP 的价值在于让单卡装不下的模型能跑起来，或者在 compute-bound 的大 batch prefill 上分摊算力。

---

*数据来源：`本地资料/实验记录/2026-03-22-phase13-benchmark.md`*
*实现代码：`mini_infer/tp_model_runner.py`，`mini_infer/tp_engine.py`*

---

## 自查结果（QUALITY_CHECKLIST.md）

- [x] 文章主题对应真实项目阶段或真实技术问题 — Phase 13 真实实现，非概念综述
- [x] 结构覆盖：背景、问题定义、方案设计、实现细节、实验结果、坑点反思、总结 — 全部覆盖
- [x] 有明确背景和问题定义 — 从"一行别名"的旧实现出发，问题清晰
- [x] 有设计取舍，而不是只罗列概念 — 解释了为何用 hook 而非改源码；为何不做 shard-during-load；mp.spawn vs torchrun 的取舍
- [x] 有代码、实验或运行结果支撑 — 代码示例来自实际代码，数字来自实验记录文件
- [x] 有失败点、坑点或反思 — 4 个坑（eager mask bug、VRAM 未减半、CUDA init 顺序、mp.spawn 计时失效）全部如实描述
- [x] 语言克制、专业、可发表 — 无夸大，TP=2 比单卡慢的结果如实呈现并解释
