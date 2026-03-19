# 知识专题：LLM 推理中的数据并行与模型并行

本文深度解析 LLM 推理多卡扩展的两种主要策略——数据并行（Replica）和模型并行（Tensor Parallel / Pipeline Parallel）——的原理、工程实现方式、适用场景和常见误区，结合 mini-infer Phase 4 的真实实验数据。

---

## 主题

**LLM 推理多卡扩展：Replica vs Tensor Parallel vs Pipeline Parallel 的真实边界**

---

## 一、问题定义

单卡 GPU 在两个维度上存在上限：
1. **显存上限**：模型权重 + KV cache，超过 VRAM 就 OOM
2. **计算上限**：batch 增大到一定程度后，throughput 的边际增益递减，最终受限于显存带宽和 CUDA core 利用率

多卡扩展要解决的问题是：如何利用第二块（或更多）GPU，突破单卡的这两个上限？

---

## 二、核心原理

### 2.1 数据并行（Replica）

每块 GPU 持有**完整的模型副本**，接收不同的请求子集，独立完成 forward。结果汇总后返回。

- **显存**：每卡一份权重，总显存 = N × 单卡显存
- **吞吐**：理论 N×，实际取决于每卡 batch 效率
- **延迟**：不变（单请求只在一卡上跑）
- **通信**：无（推理时不需要卡间通信，只有结果汇总）

### 2.2 Pipeline Parallel（PP）

将模型的**层**按顺序分配到多卡。对 L 层模型分到 N 卡，每卡跑 L/N 层。层间通过激活张量（`[batch, seq, hidden]`）传递。

- **显存**：每卡 1/N 的权重，但 KV cache 可能还是集中在某卡
- **吞吐**：不提升（总计算量不变，层是串行的）
- **延迟**：单请求 TTFT 可能略增（层间等待），多请求流水线下可重叠
- **通信**：每层边界传一次激活，量级 O(batch × seq × hidden)，通常远小于计算

HuggingFace 的 `device_map="balanced"` 是 PP 的朴素实现：不做流水线调度，就是把层按顺序切分到各卡，激活张量在层边界自动搬移。

### 2.3 Tensor Parallel（TP）

将每层的**权重**按 head 维度切分到多卡。对 Attention 层，每卡持有部分 Q/K/V 头，各自计算，通过 all-reduce 合并结果。

- **显存**：每卡 1/N 的权重，KV cache 也按 head 切分
- **吞吐**：理论 N×（每层计算量减半），实际受 all-reduce 通信延迟影响
- **延迟**：单请求 TTFT 显著降低（每层计算量减半）
- **通信**：每层一次 all-reduce，量级 O(batch × seq × hidden)，时延取决于 NVLink vs PCIe

TP 需要专用 CUDA kernel 支持（flash_attn 2.5+ 的 `block_tables` 参数，或 Megatron-LM 的 ColumnParallelLinear/RowParallelLinear），不能用标准 HF `forward()` 实现。

---

## 三、工程实现方式

### Replica 实现

```python
class ReplicaEngine:
    def __init__(self, config_0, config_1):
        self.engines = [LLMEngine(config_0), LLMEngine(config_1)]

    def generate(self, prompts, max_new_tokens=128):
        group_0 = [(i, p) for i, p in enumerate(prompts) if i % 2 == 0]
        group_1 = [(i, p) for i, p in enumerate(prompts) if i % 2 == 1]
        results = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            f0 = ex.submit(self._run, 0, [p for _, p in group_0], max_new_tokens)
            f1 = ex.submit(self._run, 1, [p for _, p in group_1], max_new_tokens)
            for (i, _), o in zip(group_0, f0.result()): results[i] = o
            for (i, _), o in zip(group_1, f1.result()): results[i] = o
        return [results[i] for i in range(len(prompts))]
```

Python `ThreadPoolExecutor` 线程 + CUDA 异步：Python GIL 序列化 Python 代码，但两张 GPU 的 kernel 在硬件上真正并行执行。不需要多进程（多进程会序列化 CUDA context，代价更高）。

### PP 实现（HF device_map）

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="balanced",  # HF accelerate 自动把层分配到各 GPU
    torch_dtype=torch.float16,
)
# 输入必须送到 embedding 层所在的 device
first_device = next(model.parameters()).device
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(first_device)
output_ids = model.generate(**inputs, max_new_tokens=128)
```

注意 `padding_side="left"`：decoder-only 批推理必须左填充，否则 position_id 在 padding token 处偏移，生成质量静默降低。

### TP 实现（非 mini-infer 当前实现）

需要：
- flash_attn 2.5+ 的 `block_tables` 参数，支持在 attention kernel 内按 head 寻址不同 GPU 上的 KV
- 或 Megatron-LM / DeepSpeed 的 `ColumnParallelLinear`/`RowParallelLinear`
- 或手动切分 Q/K/V/O 权重 + 实现 all-reduce（复杂度高）

mini-infer Phase 4 未实现真正 TP，仅测量了 HF PP（`device_map="balanced"`）作为对比基线。

---

## 四、设计取舍

### 什么时候选 Replica

适用条件：
1. 单卡显存能装下完整模型
2. 总 batch 超过单卡最优 batch（单卡 OOM 或效率下降）

关键洞察：**Replica 的吞吐收益 = 两卡并发的上限 - 单卡当前效率**。如果单卡已经高效（batch 大、GPU 利用率高），Replica 收益小；如果单卡 batch 被显存限制，Replica 可以真正 2×。

mini-infer Phase 4 数据：单卡 batch=8 是 361 tok/s，Replica batch=4+4 理论上限 388 tok/s，增量只有 7%，实测 +4.1%。

真正 2× 的场景：总 batch=16，单卡因 KV 显存不足 OOM，Replica 每卡 batch=8，各跑 361 tok/s，合计 722 tok/s。

### 什么时候选 PP

适用条件：
1. 单卡显存**装不下**完整模型权重（如 70B 模型，单卡 24 GB 不够）
2. 对延迟要求不高，接受层间串行的开销

PP 不提升吞吐，只解决显存问题。通信量小（激活传输远小于权重），但层间有串行等待。

### 什么时候选 TP

适用条件：
1. 单卡显存装不下，且对**单请求延迟**有要求（TTFT sensitive）
2. 有 NVLink（PCIe 的 all-reduce 延迟会抵消 TP 的计算收益）

TP 真正拆分层内计算，每层的 TTFT 贡献减半，但需要高带宽卡间通信（NVLink bandwidth >> PCIe）。在 PCIe 连接的双卡上，TP 的 all-reduce 延迟可能完全抵消计算收益，不如 Replica 或 PP。

---

## 五、常见误区

**1. "加一块 GPU 就应该翻倍"**

不对。Replica 翻倍的前提是每卡 batch 的效率与单卡大 batch 相当。当单卡已在最优 batch 附近高效运行时，Replica 的增量很小。

**2. "HF device_map='balanced' 是 Tensor Parallel"**

不对。这是 Pipeline Parallel——层是完整的，只是分布在不同 GPU 上。TP 需要在层内切分权重并 all-reduce，HF 的标准 `forward()` 不做这件事。

**3. "PP 会让推理更快"**

对于吞吐，不对。PP 的总计算量不变，只是分在两卡上串行完成，批推理 throughput 与单卡相同。PP 的潜在价值是流水线并行（多个 micro-batch 同时在不同层运行），但 HF 的 `model.generate()` 不实现这个。

**4. "右填充可以用于批推理，只要 attention_mask 正确"**

理论上 attention_mask 可以让 transformer 忽略 padding token，但 decoder-only 模型生成时，新 token 紧接在序列末尾，如果末尾有 padding token，position_id 会偏移。实践中左填充更安全，且 HF 推荐。

---

## 六、和 mini-infer 的关系

Phase 4 实现了：
- `ReplicaEngine`：完整的 Replica 双卡引擎，集成 mini-infer 自定义 KV cache
- `TPEngine`：测量用 PP 引擎（不集成自定义 KV cache，因 PP 下每层 KV 设备不同，与 `KVCacheManager` 单设备 pool 不兼容）

实验结论：
- Replica batch=8：+4.1%（batch 太小，单卡已接近最优）
- PP batch=8：+0.1%，每卡显存减半

真正值得做的下一步（Phase 5 评估）：
- 扩充测试 batch 到 16+，验证 Replica 的真实 2× 场景
- 升级 flash_attn 2.5+，实现真正 TP（消除 gather 复制 + 层内并行）

---

## 七、进一步阅读

- Megatron-LM 论文（Shoeybi et al., 2019）：Tensor Parallel 的原始设计
- PipeSwitch / GPipe：Pipeline Parallel 的微批次流水线实现
- vLLM 论文中的多卡扩展章节：Tensor Parallel + Replica 结合
- HuggingFace Accelerate 文档：`device_map` 的工作原理
- NVLink vs PCIe 带宽对 all-reduce 延迟的影响
