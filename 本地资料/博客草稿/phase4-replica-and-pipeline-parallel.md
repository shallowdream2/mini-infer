# 为什么 Replica 双卡只快了 4%：批大小才是真正的瓶颈

这篇文章是 mini-infer 项目的第四篇技术复盘。Phase 3 的单卡吞吐已经达到 HF baseline 的 88%，Phase 4 想往上走：加一块 GPU，理论上吞吐翻倍。实测结果是 +4.1%。这篇文章解释为什么。

---

## 一、两种双卡策略

**Replica（数据并行）**：两块 GPU 各跑一个完整模型。请求按 round-robin 分配，两个引擎并发执行，结果合并返回。显存翻倍（每卡一份完整权重），吞吐理论 2×，单请求延迟不变。

**Pipeline Parallel（PP）**：用 HuggingFace 的 `device_map="balanced"` 把模型层均匀分到两卡。对 Qwen2.5-7B（28层），每卡跑 14 层，激活张量在层边界传递。单卡显存减半，但吞吐不变（层是串行的）。

注意这里的 PP 不是 Tensor Parallel（TP）。真正的 TP 是把每层的权重按 head 维度切分，需要 all-reduce，需要 flash_attn 或 Megatron-LM 等框架支持，实现复杂很多。这里用 HF 的 `device_map` 做的是最朴素的 PP——层间流水，不是层内切分。

---

## 二、Replica 实现

实现很简单：两个独立的 `LLMEngine`，`ThreadPoolExecutor` 并发跑，合并结果时按原始索引恢复顺序。

```python
class ReplicaEngine:
    def __init__(self, config_0: EngineConfig, config_1: EngineConfig) -> None:
        if config_0.device == config_1.device:
            raise ValueError("ReplicaEngine 要求两个 config 使用不同的设备")
        self.engines = [LLMEngine(config_0), LLMEngine(config_1)]

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        group_0 = [(i, p) for i, p in enumerate(prompts) if i % 2 == 0]
        group_1 = [(i, p) for i, p in enumerate(prompts) if i % 2 == 1]

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(self._run_engine, 0, [p for _, p in group_0], max_new_tokens)
            future_1 = executor.submit(self._run_engine, 1, [p for _, p in group_1], max_new_tokens)
            outputs_0, outputs_1 = future_0.result(), future_1.result()

        for (orig_idx, _), out in zip(group_0, outputs_0):
            results[orig_idx] = out
        for (orig_idx, _), out in zip(group_1, outputs_1):
            results[orig_idx] = out
        return [results[i] for i in range(len(prompts))]
```

两个引擎绑定不同 GPU（`cuda:0`/`cuda:1`），KV cache 和模型权重完全独立，不共享任何状态。Python 的 GIL 会序列化 Python 代码，但 CUDA 操作是异步的——两个 GPU 可以在 Python 层被 GIL 持有时各自在硬件上跑，实际并发是真实的。

---

## 三、数据

环境：Ubuntu 24.04，2 × RTX 4090（各 24 GB），Qwen2.5-7B-Instruct，float16，transformers 4.43.4。

```
bench: batch=8, max_new_tokens=128

single  : 361.4 tok/s   TTFT 19.4ms  Peak GPU0 16.42 GB
replica : 376.1 tok/s   TTFT 23.1ms  Peak GPU0 16.31 GB  GPU1 16.31 GB
tp2(PP) : 361.5 tok/s   TTFT 21.4ms  Peak GPU0  7.00 GB  GPU1  8.97 GB
```

Replica：+4.1%。PP：+0.1%（基本无差别），但每卡显存减半。

---

## 四、为什么 Replica 只快了 4%

直觉上 "两块 GPU 应该快 2×"，但这个直觉有一个隐含前提：**每块 GPU 单独跑的效率与单卡跑整个 batch 时相同**。在这里这个前提不成立。

关键数据来自 Phase 3 的单卡 benchmark：

| batch | 单卡 throughput |
|-------|----------------|
| 1 | 53.7 tok/s |
| 4 | 194.2 tok/s |
| 8 | 361.3 tok/s |

batch=8 时，Replica 将请求拆为两组各 4 条，每卡跑 batch=4。两卡并发的理论上限是 2 × 194.2 = **388.4 tok/s**。实测 376.1 tok/s，效率 96.9%，说明并发确实是真实的。

但单卡 batch=8 已经是 361.3 tok/s——它已经接近 388.4 的 93%。所以 Replica 的增量空间只有 7%，实测 4.1%。

这不是 Replica 的失败，是 **GPU 吞吐对 batch 的次线性增长**造成的。batch 从 4 → 8，吞吐从 194 → 361（+86%），而不是线性的 +100%。单卡在 batch=8 就已经在高效利用 GPU，Replica 把 batch=8 拆成 4+4 反而丢失了这部分效率。

**什么时候 Replica 真的有 2× 效果？**

当单卡 batch 超过显存或调度上限时。假设需要同时处理 16 条请求：单卡 batch=16 的 KV cache 可能导致 OOM，而 Replica 可以每卡跑 batch=8，各自 361 tok/s，合计 722 tok/s ≈ 2×。

当前测试用 8 条 prompt，最多跑 batch=8，正好是单卡也能高效处理的规模。没有见到 Replica 发挥空间的场景。

---

## 五、PP 的实际用途

PP（`device_map="balanced"`）吞吐与单卡持平，但每卡显存从 16.42 GB 降到 7~9 GB。

这对 Qwen2.5-7B 来说没什么意义——它本来就能放进单卡。PP 的真正价值场景：**70B 模型，单卡 24 GB 装不下，双卡 PP 可以装下并运行**。

PP 的通信代价在批推理场景很小：每层边界只传一次激活张量（`[batch, seq_len, hidden_size]`），计算远多于通信。但在单请求低延迟场景，层间等待会增加 TTFT，因为 GPU0 要等 GPU1 接收激活才能开始下一层的计算，真正的流水线深度需要多个独立请求才能填满。

---

## 六、一个值得记录的坑

跑 tp2 的 benchmark 时出现了这个警告：

```
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set padding_side='left'.
```

HuggingFace tokenizer 默认右填充（padding 加在序列末尾），但 decoder-only 模型生成时，新 token 紧接在输入序列之后，如果输入序列末尾是 padding token，模型会在 padding 后面生成，position_id 的计算就错了。

正确做法是**左填充**（padding 在左边），这样无论输入多长，真实 token 都靠右对齐，新生成的 token 在正确位置接续。

修复一行代码：

```python
self.tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name,
    trust_remote_code=True,
    padding_side="left",  # 加这一行
)
```

这个问题在单请求场景（batch=1）不出现，因为没有填充。只要批处理 prompt 长度不一致，右填充就会静默地影响生成质量——而且模型通常还是会输出"看起来正常"的文本，只是内容偏了，不是明显的乱码。很容易被忽略。

---

## 七、小结

这一阶段的实验结论可以简单归纳：

- **Replica 适合高并发、大 batch 场景**，而不是"加块 GPU 就翻倍"。在 batch=8 的规模下，单卡已经在高效运行，Replica 没有足够的 scaling 空间。
- **PP 的价值是显存，不是吞吐**。`device_map="balanced"` 可以跑装不进单卡的大模型，但对单卡能装下的模型没有吞吐收益。
- **padding_side 是容易被忽略的正确性 bug**，批推理必须设 left-padding，否则生成质量静默降级。

对于 Qwen2.5-7B 这个规模的模型，当前最高效的双卡策略仍然是：单卡高效跑，Replica 做水平扩展，当请求量超过单卡最优 batch 时再引入第二卡。
