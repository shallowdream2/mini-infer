# Phase 4 知识笔记：Replica 数据并行与 Pipeline Parallel

本文件是 mini-infer Phase 4 的核心概念速记，供自己回顾。

---

## Replica（数据并行）要点

- 每张 GPU 跑一个完整模型，请求 round-robin 分发，结果合并
- 显存翻倍（每卡一份权重），吞吐理论 2×，单请求延迟不变
- Python 线程 + CUDA 异步：GIL 序列化 Python 代码，但两个 GPU 在硬件上真正并发
- **batch 大小决定 Replica 的实际收益**：如果每卡 batch=N 的效率接近单卡 batch=2N，则 Replica 几乎无增益

实测：单卡 batch=8 = 361 tok/s，Replica batch=4+4 = 376 tok/s（+4.1%）。因为单卡 batch=4 = 194 tok/s，2×194 = 388，而单卡 batch=8 已经是 361（388 的 93%）。Replica 的增量空间只有 7%。

---

## Pipeline Parallel（HF device_map="balanced"）要点

- 不是 Tensor Parallel：层是完整的，每层运行在一块 GPU 上
- 层间通信：激活张量 `[batch, seq, hidden]` 在 GPU 边界传递（PCIe 或 NVLink）
- 显存：每卡约减半（28层 → 14层/卡）
- 吞吐：与单卡相同（总计算量不变，只是分在两卡上串行跑）

**真正的 TP vs PP**：
- PP：每层完整在一卡，层间流水 → 显存减半，吞吐不变
- TP：每层按 head 切分，层内 all-reduce → 每卡计算量减半，延迟可以降低，但需要 flash_attn 2.5+ 或 Megatron-LM

---

## padding_side="left" 的重要性

decoder-only 模型批推理时：
- 右填充（默认）：PAD PAD PAD [T1 T2 T3]，生成的新 token 在 PAD 后面，position_id 不对
- 左填充（正确）：PAD PAD PAD T1 T2 T3 → 新 token 直接跟在 T3 后，position_id 正确

单请求不需要填充，问题不显现。批推理 prompt 长度不一时右填充会静默降低生成质量。

---

## HF cache 路径陷阱

- HF Hub ID（`Qwen/Qwen2.5-7B-Instruct`）每次加载都联网检查版本
- snapshot 子目录软链接不完整 → 联网发现缺失 → 触发重下载
- 解法：`HF_HUB_OFFLINE=1` + 根目录绝对路径（根目录有完整文件，不是 snapshot）

---

## Replica 何时真的有 2×

- 单卡 KV cache 装不下总 batch（OOM 场景）
- 总请求量超出单卡最优 batch（如需处理 16+ 条，单卡 batch=16 OOM，Replica 每卡 8 条）
- 对 Qwen2.5-7B 这类小模型在 batch ≤ 8 的测试规模下，单卡已经高效，Replica 增益很小
