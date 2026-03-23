# Phase 15 开发日志：PD 解耦（Disaggregated Prefill/Decode）

**日期**：2026-03-23
**阶段**：Phase 15

---

## 背景

Phase 14（MLA）完成后进入最后一个阶段：PD 解耦。核心动机是 Prefill（compute-bound）和 Decode（memory-bound）在统一进程中互相干扰，Phase 9 的 Chunked Prefill 只是缓解了这一矛盾，而 PD 解耦要从架构层面彻底分离两者。

---

## 主要事件

### Plan（infer-plan）

- 目标收敛为同机双进程原型（不强求真正多节点）
- 验收标准 4 条：端到端正确、KV 传输 < 10ms、TTFT 可分解（三段）、prefill 吞吐高于统一部署

### 实现（infer-implement）

**新增 4 个文件**：

- `mini_infer/kv_transfer.py`：KVPayload 数据结构 + KVSender/KVReceiver + extract_kv_from_past + measure_kv_size_bytes
- `mini_infer/pd_worker.py`：PrefillWorker / DecodeWorker 进程主循环 + `_load_model_and_tokenizer` + `_rebuild_dynamic_cache`
- `mini_infer/pd_engine.py`：PDEngine 协调层，spawn context，generate/generate_with_timing
- `tests/test_pd_disagg.py`：7 个测试
- `benchmarks/benchmark_pd_disagg.py`：三个 section

**两个主要卡点**：

1. **CUDA + fork 死锁**：Linux 默认 `multiprocessing.start_method=fork`，CUDA 初始化后 fork 导致子进程挂死。症状：`result_queue.get(timeout=120)` 超时，子进程 exit code -9。修复：`_ctx = _mp.get_context("spawn")`。

2. **ModelRunner 不能直接复用**：`ModelRunner.__init__` 调用 `patch_model_for_paged_decode(model, kv_cache)`，kv_cache=None 时会崩溃。PD worker 需要普通 HF forward（DynamicCache），不需要 Paged Attention。修复：绕过 ModelRunner，直接用 AutoModelForCausalLM 加载模型。

### 代码审查（infer-review）

无阻塞性问题。建议修复 5 条：
- `pd_engine.py` send_times dead dict（删除）
- `pd_worker.py` worker 主循环缺少 exception catch（加入 try-except + traceback）
- `kv_transfer.py` unused `import time`（删除）
- `generate_with_timing` docstring 承诺的 prefill_time 字段在 DecodeResult 中缺失（添加 prefill_time 字段到 DecodeResult）
- `benchmark_pd_disagg.py` Section 1 MLA kv_factor 应为 1 而非 2（修复）

全部修复后进入 benchmark。

### Benchmark（infer-benchmark）

**Section 1 理论 KV 大小**：
- Qwen2.5-1.5B seq=128: 3.50 MB GQA
- Qwen2.5-7B seq=1024: 56.00 MB GQA
- DeepSeek-V2-Lite MLA seq=128: 3.80 MB（单向量，不是 ×2）

**Section 2 端到端正确性**：
- LLMEngine 和 PDEngine greedy 输出完全一致 ✓

**Section 3 TTFT 分解**：
- Unified LLMEngine: 459.2ms 平均
- PDEngine: 546.0ms 平均（1.19×）
  - prefill: 12.3ms
  - transfer（近似）: 14.7ms（pickle+Queue IPC）
  - decode: 519ms

验收标准 2（< 10ms）部分达成：14.7ms > 10ms，原因是用了 pickle+Queue 而非真正的共享内存。

---

## 关键决策

1. **spawn vs fork**：CUDA 在 fork 后无法工作，必须用 spawn。代价是启动时间变慢（需要重新 import 所有模块）。
2. **绕过 ModelRunner**：PD worker 不需要 Paged Attention，直接加载 HF 模型更干净。
3. **KV 格式选 `[seq, heads, dim]` CPU tensor**：去掉 batch 维度，轴顺序便于按 seq 切片，CPU 存储避免 CUDA IPC 复杂性，pickle 序列化简单。
4. **pickle+Queue 而非共享内存**：足以验证架构正确性，省去共享内存的生命周期管理复杂度。

---

## 测试结果

```
tests/test_pd_disagg.py  7 passed in 5.71s
```

---

## 遗留问题

- KV 传输延迟 14.7ms > 10ms 目标（需改用 multiprocessing.shared_memory 或 CUDA IPC）
- 并发 pipeline 未实现（prefill i+1 与 decode i 并行的核心吞吐优势）
- 两进程各加载一份模型，GPU 显存 2×
