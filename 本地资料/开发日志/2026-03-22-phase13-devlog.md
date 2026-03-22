# Phase 13 开发日志：Tensor Parallelism

**日期**：2026-03-22
**阶段**：Phase 13 — Tensor Parallelism（真 TP，NCCL all-reduce）

---

## 主要工作

### 1. 确认旧 TPEngine 是 PP 别名

发现 `mini_infer/tp_engine.py` 只有一行：
```python
from .pp_engine import PPEngine as TPEngine
```
这是 Phase 4 遗留的。Phase 13 的目标是实现真正的 Megatron-LM 风格 TP。

### 2. 实现 tp_model_runner.py

新建 `mini_infer/tp_model_runner.py`，包含：
- `col_shard(weight, rank, tp_size)` — dim=0 切分（column parallel）
- `row_shard(weight, rank, tp_size)` — dim=1 切分（row parallel）
- `_shard_qwen2_weights(model, rank, tp_size)` — 遍历所有 decoder layers 就地替换权重，更新 attn 属性
- `_register_tp_allreduce_hooks(model)` — 在 self_attn 和 mlp 注册 forward hook
- `TensorParallelModelRunner` — 主类，封装模型加载 + 权重切分 + generate

关键发现：`attn.hidden_size` 必须同步更新为 `num_heads_per_rank × head_dim`，否则 Qwen2.5 forward 里的 `attn_output.reshape(bsz, q_len, self.hidden_size)` 会报 shape mismatch。

### 3. 重写 tp_engine.py

改为真 TP 引擎：`mp.spawn` 启动 tp_size 个 worker 进程，文件锁 rendezvous（`file:// init_method`），rank 0 把结果写 JSON，主进程读取后返回。

### 4. 写测试

`tests/test_tp_engine.py` — 13 个 dry_run 测试，全部通过。不需要 GPU/NCCL。

### 5. 写 benchmark

`benchmarks/benchmark_tp.py` — 支持 single/pp/tp/torchrun_tp 四种模式。

---

## 踩坑记录

### 坑 1：eager 模式输出乱码
第一次跑 TP=2 greedy，3 个 prompt 输出全是 `!!!...`。单卡 eager 模式也复现。
原因：transformers 4.43.4 的 Qwen2Attention eager 模式有 attention mask bug。
解决：改为 `attn_implementation="flash_attention_2"`，输出恢复正常。

### 坑 2：TP=2 与单卡对比方法错误
最初用 token ID 逐一比对（encode 解码后的文字再比），发现第 2 条 prompt 只有 0% 匹配。
原因：decode 后再 encode 因分词边界（`\n`、引号等）产生不同 token ID，不是 TP 错误。
解决：改为文字级别比对，3/3 prompts 100% 匹配。

### 坑 3：benchmark 的 _bench_pp() 是死代码
`_bench_pp()` 调用 `PPEngine(config=None, ...)` 但 PPEngine 构造函数只接受 `config: EngineConfig`。
这个函数在 main() 里也从未被调用（pp 走的是 _bench_pp_direct()）。
处理：review 阶段标为阻塞性问题，implement 阶段删除。

### 坑 4：--mode tp 计时含模型加载
mp.spawn 每次 generate() 都重新启动进程和加载模型，计时无效。
处理：将 tp 模式从 --mode all 排除，加警告提示；真实吞吐用 --mode torchrun_tp。

### 坑 5：torchrun 下 CUDA 初始化顺序
`torch.cuda.reset_peak_memory_stats(device)` 在 dist.init_process_group 后立即调用报错：
```
RuntimeError: Invalid device argument 0: did you call init?
```
解决：加 `torch.cuda.set_device(rank)` 在前面。

---

## 最终结果

- 13/13 测试通过
- TP=2 greedy 输出与单卡一致（3/3 prompts 100% 文本匹配）
- 吞吐：single 98.0 → pp 82.4 → tp=2 76.5 tok/s（符合预期，小模型 decode memory-bound）
- VRAM：load-then-shard 导致 peak ≈ 100%（未达理想 50%）

---

## 未完成项（可接受的限制）

- 7B 模型未验证（软链接缺失，降级为 1.5B）
- all-reduce 开销未单独量化
- shard-during-load（VRAM 减半）未实现
