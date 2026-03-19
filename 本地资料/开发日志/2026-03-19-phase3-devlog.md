# Phase 3 开发日志（2026-03-19）

本文件记录 mini-infer Phase 3 性能优化阶段的开发过程。

---

## 任务背景

Phase 2 benchmark 发现 batch=8 throughput 只有 HF baseline 的 49.1%，分析后定位到两个问题：

1. `gather_batch_kv()` 内部有 3 层 Python 嵌套循环（layers × batch × seq_len），Python 解释器开销随 batch size 线性放大
2. `past_key_values` 以 tuple 传入 transformers 时持续触发 UserWarning

Phase 3 任务：向量化 gather、迁移 DynamicCache、新增混合长度 benchmark。

---

## 过程记录

### 1. 向量化 gather_batch_kv

思路：把 BlockTable 从 Python dict 变成 `[batch, max_num_blocks]` 的 GPU tensor，然后通过 advanced indexing 一次性 gather。

核心数学：左填充对齐的 token 位置 = `output_position - (max_seq_len - seq_len)`，负值是填充区。填充区 clamp 到 0 后 gather 出来的值用 `valid_mask_f=0` 乘掉。

实现中注意到 `valid_mask` 是 bool tensor，直接乘 float16 KV 会有隐式类型转换，加了 `.to(dtype=cache_dtype)` 显式转换。

为了保证向量化逻辑正确，在 CPU 上写了 `test_gather_batch_kv_correctness()`，用已知数值验证了 2 个不同长度请求的 gather 结果，包括 padding 置零。

### 2. DynamicCache 迁移

先改了 `decode_batch()`，把 tuple 替换成 `DynamicCache` + `cache.update(k, v, l)`，从 `out.past_key_values.key_cache[l]` 提取新 KV。

改完跑测试，warning 还在。加 `-W error::UserWarning` 定位，发现 traceback 指向 `prefill()` 而不是 `decode_batch()`：

```
File ".../model_runner.py", line 121, in prefill
    out = self.model(input_ids=input_ids, use_cache=True)
```

原因：transformers 4.43.4 在 `past_key_values=None` 时内部生成 tuple 并发出 warning，不是调用方传了 tuple。修复：在 prefill 里传 `past_key_values=DynamicCache()`，让模型走新路径，warning 消失。

### 3. infer-review 发现的问题

review 阶段发现 `benchmark_mini.py` 里有几处遗漏：
- `engine_phase` 仍写 `"mini-infer-phase2"`
- 打印文本仍写 "Phase 2"
- `MixedBenchmarkResult` 字段名 `short_req_count`/`long_req_count` 语义不准确（实际所有请求统一 max_new_tokens=256，不是短请求真的少跑了）

全部修正：重命名字段为 `short_prompt_count`/`long_prompt_count`，新增 `actual_max_new_tokens`，打印文本明确说明统一值。

### 4. benchmark 结果

GPU 0 (RTX 4090) 上实测：

| batch | Phase 2 | Phase 3 | HF |
|-------|---------|---------|-----|
| 1 | 49.4 | 53.7 | 56.2 tok/s |
| 4 | 135.2 | 194.2 | 210.5 tok/s |
| 8 | 201.0 | 361.3 | 408.9 tok/s |

Mixed (8 req, max_tokens=256)：356.0 tok/s，Peak 17.00 GB。

Phase 3 batch=8 从 49.1% → 88.4% HF，基本追平。

---

## 关键决策

- `valid_mask` 显式转 float 而非依赖隐式转换
- DynamicCache 空实例在 prefill 必须传，不传就触发 warning
- `gather_batch_kv` 仍有 num_layers 层 Python 循环，但层数固定（28），不随 batch/seq_len 扩大，可接受
- 残余 10~12% 差距来自 gather 本身的 KV 复制，需 flash_attn 2.5+ 才能彻底消除，留 Phase 4

---

## 残余问题

1. gather 每步仍有 KV 复制（dense tensor），需 flash_attn 2.5+ 或 Triton kernel
2. `generate()` 不支持 per-request `max_new_tokens`，mixed benchmark 语义不完整
3. TPOT 是 amortized 值，缺 p50/p95 单请求延迟数据
