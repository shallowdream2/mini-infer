# Phase 6 开发日志（2026-03-20）

这个文件记录 mini-infer Phase 6（True PagedAttention）的开发过程，包含关键决策、卡点和解决方案。

---

## 背景

Phase 5 profiling 确认 batch=8 decode 路径：model_forward 占 97%，gather 占 1.7%，write_kv 占 1.1%。整体 88.4% HF baseline，还差 12%。Phase 6 目标：用 flash_attn_with_kvcache 的 block_table 接口替换 gather→DynamicCache→write_kv 三段。

---

## 前置条件（Step 0）

flash_attn 2.5.9.post1 已提前编译安装，验证 block_table 参数可用：

```bash
python -c "from flash_attn import flash_attn_with_kvcache; import inspect; assert 'block_table' in inspect.signature(flash_attn_with_kvcache).parameters; print('OK')"
# OK
```

---

## 实现过程

### Step 1：kv_cache.py 新增方法

新增三个方法支持 Phase 6 的 block table 管理：
- `ensure_next_slot(request_ids)`：确保下一个 decode 位置已有物理块
- `build_block_tables(request_ids)`：构造 `block_table [batch, max_blocks] int32` 和 `cache_seqlens [batch] int32`
- `advance_seq_lens(request_ids)`：flash_attn in-place 写入后递增 seq_len

验证：`test_build_block_tables`、`test_ensure_next_slot_allocates_new_block`、`test_advance_seq_lens` 三个单元测试通过。

### Step 2：config.py block_size 修改

flash_attn_with_kvcache 要求 `block_size % 256 == 0`，否则运行时报错：
```
RuntimeError: Paged KV cache block size must be divisible by 256
```
将 `EngineConfig.block_size` 默认值从 16 改为 256。

### Step 3：attention.py 新建

核心模块，包含三个部分：
1. `PagedDecodeContext`：每 decode step 的共享状态容器，存放 block_table / cache_seqlens / max_kv_len
2. `paged_decode_attention()`：封装 flash_attn_with_kvcache
3. `patch_model_for_paged_decode(model, kv_manager)`：永久 patch 28 层 attention，decode 路径走 flash_attn，prefill 路径回退到原始 HF forward

patch 的路由逻辑：`ctx.block_table is None` → prefill（原始 HF forward）；非 None → decode（paged attention）。

RoPE 处理：用 Qwen2 原生的 `rotary_emb` 模块手动 apply，不用 flash_attn 内置 `rotary_cos/sin` 参数（格式不兼容风险）。

### Step 4：model_runner.py 修改

- `__init__`：调用 `patch_model_for_paged_decode()`，获得 `self._paged_ctx`
- `decode_batch()`：替换原来的 gather→DynamicCache→model_forward→write_kv 为：
  ```
  ensure_next_slot → build_block_tables → paged_ctx.set → model.forward → advance_seq_lens → paged_ctx.clear
  ```
  用 `try/finally` 包裹 forward，保证异常时 paged_ctx 必被清除。

### Step 5：GPU 测试验证

`test_paged_attention_matches_hf_greedy`：Phase 6 paged decode 与 HF greedy（cuda:1）逐 token 比对，完全一致。

---

## 阻塞点

### 阻塞 1：flash_attn block_size constraint（第一次运行时发现）

运行 GPU 测试时报错 `RuntimeError: Paged KV cache block size must be divisible by 256`。

在测试文件中显式传 `block_size=256` 解决，同时修改 `EngineConfig` 默认值。

### 阻塞 2：`.item()` 28 次 CPU-GPU sync（benchmark 时发现）

**现象**：首次 benchmark 结果 3.7%（15 tok/s vs HF 405 tok/s）。完全不合理。

**诊断过程**：
1. 发现 profiler 测到 model_forward 527ms/step（Phase 5 是 17.9ms）
2. 推算：527ms ≈ 28 sync × 18ms/sync + 17ms forward
3. 定位到 `patched_forward` 内的 `int(ctx.cache_seqlens.max().item()) + 1`
4. 该行被 28 层各调用一次，每次 `.item()` 触发一次完整 CPU-GPU sync

**修复**：将 `max_kv_len` 计算移到 `decode_batch()` 中（一次 `.item()`），通过 `PagedDecodeContext.max_kv_len: int` 传入所有层。

修复后 benchmark：100.0% HF baseline。

### 阻塞 3：profile_decode.py OOM

`num_gpu_blocks=2048`（Phase 5 为 block_size=16 设置）在 block_size=256 时需要 28 GB KV cache，超出 24 GB 显存。改为 `num_gpu_blocks=200`（51200 token 容量，profiling 场景足够）。

---

## 最终结果

| 指标 | Phase 3 | Phase 6 | HF baseline |
|------|---------|---------|-------------|
| Throughput | 361.3 tok/s (88.4%) | **406.3 tok/s (100.0%)** | 406.4 tok/s |
| gather_batch_kv | 0.31ms/step | **0** | — |
| model_forward | 17.89ms/step | 17.06ms/step | — |

---

## 关键教训

**`.item()` 是隐式 CPU-GPU sync。** 在被多层各调用一次的函数里，任何 `.item()` / `.numpy()` / Python 打印 CUDA tensor 都是 O(层数) 的同步开销。28 层 × 18ms = 504ms，足以把一个正确实现变成无用的慢代码。

review 三轮都没发现这个问题，因为它是性能问题不是正确性问题，只有真实运行 profiler 才能量化。这验证了"benchmark 不只为了得数字，也为了发现隐藏 bug"。
