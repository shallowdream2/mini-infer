# Phase 6 环境配置记录（2026-03-20）

这个文件记录 mini-infer Phase 6（True PagedAttention）的关键环境变更，包括 flash_attn 升级和相关约束。

---

## flash_attn 升级记录

### 升级前状态

```
flash_attn 2.3.6（Phase 1-5 使用的版本）
```

### 升级命令

```bash
pip install "flash-attn==2.5.9.post1" --no-build-isolation
```

### 升级后验证

```bash
# 验证 block_table 参数可用
python -c "
from flash_attn import flash_attn_with_kvcache
import inspect
assert 'block_table' in inspect.signature(flash_attn_with_kvcache).parameters
print('OK')
"
# 输出：OK
```

### 当前版本（2026-03-20）

| 包 | 版本 |
|----|------|
| flash_attn | **2.5.9.post1** |
| PyTorch | 2.1.2+cu121 |
| transformers | 4.43.4 |
| Python | 3.10 |
| CUDA | 12.1 |

---

## flash_attn 2.5+ 的关键约束（Phase 6 发现）

### block_size 必须是 256 的倍数

flash_attn_with_kvcache 内核要求：
```
k_cache.shape[1] % 256 == 0
```

否则运行时报错：
```
RuntimeError: Paged KV cache block size must be divisible by 256
```

**影响**：`EngineConfig.block_size` 默认值从 16 改为 256。

### block_size=256 的显存代价

改变 block_size 后需同步调整 num_gpu_blocks，防止 OOM：

| block_size | num_gpu_blocks | KV cache 显存（Qwen2.5-7B）| 备注 |
|-----------|----------------|--------------------------|------|
| 16 | 2048 | 1.79 GB | Phase 1-5 配置 |
| 256 | 2048 | 28.67 GB | **OOM**（RTX 4090 24 GB）|
| 256 | 200 | **2.93 GB** | Phase 6 profiling 配置 |
| 256 | 512 | 7.52 GB | 适合生产场景 |

计算公式：
```
显存 = num_blocks × block_size × num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
     = 200 × 256 × 28 × 2 × 4 × 128 × 2 ≈ 2.93 GB
```

### block_table 写入是 in-place 操作

`flash_attn_with_kvcache` 调用后，`k_cache` 和 `v_cache` 被 in-place 修改（新 token KV 已写入）。调用完成后无需手动 write_kv 步骤，但需调用 `kv_cache.advance_seq_lens()` 递增逻辑序列长度计数器。

---

## 回退方案

如需回退到 flash_attn 2.3.6（Phase 1-5 兼容版本）：
```bash
pip install "flash-attn==2.3.6" --no-build-isolation
```

注意：回退后 block_size 需改回 16，EngineConfig 默认值需手动调整。

---

## 已验证的兼容性

- PyTorch 2.1.2+cu121 + flash_attn 2.5.9.post1：**兼容**（--no-build-isolation 成功）
- 现有单元测试（pytest tests/）：**全部通过**（dry_run 路径 block_size 固定传参，不受默认值影响）
- Qwen2.5-7B-Instruct float16 + flash_attn_with_kvcache：**正常工作**（输出与 HF greedy 逐 token 一致）
