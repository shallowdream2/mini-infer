# 本地模型参考手册

> 本文件记录 mini-infer 项目中已下载的本地模型路径、架构参数和使用方式。
> 最后更新：2026-03-22

---

## 已下载模型总览

| 模型 | VRAM | 用途 |
|------|------|------|
| Qwen2.5-0.5B-Instruct | ~1.9 GB | spec draft 模型；快速 dev/test |
| Qwen2.5-1.5B-Instruct | ~3 GB | Phase 12+ 主要开发调试用 |
| Qwen2.5-7B-Instruct | ~18.2 GB | 最终 benchmark 验证 |

---

## 路径与架构参数

### Qwen2.5-0.5B-Instruct

```
路径：~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775
```

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 24 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 |
| head_dim | 64 |
| vocab_size | 151936 |
| 文件大小 | ~1.9 GB（float16） |

EngineConfig 示例：
```python
cfg = EngineConfig(
    model_path="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
    num_hidden_layers=24,
    num_kv_heads=2,
    head_dim=64,
)
```

---

### Qwen2.5-1.5B-Instruct

```
路径：~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306
```

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 28 |
| num_attention_heads | 12 |
| num_key_value_heads | 2 |
| head_dim | 128 |
| vocab_size | 151936 |
| 文件大小 | ~3 GB（float16） |

EngineConfig 示例：
```python
cfg = EngineConfig(
    model_path="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    num_hidden_layers=28,
    num_kv_heads=2,
    head_dim=128,
)
```

**适用场景**：
- Phase 12 CUDA Graph 开发调试（比 7B 快 ~5×，比 0.5B 更接近 7B 行为）
- 与 0.5B 做 speculative decoding（同 vocab_size=151936，不需要 vocab 对齐）

---

### Qwen2.5-7B-Instruct

```
路径：~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
（使用根目录，NOT snapshot 子目录）
```

| 参数 | 值 |
|------|-----|
| num_hidden_layers | 28 |
| num_attention_heads | 28 |
| num_key_value_heads | 4 |
| head_dim | 128 |
| vocab_size | 152064 |
| 文件大小 | ~14.3 GB（float16，4 个 shard） |

EngineConfig 示例：
```python
cfg = EngineConfig(
    model_path="~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",
    # num_hidden_layers/num_kv_heads/head_dim 可用 7B 默认值
)
```

**注意**：snapshot 子目录不可用。
`snapshots/a09a35458c702b33eeacc393d103063234e8bc28/` 下 shard 2-4 软链接缺失，
transformers 检测到后会触发重下载（约 4-5 小时）。**务必用根目录路径**。

---

## 通用使用规则

### 1. 必须设置 HF_HUB_OFFLINE=1

所有本地模型运行前必须设置：

```bash
export HF_HUB_OFFLINE=1
```

或在 Python 中：

```python
import os
os.environ["HF_HUB_OFFLINE"] = "1"
```

原因：transformers 默认联网检查版本更新。7B 的 snapshot 子目录不完整，
检测到后会触发 xet 协议重下载（~1 MB/s）。OFFLINE=1 强制只用本地缓存。

### 2. EngineConfig 必须显式指定架构参数

mini-infer 的 EngineConfig 默认值是 7B 的参数（28层/4KV heads/head_dim=128）。
使用 0.5B 或 1.5B 时必须明确传入对应参数，否则 KV cache 块大小计算错误。

### 3. Spec Decoding 跨模型注意事项

- 0.5B (vocab=151936) + 7B (vocab=152064)：vocab_size 不同，rejection sampling 前需 zero-padding 对齐
- 0.5B (vocab=151936) + 1.5B (vocab=151936)：vocab_size 相同，不需要额外处理
- 两个模型需分配在不同设备（cuda:0 / cuda:1）时，rejection sampling 前需 `.to(target_device)`

---

## 环境变量速查

```bash
# 7B（benchmark）
export MODEL_7B=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct

# 1.5B（Phase 12+ 开发）
export MODEL_1B5=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306

# 0.5B（spec draft / dev）
export MODEL_0B5=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775

# 必须
export HF_HUB_OFFLINE=1
```
