# Phase 4 环境配置补充说明

这个文件记录 Phase 4（双卡扩展）过程中发现的新环境约束和配置问题，补充到现有环境配置之外。

## HF 模型缓存结构问题

### 现象

Phase 4 benchmark 首次运行时，使用 `Qwen/Qwen2.5-7B-Instruct` Hub ID 加载模型，进程启动后 GPU 显存未增长，进程挂起，实际在触发 HF 重新下载。

### 根因

HF cache 的 snapshot 目录（`snapshots/a09a35458.../`）只有 shard 1 的软链接，shard 2-4 的 blobs 文件存在但未链接。`from_pretrained` 使用 Hub ID 时会联网检查版本，发现 snapshot 不完整后走 xet 协议重下载（限速约 1 MB/s，全量需 4-5 小时）。

### 解决方式

```bash
# 推荐：用本地根目录路径 + offline 模式
export MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1
python benchmarks/benchmark_multi_gpu.py --model $MODEL ...
```

根目录（非 snapshot 子目录）有完整文件：config.json、tokenizer 文件、4 个 safetensors shard。

### 当前状态

- 根目录：完整，4 个 shard 均可用
- snapshot 子目录：shard 1 软链接正常，shard 2-4 缺少软链接（blobs 存在但未链接）
- **不需要重新下载**，直接用根目录路径即可

## accelerate 多卡 P2P 警告

```
We've detected an older driver with an RTX 4000 series GPU. These drivers have issues with P2P.
This can affect the multi-gpu inference when using accelerate device_map.
```

当前驱动版本偏旧，accelerate 在 device_map 多卡推理时发出此警告。实测 tp2（device_map="balanced"）功能正常，但性能可能受影响。建议升级驱动后重测多卡通信密集场景。

## TPEngine 右填充问题（已修复）

**现象**：tp2 模式运行时输出 `right-padding was detected` 警告。

**原因**：decoder-only 模型批推理应使用左填充，右填充会导致新 token 的 position_id 计算错误，生成质量下降。

**修复**：`tp_engine.py` 的 tokenizer 初始化加入 `padding_side="left"`（已修复于 Phase 4 infer-implement 修复轮）。

## 双卡 benchmark 推荐运行方式

```bash
cd ~/projects/mini-infer
conda activate ai-infra
export MODEL=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
export HF_HUB_OFFLINE=1

# 单卡基线
python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode single --batch-size 8 --max-new-tokens 128

# 双卡数据并行
python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode replica --batch-size 8 --max-new-tokens 128

# 双卡 HF Pipeline Parallel
python benchmarks/benchmark_multi_gpu.py --model $MODEL --mode tp2 --batch-size 8 --max-new-tokens 128
```
