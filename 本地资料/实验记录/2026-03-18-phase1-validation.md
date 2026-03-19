# Phase 1 推理链路验证与 Baseline Benchmark

这个文件记录 Phase 1 单卡推理链路首次 GPU 验证结果，以及与 HuggingFace baseline 的初步对比数据。

## 环境

- 日期：2026-03-18
- 系统：Ubuntu 24.04
- GPU：2 × RTX 4090（验证在 cuda:0 单卡进行）
- CUDA：12.2
- Python：3.10.19（ai-infra conda env）
- PyTorch：2.1.2+cu121
- transformers：4.36.2
- 模型：`facebook/opt-125m`（Qwen 权重暂不可用，用小模型验证链路正确性）

## 环境问题修复

`pyproject.toml` 缺少 `[tool.setuptools.packages.find]` 配置，导致 setuptools 把 `本地资料/` 误识别为 Python 包，安装报错。已添加：

```toml
[tool.setuptools.packages.find]
include = ["mini_infer*"]
```

## Smoke Test 结果

```
tests/test_smoke.py::test_engine_generate_smoke PASSED
tests/test_scheduler.py::test_scheduler_batching PASSED
tests/test_kv_cache.py::test_kv_cache_block_growth PASSED
3 passed in 0.66s
```

## 真实推理链路验证

用 `opt-125m` 在 `cuda:0` 上验证完整 generate 路径：

```
加载模型 → 显存 0.26 GB → 生成 2 条输出 → 推理链路 OK
```

输出示例合理，decode 采样正常。

## Benchmark 对比（opt-125m, batch=4, max_new_tokens=128, cuda:0, float16）

| 指标 | HF Baseline | mini-infer Phase 1 |
|------|------------|-------------------|
| Throughput | 1753 tokens/s | 463 tokens/s |
| Peak Memory | 0.32 GB | 0.30 GB |
| TTFT | 3.4 ms | 未测量 |
| TPOT | 2.27 ms/token | 未测量 |
| 总耗时 | ~0.30 s | 1.10 s |

**差距说明：mini-infer Phase 1 比 HF baseline 慢约 3.8x。**

原因分析：
- HF baseline：4 条 prompt 一次性 pad 到同一长度，整批做 prefill + decode
- mini-infer Phase 1：4 条请求**逐条串行** prefill，decode 阶段每条独立执行，无真实 batch 并行
- 当前 `decode_step` 是 `for state in states: model(single_token)` 的串行结构

这是 Phase 1 的预期设计，Phase 2 的 Continuous Batching 将合并 decode step 解决这个问题。

## 待改进

1. `benchmark_mini.py` 缺少 TTFT / TPOT 分离计时（当前只有总 throughput）
2. Qwen 模型权重不可用，后续需要 HF 登录或本地下载

## 结论

Phase 1 链路验证通过。真实 GPU 推理可用。可以开始 Phase 2 的 Paged KV Cache 设计。

---

## 2026-03-19 补充：Qwen2.5-7B 环境修复

### 问题与修复

1. **transformers 版本过旧**：4.36.2 不支持 Qwen2Tokenizer，升级到 4.43.4 解决。
   - 注意：transformers 5.x 要求 PyTorch >= 2.4，当前 2.1.2 不兼容，上限为 4.x。

2. **模型权重下载中断**：`snapshot_download` 的 sha256 校验在 3.6GB 文件上卡死超过 5 小时。
   - 原因：huggingface_hub 校验逻辑在某些版本和网络条件下会挂起。
   - 解法：确认 `.incomplete` 文件大小与服务器 `Content-Length` 一致后，直接 `cp` 重命名，跳过校验。

3. **模型路径**：权重存放在 `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/`（非标准 blobs 结构，用 `local_dir` 参数下载）。

### Qwen2.5-7B 验证结果

```
加载 4 个 shard 耗时：~2.7s
显存占用：15.7 GB（单张 RTX 4090，float16）
生成测试：通过，回答 PagedAttention 问题输出正确
```

### 当前可用状态

- transformers：4.43.4
- 模型：Qwen/Qwen2.5-7B-Instruct（local，float16，cuda:0）
- 可以运行 `benchmark_hf.py` 和 `benchmark_mini.py`
