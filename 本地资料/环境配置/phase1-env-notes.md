# Phase 1 环境配置补充说明

这个文件记录 Phase 1 过程中发现的新环境约束和配置问题，补充到现有环境配置之外。

## 已确认的版本约束

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.10.19 | ai-infra conda env |
| PyTorch | 2.1.2+cu121 | 不要升级，升级会拉高 transformers 依赖 |
| transformers | **4.43.4** | 支持 Qwen2，上限为 4.x（5.x 要求 PyTorch ≥ 2.4）|
| CUDA | 12.2 | 驱动 535.288.01 |

## transformers 版本锁定说明

- transformers < 4.40.0：不支持 Qwen2Tokenizer，报 `ValueError: Tokenizer class Qwen2Tokenizer does not exist`
- transformers ≥ 5.0.0：使用了 `torch.utils._pytree.register_pytree_node`，PyTorch 2.1.x 不存在该 API，导致 `import transformers` 直接崩溃
- **结论：在当前 PyTorch 2.1.2 环境下，transformers 必须 ≥ 4.40.0 且 < 5.0.0，推荐 4.43.4**

更新 `pyproject.toml` 中的依赖约束：
```toml
dependencies = [
    "torch>=2.1.2",
    "transformers>=4.40.0,<5.0.0",
]
```

## Qwen2.5-7B 模型路径

- 权重位置：`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/`
- 下载方式：`snapshot_download` + `local_dir` 参数（非标准 blobs 结构）
- 显存：float16 约 15.7 GB，单张 RTX 4090 可用

## 已知问题：snapshot_download sha256 校验挂死

**现象**：下载大文件（3+ GB）时，`.incomplete` 文件达到完整大小后，进程不再写盘，sha256 校验步骤挂起，可能持续数小时。

**判断是否可以跳过**：
```bash
# 1. 获取服务器文件大小
curl -sI -L "https://huggingface.co/<model>/<file>" | grep content-length

# 2. 检查 .incomplete 文件大小
stat --printf="%s\n" <path>.incomplete

# 3. 两个数字完全一致时，可以直接重命名
cp <path>.incomplete <target_path>
```

**根本原因**：huggingface_hub 某些版本在非 tty 进程（如后台任务）中计算 hash 时会挂起，已知 bug。

## mini-infer 安装

```bash
# 必须在 ai-infra 环境中
/home/shh/anaconda3/envs/ai-infra/bin/pip install -e ".[dev]"
```

注意：项目根目录有 `本地资料/` 中文目录，setuptools 会误识别为 Python 包，已通过 `pyproject.toml` 的 `packages.find` 解决，无需手动处理。
