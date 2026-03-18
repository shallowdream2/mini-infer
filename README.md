# Mini-Infer

这个文件用于说明项目的最终目标、当前状态、Ubuntu 工作方式和后续阶段边界。

## 项目题目

基于 PagedAttention 的高性能大模型推理引擎

## 当前状态

当前仓库仍处于初级骨架阶段，正式的推理系统实现还没有开始。

目前只保留了：

- 最小代码结构
- 最小 smoke test
- 最小 benchmark 入口
- Claude Code 协作配置

当前代码不能代表已经完成的 `PagedAttention`、`Continuous Batching` 或高性能推理实现。

## 当前默认前提

- 当前协作默认假设：你已经位于 Ubuntu 24.04 项目终端内，而不是站在 Windows 侧做远程控制
- 当前开发默认直接复用 [本地资料/环境配置/AI-Infra学习之旅-服务器环境配置.md](本地资料/环境配置/AI-Infra学习之旅-服务器环境配置.md) 中已经配置完成的 `ai-infra` 环境
- 只做代码阅读、骨架开发和单元测试时，可以没有 GPU
- 真实模型推理、benchmark、多卡实验仍然需要 CUDA GPU、模型权重和对应依赖

## Ubuntu 快速开始

### 直接复用现有环境

```bash
conda activate ai-infra
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests/test_smoke.py tests/test_scheduler.py tests/test_kv_cache.py
```

### 真实 GPU 运行与 benchmark

```bash
conda activate ai-infra
nvidia-smi
python benchmarks/benchmark_hf.py --model Qwen/Qwen2.5-7B-Instruct --batch-size 1 --max-new-tokens 32
```

- 如果当前环境里缺少项目依赖，再直接安装到现有 `ai-infra` 环境，不默认新建环境
- 如果当前环境真的损坏，再回到环境文章中的对应步骤修复，而不是先新建一个并行环境
- benchmark 前确认模型权重可访问，例如已完成 HuggingFace 登录或本地已有权重
- 没有 Ubuntu + CUDA 实测数据时，不写性能结论

## 最终目标

在单机双 `RTX 4090` 环境下，面向 `Qwen2.5` 这一类 `decoder-only` 大模型，实现一个基于 `PagedAttention` 的高性能推理引擎。该引擎支持块化 `KV Cache`、`Block Table` 映射、`Prefill / Decode` 分离、`Continuous Batching` 和流式生成，并在真实 benchmark 中相较 `HuggingFace Transformers` baseline 展现更好的吞吐、显存利用率和并发承载能力。

## 验收标准

- 能稳定跑通单卡 `Qwen2.5-7B`
- 支持多请求和不同长度 prompt 的并发生成
- `KV Cache` 采用分页块管理，而不是简单连续缓存
- 具备 `Prefill / Decode` 两条执行路径
- 具备 `Continuous Batching` 调度器
- 提供可复现 benchmark，至少覆盖 `throughput`、`TTFT`、`TPOT`、`peak memory`
- 相较 `HuggingFace Transformers` baseline，在吞吐、显存峰值、并发承载能力中至少两项有明显收益

## 开发与运行环境

### 主开发环境

- Ubuntu 24.04 LTS
- bash
- Python 3.10+

### 目标运行环境

- Ubuntu 24.04 LTS
- 2 × NVIDIA GeForce RTX 4090

### 约束

- 当前默认已经在 Ubuntu 项目环境内工作，不再以 Windows Remote SSH 为前提
- 真实模型运行、性能测试和多卡实验默认在 Ubuntu + CUDA 环境进行
- 没有 Ubuntu GPU 实测数据时，不写性能结论

## 阶段目标

### 第一阶段

- 完成单卡最小推理链路
- 跑通真实模型加载和 `generate()`
- 建立基础 benchmark

### 第二阶段

- 实现 `Paged KV Cache`
- 实现 `Prefill / Decode` 分离
- 实现 `Continuous Batching`

### 第三阶段

- 做单机双卡扩展
- 优先考虑双卡 `replica` 提升总吞吐
- 评估是否继续做 `TP=2`

## 当前目录

```text
mini_infer/   核心代码骨架
benchmarks/   benchmark 骨架
tests/        最小测试骨架
.claude/      Claude Code 规则、skills 和项目设置
CLAUDE.md     Claude Code 项目级协作说明
本地资料/     个人记录与知识整理，不上传 Git
```
