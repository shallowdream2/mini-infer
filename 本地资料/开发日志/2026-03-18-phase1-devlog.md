# Phase 1 开发日志（2026-03-18 ~ 2026-03-19）

这个文件记录 Phase 1 单卡推理链路的完整开发过程，包括做了什么、卡在哪、怎么解决的。

---

## 2026-03-18

### 做了什么

- 确认项目当前状态：骨架代码，model_runner/engine/scheduler/kv_cache 均为桩实现
- 修复 `pyproject.toml` 包发现问题（setuptools 把 `本地资料/` 识别为 Python 包）
  - 加 `[tool.setuptools.packages.find] include = ["mini_infer*"]`
- 在 ai-infra conda 环境中 `pip install -e .[dev]` 成功
- 跑通 3 个 smoke test（dry_run 模式）
- 用 `opt-125m` 在 cuda:0 上做首次真实 GPU 推理验证
- 发现 Qwen 权重不可用（未登录 HF，无本地缓存）
- 发起 Qwen2.5-7B 后台下载任务

### 卡点

- `conda activate ai-infra` 在 Claude Code 的 bash 环境里不生效（需要用绝对路径 `/home/shh/anaconda3/envs/ai-infra/bin/python`）

---

## 2026-03-19 凌晨

### 做了什么（模型下载）

- 发现 `model-00002-of-00004.safetensors` 下载进程卡死超过 5 小时
  - 进程存活，有网络连接，但 `.incomplete` 文件大小自 02:54 起不再变化
  - 原因：`snapshot_download` 的 sha256 校验在后台进程中挂起
- 确认 `.incomplete` 文件大小（3,864,726,352 bytes）等于服务器 `Content-Length`
- 直接 `cp .incomplete → model-00002-of-00004.safetensors`，跳过校验
- 所有 4 个 shard 到位

### transformers 版本兼容问题

- transformers 4.36.2 不支持 Qwen2Tokenizer → 升级
- transformers 5.x 要求 PyTorch ≥ 2.4，当前 2.1.2 崩溃
- 锁定 4.43.4：支持 Qwen2 + 兼容 PyTorch 2.1.2

### Qwen2.5-7B 真实推理验证

- 模型加载正常（4 shards，~2.7s，显存 15.7 GB）
- `engine.generate()` 输出正确
- 推理链路完整验证通过

---

## 2026-03-19 上午

### Benchmark

- 跑 `benchmark_hf.py`（batch=1/4/8，Qwen2.5-7B，128 tokens）
- 跑 `benchmark_mini.py`（同样配置）
- 关键发现：mini-infer Phase 1 吞吐不随 batch 增长（batch=4 差 3.7×，batch=8 差 7.3×）
- 原因确认：`decode_step` 是串行 for 循环，每次 1 个 forward

### 代码审查

- 发现 P1 问题：
  1. `eos_token_id or 0` 的 falsy 判断（Qwen2.5 不触发，换模型是隐患）
  2. 模型先 load CPU 再 `.to(device)`，显存峰值约 2×
  3. 异常中断时 `_past_kv` 不释放（无 try/finally）
- 发现设计问题：`enable_paged_kv_cache` 字段未被使用

### 收尾

- 写 Phase 1 里程碑总结
- 完善 CLAUDE.md 阶段工作流规则（7步 skill 顺序）
- 新增 `infer-archive` skill
- 写 Phase 1 博客草稿

---

## 关键决策记录

| 决策 | 原因 |
|------|------|
| Phase 1 不做 batch decode | 等 Phase 2 的 Paged KV Cache 接口稳定后统一改，避免重复工作 |
| transformers 锁定 4.43.4 | 5.x 要求 PyTorch 2.4，当前环境不满足 |
| `past_key_values` 存在 dict 里 | Phase 1 最简实现，Phase 2 替换为 GPU block tensor |
| 跳过 sha256 校验直接 cp | 文件大小与 Content-Length 精确一致，内容完整 |
