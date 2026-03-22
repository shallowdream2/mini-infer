# Phase 12 开发日志：CUDA Graph

日期：2026-03-22

---

## 做了什么

Phase 12 目标是给 decode_batch 接入 CUDA Graph，绕过 Python→CUDA dispatcher 的每步调度开销。

实现分四步：
1. `config.py` 加 `use_cuda_graph: bool = False` 字段
2. `model_runner.py` 加 `warmup_cuda_graphs()`、`_graph_decode_forward()`、`_eager_decode_forward()`，重构 `decode_batch()` 分路逻辑
3. `engine.py` 在 `__init__` 末尾触发 warmup
4. 新建 `benchmarks/benchmark_cuda_graph.py` 和 `tests/test_cuda_graph.py`

## 关键决策

**max_kv_len 固定为 config.max_model_len**：这是最重要的一个决策。RoPE 用 `rotary_emb(v, seq_len=max_kv_len)` 生成 cos/sin 表，如果 max_kv_len 每步变化，形状就变，graph 就失效。固定为 max_model_len 后，cos/sin 形状恒定，position_ids 通过 copy_() 更新，正确索引到对应行。

**Graph Pool**：为 bs={1,2,4,8} 各捕一张图，actual_bs pad 到 padded_bs，输出取前 actual_bs 行。

**静态 buffer + copy_()**：block_table、input_ids、position_ids、cache_seqlens 全部用静态 GPU tensor，每步只做 copy_()，不分配新 tensor。

## 卡在哪里

没有卡点，流程基本顺利。但 review 发现了两个问题：
- `_graph_decode_forward` 缺少 `try/finally`
- 每步重新 `torch.zeros(...)` 分配临时 block_table staging buffer

两个问题都在 benchmark 前修复。

## 意外发现

1.5B 和 7B 的加速比差异（28% vs 5%）并不是预期内的"量化"结论，而是在 profiler 数据出来后才清楚：Python dispatch 在 1.5B 上占 29%，在 7B 上只占 5%，所以收益本质上受限于"dispatch 占比"，不是"模型大小"。这个认识对后续调优有指导价值。

## benchmark 结果摘要

- 1.5B bs=1: eager 7.33ms → graph 5.21ms (**+28.9%**)
- 1.5B bs=8: eager 8.31ms → graph 6.79ms (**+18.3%**)
- 7B bs=8:   eager 21.97ms → graph 21.01ms (**+4.4%**)
- CPU dispatch overhead：-51.8%（20步总计 257.5ms → 124.1ms）
