# Phase 7 知识笔记 — Preemption + Priority Scheduling

## 核心概念

**为什么需要 preemption？**

Continuous batching 里，每个请求占用 KV cache 直到生成完毕。当 GPU KV cache 满了，新来的高优先级请求没有空间，即使 GPU 算力是空闲的。解法：把低优先级请求的 KV 换出到 CPU，腾出 GPU 空间。

**swap vs recompute**

- swap：把 GPU KV tensor 拷贝到 CPU 内存，之后换回继续生成。好处：被换出的请求不需要重做 prefill。坏处：需要 CPU 内存，拷贝有延迟。
- recompute：直接丢掉 KV，被换出的请求等到资源充足后重新 prefill。好处：不需要 CPU 内存。坏处：长 prompt 的 prefill 成本高。

vLLM 默认用 swap，也支持 recompute。

## 状态机

```
waiting → running → (finished)
             ↑         ↓
          swap_in   swap_out
             ↑         ↓
          swapped ←────┘
```

三个容器：`_waiting (deque)`、`_running (dict)`、`_swapped (deque)`。

## prefilled 标志的作用

`RequestState.prefilled` 表示请求是否已完成第一次 prefill forward。

如果一个请求刚被准入（已分配 KV 块、加入 running），但还没有 prefill（KV 全是零值），此时对它调用 `swap_out` 是有害的——它的 KV 是垃圾数据，换出到 CPU 再换回来会污染续写。

正确处理：`prefilled == False` 时走 `un_admit`（撤销准入，块归还，放回 waiting 队尾）。

## PCIe 带宽陷阱

实测 GPU↔CPU swap 有效带宽约 1100 MB/s，远低于 PCIe 4.0 x16 理论峰值 32 GB/s。

原因：Python 层逐块拷贝（每次 ~16 KB），每次 `.cpu()` 都是独立的 PCIe 传输 + CUDA kernel launch 开销。小包传输下协议开销占主导。

vLLM 把所有层 KV 合并成一个大 tensor 再一次性拷贝，有效带宽能达到 ~10–20 GB/s。

## un_admit 必须放队尾

如果把 un_admit 的请求放回 waiting 队头，会导致：
1. A（高优先级）因块不够触发换出 B（低优先级）
2. B 放队头
3. 下一次准入轮次，B 又排在最前面被准入
4. 块还是不够，B 再次被换出
5. 无限循环

放队尾保证高优先级请求 A 能先完整地跑几步，消耗完 token 或释放块后，B 才再尝试准入。

## 与 vLLM 的差距

| 方面 | mini-infer Phase 7 | vLLM |
|------|-------------------|------|
| swap 带宽 | ~1100 MB/s（逐块） | ~10–20 GB/s（合并拷贝） |
| 支持 recompute | 否 | 是 |
| 优先级策略 | 简单数值比较 | 支持 FCFS + 抢占阈值 |
| 端到端 swap 测试 | dry_run 覆盖 | 完整集成测试 |
