# Phase 7 开发日志 — Preemption + Priority Scheduling（2026-03-20）

本文件记录 Phase 7 的开发过程，包括实现步骤、卡点和关键决策。

---

## 目标

给推理引擎加入 KV cache swap（GPU↔CPU）和优先级调度。核心场景：GPU KV cache 满时，把低优先级请求换出，让高优先级请求继续运行。

---

## 实现过程

### Step 1：数据结构

首先在 `request.py` 里扩展了两个字段：
- `Request.priority: int = 0`：请求优先级（越小越高）
- `RequestState.prefilled: bool = False`：标记该请求是否已完成 prefill
- `RequestState.cpu_kv`：存 swap_out 后的 CPU KV tensor
- `RequestState.swapped_seq_len`：换出时的 seq_len，供 swap_in 恢复块分配

`prefilled` 字段后来证明是 bug 修复的关键。

### Step 2：KV cache 层 swap_out / swap_in

在 `kv_cache.py` 里新增两个方法：
- `swap_out`：逐层逐块把 GPU KV 拷贝到 CPU tensor，再调 `free_request` 归还 GPU 块
- `swap_in`：重新分配 GPU 块，把 CPU KV 写回 GPU，清除 `cpu_kv`

实现方式是 Python 层的 for 循环，每次 `.cpu()` 一个小 tensor。这样写清楚但带宽效率低（见 benchmark 结论）。

### Step 3：调度器扩展

在 `scheduler.py` 里新增：
- `_swapped: deque`：第三个队列，存换出状态的请求
- `mark_swapped`：running → swapped
- `move_swapped_to_running`：swapped → running（swap_in 后）
- `get_lowest_priority_running`：返回 running 中 priority 最大的请求
- `un_admit`：running → waiting 队尾（bug 修复后新增，见下）

### Step 4：引擎主循环

在 `engine.py` 准入循环里加入 preemption 逻辑：块不足时找最低优先级 running 请求换出，然后继续尝试准入。主循环退出条件扩展为 `has_waiting() or num_running() > 0 or has_swapped()`。

每步末尾加入 swap_in 阶段：有足够空闲块时将 swapped 请求换回。

---

## 卡点：never-prefilled crash bug

**问题发现时机**：infer-review 阶段，代码 review 时发现。

**复现路径**：

1. waiting 队列：[B（低优先级），A（高优先级）]（A 先加入但块不够）
2. 准入循环尝试接入 B，成功，B 加入 `newly_admitted` 和 `_running`
3. 继续尝试接入 A，块不够
4. victim = B（低优先级），`swap_out(B)` → `free_request(B)` 删掉 `_block_tables[B]`
5. B 还在 `newly_admitted` 里
6. prefill 阶段：`write_prefill_kv` 访问 `_block_tables[B.request_id]` → **KeyError**

**为什么没被 dry_run 测试抓到**：`write_prefill_kv` 有 `if self._dry_run: return` 早返回，完全跳过了 `_block_tables` 访问。所有测试都经过 dry_run 路径，crash 只在真实 GPU 模式下才出现。

**修复**：加 `prefilled` 分支——`victim.prefilled == False` 时走 `un_admit`（撤销准入，块归还，放回 waiting 队尾），而不是 `swap_out`。

**un_admit 放队尾的原因**：如果放队头，会导致同一个请求被反复准入、反复因块不足被驱逐，形成死循环。放队尾让当前批次的其他高优先级请求先跑，等它们消耗完 token 或释放块，被驱逐的请求才再次尝试准入。

---

## 关键决策

| 决策 | 选择 | 放弃 | 原因 |
|------|------|------|------|
| 换出策略 | swap（CPU 保存 KV） | recompute（重做 prefill） | 展示 GPU↔CPU 数据流，CPU 内存限制不是本项目瓶颈 |
| swap 实现粒度 | 逐块 Python for 循环 | 合并所有层一次拷贝 | 清楚展示原理；工程优化留给后续 |
| 优先级反转保护 | un_admit 放队尾 | appendleft 队头 | 防止循环抢占死锁 |
| swapped 恢复顺序 | FIFO | 按优先级恢复 | 简单可预测，避免换入换出震荡 |

---

## Benchmark 结论摘要

- 调度器纯元数据 swap 延迟：swap_out 0.36 µs，swap_in 0.60 µs（可忽略）
- GPU↔CPU KV 拷贝：~1100 MB/s 有效带宽（PCIe 理论 32 GB/s，差距 ~29×，原因是逐块 16 KB 小传输）
- 吞吐无回归：priority 路径 vs 无优先级路径差值 -0.0%

---

## 测试覆盖

21 个测试，覆盖 Scheduler、KVCacheManager、LLMEngine 三层：
- 调度器状态机（swap、un_admit、有序恢复）
- KV 换出换入的元数据一致性（dry_run + GPU）
- 引擎准入循环的 preemption 触发和 never-prefilled 路径

所有测试在 `python -m pytest tests/test_preemption.py -v` 全部通过。
