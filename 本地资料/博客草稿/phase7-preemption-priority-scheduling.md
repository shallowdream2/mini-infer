# 在推理引擎里实现 Preemption：一个让 dry_run 遮住的 crash bug 和 1100 MB/s 的 swap 带宽

本文是 mini-infer 系列的第 7 篇，记录 Phase 7 的实现过程：在推理引擎里加入 KV cache swap（GPU↔CPU）和优先级调度。

主要内容有三块：
1. 为什么推理引擎需要 preemption，设计时选了什么方案、舍弃了什么
2. 实现时遇到的真实 bug：一个在 dry_run 下完全不可见的 crash
3. GPU↔CPU swap 实测只跑出 1100 MB/s 带宽的原因分析

---

## 背景：为什么需要 swap

连续批处理（continuous batching）解决了"等 batch 填满才推理"的问题，但引入了新的资源竞争：每个请求都要占据一段 KV cache 直到生成完毕，总 GPU 显存是硬上限。

当一批高优先级的短请求进来时，如果低优先级的长请求正在占用大量 KV cache 块，新请求就没有空间可用，只能等——即使 GPU 算力是空闲的。解法是**抢占**（preemption）：把低优先级请求的 KV cache 从 GPU 换出到 CPU，腾出空间给高优先级请求，稍后再换回来继续生成。

Preemption 的另一种实现是**重计算**（recomputation）：直接丢掉低优先级请求的 KV cache，被换出的请求等到资源充足后重新 prefill。这不需要 CPU 内存，但被换出的请求要重做一次完整的 prefill forward（对于长 prompt，成本很高）。mini-infer 选择了 swap 方案，原因是：在学习项目里 CPU 内存限制不是首要问题，swap 能完整展示 GPU↔CPU 拷贝的工程细节。

---

## 设计：三种状态的请求

Phase 7 给请求状态机新增了一个 `swapped` 状态，与原有的 `waiting`、`running` 并列：

```
waiting → running → (finished)
              ↑         ↓
           swap_in   swap_out
              ↑         ↓
           swapped ←────┘
```

调度器维护三个容器：
- `_waiting: deque`：还未进入 GPU
- `_running: dict[request_id, state]`：正在 GPU 上推理
- `_swapped: deque`：已换出到 CPU，等待换回

优先级规则简单：`priority` 数值越小优先级越高（`0` = 最高）。换出时选 `_running` 中 `priority` 最大的那个。

---

## 实现：准入循环的两个分支

抢占逻辑集中在引擎的准入循环里。每次迭代，引擎尝试把等待队列头部的请求接入 GPU：

```python
if self.kv_cache.num_free_blocks() >= blocks_needed:
    # 正常准入
    state = self.scheduler.pop_next_waiting()
    self.kv_cache.init_request(state)
    self.scheduler.add_to_running(state)
    newly_admitted.append(state)
else:
    # 块不足：找优先级更低的 running 请求换出
    victim = self.scheduler.get_lowest_priority_running()
    if victim is not None and victim.request.priority > next_state.request.priority:
        if not victim.prefilled:
            # 分支 A：刚准入但还没 prefill → 撤销准入
            self.kv_cache.free_request(victim)
            self.scheduler.un_admit(victim)
            newly_admitted.remove(victim)
        else:
            # 分支 B：已完成 prefill → swap_out 到 CPU
            self.kv_cache.swap_out(victim)
            self.scheduler.mark_swapped(victim)
        continue  # 腾出块后重新检查
```

这里有一个细节：`victim.prefilled` 的判断。如果没有这个分支，会有一个严重的 bug——见下节。

---

## Bug：dry_run 遮住的 never-prefilled crash

在代码 review 时发现了一个 crash 场景，复现路径是：

1. 等待队列有请求 A（高优先级）和请求 B（低优先级）
2. 引擎准入循环先接入 B（因为 A 不够块），B 加入 `newly_admitted`
3. 继续尝试接入 A，块还是不够
4. 找到 victim = B（低优先级），调用 `swap_out(B)`
5. `swap_out` 内部调用 `free_request(B)`，清空 `_block_tables[B.request_id]`
6. B 还在 `newly_admitted` 列表里
7. 循环结束后进入 prefill 阶段，`write_prefill_kv` 访问 `_block_tables[B.request_id]` → **KeyError**

问题的根源在于：`_block_tables` 在 `free_request` 里被 `pop` 掉了，但 B 还在本轮的局部列表 `newly_admitted` 里，prefill 不知道 B 已经被撤销。

**为什么 dry_run 看不见这个 bug？**

`write_prefill_kv` 有这一行：

```python
def write_prefill_kv(self, ...):
    if self._dry_run:
        return  # ← 提前返回，根本不访问 _block_tables
```

所有 dry_run 测试都经过这个早返回，完全不会触发 `_block_tables` 访问。测试全绿，但真实 GPU 路径会 crash。

**修复**：增加 `un_admit()` 方法——不走 swap 路径，直接把请求从 `_running` 移回 `_waiting` 队尾，同时从 `newly_admitted` 里移除：

```python
def un_admit(self, state: RequestState) -> None:
    """撤销准入：从 running 移回 waiting 队尾。仅用于 prefilled == False 的请求。"""
    self._running.pop(state.request.request_id, None)
    self._waiting.append(state)  # 队尾，不是队头
```

注意是 `append`（队尾）而不是 `appendleft`（队头）。如果放队头，高优先级请求 A 会再次触发同样的换出逻辑，形成循环抢占死锁（A 接不进来 → 换出 B → B 回队头 → A 还是接不进来……）。放队尾可以让 A 先跑完一步、释放一些块，再轮到 B 重新尝试准入。

---

## swap_out / swap_in 的实现

`swap_out` 逐层、逐块把 GPU KV tensor 拷贝到 CPU 的一个新 tensor 里：

```python
for l in range(self.num_layers):
    k_cpu = torch.zeros(seq_len, self.num_kv_heads, self.head_dim)
    v_cpu = torch.zeros(seq_len, self.num_kv_heads, self.head_dim)
    for blk_idx, phys_blk in enumerate(block_table):
        start = blk_idx * self.block_size
        end = min(start + self.block_size, seq_len)
        n = end - start
        k_cpu[start:end] = self.k_cache[l][phys_blk, :n].cpu()
        v_cpu[start:end] = self.v_cache[l][phys_blk, :n].cpu()
    cpu_kv.append((k_cpu, v_cpu))
state.cpu_kv = cpu_kv
self.free_request(state)  # 释放 GPU 块
```

换出后 KV 存在 `state.cpu_kv` 里，GPU 块立刻归还到空闲池。`swap_in` 做逆向操作：重新分配 GPU 块，再把 CPU 数据写回。

---

## 实验：swap 延迟和带宽

环境：Ubuntu 24.04 + RTX 4090，Qwen2.5-7B-Instruct（28 层，4 kv_head，head_dim=128，fp16）。

### 调度器纯元数据开销

| 操作 | median |
|------|--------|
| swap_out（dict + deque） | 0.36 µs |
| swap_in | 0.60 µs |

调度层本身的开销可忽略不计，对 ~20 ms/step 的 decode 延迟影响 < 0.005%。

### GPU↔CPU 真实拷贝延迟

| seq_len | KV 大小 | swap_out | swap_in | 有效带宽 |
|---------|---------|----------|---------|---------|
| 32 | 1.8 MB | 1.60 ms | 1.69 ms | ~1100 MB/s |
| 256 | 14.7 MB | 13.54 ms | 13.66 ms | ~1080 MB/s |
| 512 | 29.4 MB | 26.43 ms | 27.54 ms | ~1090 MB/s |

有效带宽稳定在 ~1050–1150 MB/s，远低于 PCIe 4.0 x16 的理论峰值（~32 GB/s）。

### 为什么只有 1100 MB/s？

原因是 Python 层的逐块循环。对于 seq_len=32、block_size=16：

```
28 层 × 2（K + V）× 2 个物理块 = 112 次拷贝
每次大小：16 tokens × 4 heads × 128 dim × fp16 ≈ 16 KB
```

每次 `.cpu()` 调用：
1. 触发一次独立的 CUDA kernel launch
2. 插入一个 `cudaMemcpyAsync`
3. `torch.cuda.synchronize()` 在收集全部 tensor 后统一 sync（但每次拷贝本身仍是单次 DMA）

问题不是 synchronize 的次数，而是每次 PCIe 传输的数据量太小（16 KB），PCIe 协议本身有固定启动开销（TLP header、Flow Control 等），小包传输的效率远低于大块。

vLLM 的做法是把所有层的 KV 合并成一个大 tensor 再做一次拷贝，这样 PCIe 可以跑满（理论约 20–25 GB/s 实测带宽）。mini-infer 选择逐块循环是为了清楚展示原理，不是工程优化目标。

对于 seq_len=256 的请求，swap 耗时约 13.5 ms，相当于 0.5–1 个 decode step。不频繁触发时可接受；高频换出场景会成为瓶颈。

### 吞吐无回归

| 路径 | throughput |
|------|------------|
| 无优先级（baseline） | 198.3 tok/s |
| 有优先级、块充足 | 198.2 tok/s |
| 差值 | -0.0% |

当 KV cache 充足不触发 swap 时，优先级调度对 decode forward 路径零影响（符合预期——调度逻辑只在准入循环里，不进入 CUDA kernel 路径）。

---

## 局限性

1. **真实 swap_out → swap_in 路径的端到端测试**：因为 `generate()` API 要求所有请求同时提交，无法方便地构造"请求 A 先 prefill、被换出、再换回"的完整流程。该场景的正确性通过单元测试在 dry_run 层面验证。

2. **swap 带宽未优化**：当前实现的 ~1100 MB/s 是逐块拷贝的结果，改进方向是合并所有层为一次大 tensor 拷贝。

3. **只支持 swap-based preemption**：没有实现重计算路径（recompute），swap 路径需要 CPU 内存保存 KV。

---

## 总结

Phase 7 在 mini-infer 里实现了 preemption 和优先级调度。最值得关注的两个工程细节：

1. **`prefilled` 状态的必要性**：区分"已完成 prefill 的请求（可 swap_out）"和"刚准入但还没 prefill 的请求（应 un_admit）"是正确性的关键。两者在 dry_run 下行为一致，但在 GPU 路径上语义完全不同。

2. **逐块循环是 PCIe 带宽瓶颈的根源**：从 32 GB/s 理论到 1100 MB/s 实测，不是测量误差，而是每次 16 KB 小包在 PCIe 协议层的必然开销。理解这个才能看懂 vLLM 为什么要把所有层合并后再拷贝。

---

*mini-infer 是个人推理系统学习项目，实验环境：Ubuntu 24.04 + 2 × RTX 4090，模型：Qwen2.5-7B-Instruct（本地缓存，fp16）。*
