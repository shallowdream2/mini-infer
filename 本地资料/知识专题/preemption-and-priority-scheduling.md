# 知识专题：LLM 推理引擎的 Preemption 与优先级调度

这个文件深度梳理 LLM 推理引擎中 preemption（抢占）机制的原理、实现方式、设计取舍和常见误区，结合 mini-infer Phase 7 的具体实现。

---

## 一、问题定义

LLM 推理引擎运行 continuous batching：多个请求共享 GPU，每个请求持续占用 KV cache 直到生成结束。GPU 显存是有限资源，KV cache 占用随生成长度线性增长。

**冲突场景**：GPU KV cache 接近满时，一个高优先级的短请求来了，但没有空间接收它。现有的 running 请求都还没生成完，无法主动释放空间。

如果不做任何处理，高优先级请求只能等待——即使 GPU 算力此刻是空闲的，显存是唯一瓶颈。

Preemption 的目标：**把低优先级请求的 KV cache 暂时搬离 GPU，腾出空间给高优先级请求**，低优先级请求稍后恢复。

---

## 二、核心原理

### 2.1 KV cache 的可迁移性

Autoregressive decode 的每一步只需要读取已有 KV cache，不需要从头重新计算（这正是 KV cache 的意义）。因此，如果一个请求的 KV cache 被完整保存下来，它可以在任意时刻从中断处继续——只要把 KV cache 恢复到 GPU。

这给 preemption 提供了理论基础：KV cache 是请求的"完整状态"，可以在 GPU 和 CPU 之间搬运。

### 2.2 两种恢复策略

**Swap（换出/换入）**：

```
swap_out: GPU KV → CPU RAM（保存完整 KV）
swap_in:  CPU RAM → GPU KV（恢复 KV）
```

被换出的请求在 swap_in 后直接从中断的 decode 步继续，不需要重做 prefill。代价：需要 CPU 内存，存在 PCIe 传输延迟。

**Recompute（重计算）**：

直接丢掉 GPU KV，被换出的请求重新进入 waiting 队列，等到资源充足时重新 prefill。代价：prefill forward 是 $O(L^2)$ 复杂度，长 prompt 重计算成本高。

**选择依据**：
- 如果 CPU 内存充足、prompt 较长：swap 更优（避免重计算 prefill）
- 如果 CPU 内存紧张、prompt 较短：recompute 更简单（无需 CPU ↔ GPU 拷贝）
- vLLM 支持两种策略，通过 `preemption_mode` 参数控制

### 2.3 优先级的定义

最简单的优先级是静态数值：请求提交时附带 `priority`，数值越小优先级越高。换出时选 running 中优先级最低（数值最大）的请求。

更复杂的策略：
- **FCFS（先来先服务）**：按提交时间，最早提交的最高优先级
- **LAS（Least Attained Service）**：已生成 token 最少的请求优先（公平性）
- **SJF（Shortest Job First）**：预测生成长度短的请求优先（最大化吞吐）

---

## 三、工程实现方式

### 3.1 状态机设计

请求有三种状态，调度器维护三个容器：

```
waiting: deque  → 尚未分配 GPU 资源
running: dict   → 当前在 GPU 上推理
swapped: deque  → 已换出到 CPU，等待换回
```

状态转移：
- `waiting → running`：准入（alloc GPU blocks + init KV）
- `running → swapped`：swap_out（copy KV to CPU, free GPU blocks）
- `swapped → running`：swap_in（alloc GPU blocks, copy KV from CPU）
- `running → (done)`：生成结束（free GPU blocks）

### 3.2 准入循环的抢占逻辑

每次迭代，在尝试准入新请求时检查空闲块：

```python
if free_blocks >= blocks_needed:
    # 正常准入
else:
    victim = get_lowest_priority_running()
    if victim and victim.priority < new_request.priority:
        swap_out(victim)
        # 现在块够了，再准入新请求
    else:
        break  # 无法换出，等下一步
```

注意 `continue` 而不是 `break`：换出后应立即重试准入，不要退出循环。

### 3.3 never-prefilled 请求的特殊处理

**关键问题**：刚被准入（已分配 KV 块）但还没 prefill 的请求，如果被选为换出 victim，其 KV 数据是零值（尚未写入），不应该保存到 CPU。

如果直接调用 `swap_out`，之后 `swap_in` 恢复的是零值 KV，decode 时会产生错误 token。更严重的问题：`swap_out` 内部调用 `free_request`，清空了 `_block_tables`，但该请求还在 `newly_admitted` 列表里，后续 `write_prefill_kv` 会 **KeyError crash**。

**解决方案**：用 `prefilled: bool` 标志区分两种情况。`prefilled == False` 时：
```python
free_request(victim)   # 归还 GPU 块（相当于从未分配过）
un_admit(victim)       # 从 running 移回 waiting（不进 swapped）
newly_admitted.remove(victim)  # 从本轮 prefill 列表中撤销
```

### 3.4 swap_in 的恢复顺序

换入时按 FIFO 顺序：最先换出的最先换入。这避免了优先级倒置（低优先级请求被换出很久，资源空闲时应该优先恢复它）。

---

## 四、设计取舍

### swap 粒度：逐块 vs 合并

**逐块**（mini-infer 实现）：

```python
for l in range(num_layers):
    for phys_blk in block_table:
        k_cpu[...] = k_cache[l][phys_blk, :n].cpu()
```

实测带宽：~1100 MB/s（PCIe 4.0 x16 理论 ~32 GB/s 的 3.4%）。原因：每次 `.cpu()` 是独立的 PCIe 传输，16 KB 的小包在 PCIe TLP 协议上有固定开销。

**合并拷贝**（vLLM 实现思路）：

```python
# 伪代码：把所有层 KV 合并为一个大 tensor，一次传输
all_kv = torch.cat([kv_layer_i for i in range(num_layers)], dim=0)
cpu_kv = all_kv.cpu()  # 一次 PCIe 传输
```

预期带宽：~10–25 GB/s（接近 PCIe 实测峰值）。

**取舍**：mini-infer 优先展示原理清晰度，不做工程级 swap 优化。

### 优先级更新 vs 静态优先级

mini-infer 使用静态优先级（提交时固定）。实际系统里，优先级可能会随等待时间动态调整（优先级老化，aging），防止低优先级请求饥饿。

### preemption 阈值

不是每次 KV 快满了就换出，可以设置阈值：只有空闲块低于 N 时才触发换出。频繁的小换出会浪费 PCIe 带宽。

---

## 五、常见误区

**误区 1：swap_out 后的请求不需要重新 prefill**

正确。KV cache 保存了 prompt + 已生成 token 的完整注意力状态，swap_in 后直接从 decode 第 k 步继续，不需要重新做 prefill。这是 swap vs recompute 的核心区别。

**误区 2：un_admit 的请求放回队头更"公平"**

错误。放队头会导致同一个低优先级请求被反复准入、反复因块不够被驱逐，形成无限循环。放队尾可以让当前批次的高优先级请求先推进，之后再为低优先级请求腾出空间。

**误区 3：swap 带宽 = PCIe 理论带宽**

错误。Python 层的逐块拷贝会严重拖累有效带宽。在 mini-infer 实现里，seq_len=256 的请求换出耗时 ~13.5 ms，而 PCIe 传输 14.7 MB 理论只需 ~0.5 ms。差距 ~27× 全部来自 Python 调度和 CUDA kernel launch 开销。

**误区 4：preemption 只影响调度层，不影响 decode kernel**

正确（对于 swap 策略）。swap_out 和 swap_in 发生在 decode step 的边界，不进入 CUDA kernel 路径。调度层变更对 decode forward 的延迟影响为零（mini-infer 实测吞吐差值 0.0%）。

---

## 六、和 mini-infer 的关系

Phase 7 实现了 swap-based preemption 和静态优先级调度：

- `request.py`：`priority` 字段（Request）和 `prefilled`、`cpu_kv`、`swapped_seq_len` 字段（RequestState）
- `kv_cache.py`：`swap_out`、`swap_in` 方法（逐块 Python 循环实现）
- `scheduler.py`：`_swapped` 队列、`mark_swapped`、`move_swapped_to_running`、`get_lowest_priority_running`、`un_admit`
- `engine.py`：准入循环的 preemption 逻辑（`prefilled` 分支）、swap_in 恢复阶段、主循环条件扩展

---

## 七、进一步阅读

- vLLM 论文（Kwon et al., 2023）§4 关于 preemption 的描述
- vLLM 源码 `vllm/core/scheduler.py`：`_schedule_prefills` 和 `_schedule_running` 的换出逻辑
- vLLM `BlockSpaceManager.can_swap_in/out` 的块计数逻辑
- PCIe 带宽特性：[PCIe Gen4 Bandwidth Explained](https://www.techpowerup.com/forums/threads/pcie-bandwidth.280079/)（理解为什么小包传输带宽低）
