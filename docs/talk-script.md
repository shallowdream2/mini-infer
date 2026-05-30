# 5 分钟项目讲稿：mini-infer 大模型推理系统

> 面向面试官或技术同行。建议语速：每分钟约 150 字，全文约 750 字。

---

## 开场（30 秒）

这是我用来学习大模型推理系统的一个项目，叫 mini-infer。
它不是对 vLLM 的复刻，而是我从零实现了推理系统的核心机制，
目的是把那些在文章里看起来清晰、实际上很难落地的概念——
比如 PagedAttention、Continuous Batching、Chunked Prefill——
通过真实代码和可复现的 benchmark 数据彻底搞清楚。

---

## 第一部分：请求生命周期（1 分 30 秒）

先说一个请求从进来到出去经历了什么。

用户发一个 `/v1/chat/completions` 请求，FastAPI server 收到后，
交给 AsyncEngine。AsyncEngine 的关键设计是：
它有一个**后台线程持续运行 step loop**，不等 HTTP 请求凑齐。
所以多个并发请求的 prompt 会被自然地 merge 进同一个 decode batch，
这就是 Continuous Batching。

每次 step() 分四步：
1. **准入**：检查 KV cache 有没有空闲 block，够就让请求进 running 队列；不够就抢占低优先级请求，把它的 KV blocks swap 到 CPU。
2. **Prefill**：对新进来的请求做一次完整前向，把 prompt 的 KV 写进 block pool。
3. **Decode Batch**：所有 running 请求一起做一步 forward，用 `flash_attn_with_kvcache` 加 block_table 索引，读各自的历史 KV。
4. **清理**：完成的请求释放 blocks，归还 free pool。

实测 Continuous Batching 把并发 1→8 的吞吐提升了 **3.9×**（55.7 → 219.1 tok/s）。

---

## 第二部分：PagedAttention 和 KV Cache 管理（1 分 30 秒）

PagedAttention 解决的核心问题是**显存碎片和共享**。

传统方案每个请求分配一块连续 GPU 内存，长度不可预测，碎片严重，
也没办法让两个请求共享相同的 system prompt KV。

mini-infer 里的实现：
- GPU 上预分配一个 block tensor pool，shape 是 `[num_gpu_blocks, block_size, num_kv_heads, head_dim]`
- 每个请求维护一个 BlockTable（逻辑块 → 物理块的映射），这个映射直接喂给 flash_attn 的 block_table 参数
- 物理块由 FreeBlockPool 统一管理，请求完成后立即归还

在这个基础上，我还实现了 **Prefix Caching**：
用 block-level SHA-256 链式 hash 识别相同前缀，
命中时复用已有的物理块，引用计数保证 shared block 不被提前淘汰。
实测共享 1 个 block（256 tokens）的 TTFT **降低 22%**。

---

## 第三部分：Chunked Prefill（1 分钟）

长 prompt 场景有个问题：一次 prefill 4096 个 token 需要几十毫秒，
这段时间 decode batch 被阻塞，已有请求的 Inter-Token Latency 会出现 spike。

Chunked Prefill 的思路是把 prefill 切成 256-token 的 chunks，
每步只处理一个 chunk，decode batch 正常继续执行。

调度器需要一个 `PREFILLING` 中间状态，介于 `WAITING` 和 `RUNNING` 之间。
中间的 DynamicCache 保存在内存里，最后一个 chunk 完成后写入 block pool。

实测 ITL spike **降低 57–67%**，这个数据在长 prompt + 混合 decode 的场景下很显著。

---

## 第四部分：其他关键机制（1 分钟）

除了主链路，我还验证了几个重要机制：

**Speculative Decoding**：用 0.5B draft 模型预测 K=4 个 token，7B target 模型验证。
acceptance rate 55.85%，有效减少 target model 的 forward 次数。

**CUDA Graph**：decode_batch 的输入 shape 固定（batch_size 已知），
对每种 batch size 静态捕获一张计算图，replay 消除 Python dispatch 开销。
bs=1 延迟 **降低 28.9%**。

**MLA（DeepSeek-V2/V3 架构）**：实现了 Naive / LatentCache / Absorbed 三种版本，
Latent cache 比 GQA **减少 56.25%** KV cache 体积，这是 DeepSeek 超长上下文的关键。

---

## 收尾（30 秒）

通过 mini-infer，我对推理系统的理解从"知道名词"升级到了"能说清楚每个机制在哪个文件、哪个函数、trade-off 是什么"。

最直接的收获是：我现在能解释清楚为什么 prefill 和 decode 要拆开、
为什么 PagedAttention 是显存管理的转折点、
Chunked Prefill 解决的是延迟分布问题而不是吞吐问题。

如果对某个模块感兴趣，我们可以深入聊。
