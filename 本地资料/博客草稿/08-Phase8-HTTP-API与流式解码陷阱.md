# 从推理引擎到 HTTP 服务：异步包装、Continuous Batching 与流式解码的三个坑

> mini-infer 项目 Phase 8 实战总结。

---

## 背景

mini-infer 做到 Phase 7 时，已经有了一个颇为完整的推理引擎：Paged KV Cache、flash_attn block_tables、Preemption + Priority Scheduling，benchmark 在 batch=8 时能跑到 ~406 tok/s（= HF Transformers baseline）。但它是一个纯 Python 库，只能通过 `engine.generate()` 调用。

Phase 8 的目标是把它包装成可以直接用 `openai` Python 客户端调用的 HTTP 服务：

```
POST /v1/chat/completions
GET  /v1/models
```

听起来不难，不就是在外面套一层 FastAPI 吗？实际落地过程中遇到了三个值得记录的问题。

---

## 问题一：每个 HTTP 请求独立调用 generate() 会打破 Continuous Batching

### 错误直觉

最直观的实现方式是：每来一个 HTTP 请求，启动一个线程，在这个线程里调用 `engine.generate([prompt])`。

```python
# ❌ 错误实现
@app.post("/v1/chat/completions")
async def chat(request):
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, engine.generate, [request.prompt])
    return {"choices": [{"message": {"content": text}}]}
```

这个写法有两个问题：

1. **GPU 独占**：`engine.generate()` 内部是一个完整的推理循环，拿到 GPU 就不放。两个并发请求会串行执行，而不是被 batch 在一起。
2. **线程安全**：`LLMEngine` 不是线程安全的，多线程并发调用会破坏内部状态。

实际跑下来：并发 8 个请求的总吞吐 ≈ 单请求 × 1，没有任何 batching 收益。

### 正确设计：共享 step loop

Continuous Batching 的核心是"**不等请求凑齐，来一个接一个**"。要在 HTTP 服务中保留这个特性，需要把推理循环从每个请求中剥离，变成一个独立运行的后台线程：

```
HTTP 请求 A ─┐
HTTP 请求 B ─┤──→ add_request() ──→ [后台 step loop] ──→ step() 每次处理全部 running 请求
HTTP 请求 C ─┘                                              ↓
                                                   每个请求的 token 投递到各自的 asyncio.Queue
```

每个 HTTP 请求：
1. 在 `asyncio.Queue` 上等待 token
2. 通过 `engine.add_request(rid=rid)` 把自己加入调度队列

后台线程持续运行：
1. `engine.step()` → 本轮 prefill + batch decode
2. 把每个请求新产生的 token 通过 `loop.call_soon_threadsafe(queue.put_nowait, token)` 投递给对应的 queue

这样，无论有多少 HTTP 请求同时进来，它们都会被同一个 `decode_batch()` 一起处理。

```python
class AsyncEngine:
    def _engine_loop(self) -> None:
        """后台线程：持续运行 step，不等待 HTTP 请求。"""
        while self._running:
            if self._engine.has_unfinished_requests():
                new_tokens = self._engine.step()
                for rid, tokens in new_tokens.items():
                    for text in tokens:
                        self._put(rid, text)
                    if self._engine.is_finished(rid):
                        self._put(rid, _DONE)
                        self._engine.cleanup_request(rid)
            else:
                time.sleep(0.001)  # 无请求时短暂休眠，避免空转

    def _put(self, rid: str, item: object) -> None:
        """线程安全地向 asyncio.Queue 投递 token。"""
        queue = self._token_queues.get(rid)
        if queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, item)
```

实测结果（Qwen2.5-7B-Instruct，max_tokens=64）：

| 并发数 | throughput（tok/s） | 相对单并发 |
|--------|---------------------|-----------|
| 1 | 55.7 | 1× |
| 2 | 105.0 | 1.88× |
| 4 | 193.6 | 3.47× |
| 8 | **351.4** | 6.3× |

8 并发吞吐是单并发的 6.3 倍，证明 continuous batching 通过 HTTP 层正常工作。

---

## 问题二：竞态条件——token 会在 queue 注册前被投递

### 现象

第一版 `generate_stream()` 的顺序是这样的：

```python
# ❌ 有竞态：queue 注册在 add_request() 之后
rid = engine.add_request(prompt, max_new_tokens)  # step loop 立刻可见这个请求
queue = asyncio.Queue()
self._token_queues[rid] = queue                    # ← queue 还没注册
```

如果后台线程恰好在 `add_request()` 之后、`_token_queues[rid] = queue` 之前执行了 `step()`，并调用了 `_put(rid, token)`，此时 `_token_queues.get(rid)` 返回 `None`，token 直接丢弃。

消费者永远等不到这个 token，`asyncio.wait_for()` 超时，请求失败。

这个窗口很短（微秒级），但在高并发压力或系统调度延迟时完全可以触发。

### 修复：先注册 queue，再提交请求

将 `request_id` 的生成从 `add_request()` 内部移出，允许外部预先生成 `rid`，在注册 queue 之后再提交请求：

```python
# ✅ 正确顺序
rid = str(uuid4())
queue: asyncio.Queue = asyncio.Queue()
self._token_queues[rid] = queue          # 先注册
self._engine.add_request(               # 后提交（step loop 才能看见这个请求）
    prompt, max_new_tokens, priority, request_id=rid
)
```

对应地，`LLMEngine.add_request()` 新增可选的 `request_id` 参数：

```python
def add_request(self, prompt, max_new_tokens=128, priority=0, request_id=None):
    if request_id is None:
        request_id = str(uuid4())
    ...
```

---

## 问题三：流式解码的空字符串陷阱

这是整个 Phase 8 最有意思的 bug，在 benchmark 阶段才被发现。

### 现象

`benchmark_server.py` 首次运行，发现 throughput = 0 tok/s：

```
并发=1:  0.0 tok/s (0 tokens / 1.14s)
并发=8:  0.0 tok/s (0 tokens / 1.37s)
```

模型明显在运行（每个请求耗时约 1.1s），但一个 token 都没有生成出来。

快速诊断：

```python
rid = engine.add_request("请介绍大语言模型。", max_new_tokens=5)
for i in range(10):
    new_tokens = engine.step()
    print(f"step {i}: {new_tokens}")
    if engine.is_finished(rid):
        break
```

输出：

```
step 0: {'rid': ['', '']}
step 1: {'rid': ['']}
step 2: {'rid': ['']}
step 3: {'rid': ['']}
```

`step()` 返回的全是空字符串。但同样的请求用 `engine.generate()` 完全正常：

```python
out = engine.generate(["请介绍大语言模型。"], max_new_tokens=5)
# 输出：' 大语言模型是一种能够生成自然语言文本'
```

### 根因：逐 token decode 对多字节字符返回空串

追溯到 `model_runner.py`：

```python
# prefill()
next_token_id = _sample_token(out.logits[0, -1], ...)
state.append_generated(next_token_id, "")    # ← token_text 始终是空串

# decode_batch()
next_token_id = _sample_token(logits_batch[b], ...)
state.append_generated(next_token_id, "")    # ← 同上
```

两处都存了空字符串作为 token 的文本。

而 `generate()` 没有这个问题，因为它在所有 decode 步骤完成后，一次性对整个 token 序列做批量解码：

```python
text = tokenizer.decode(state.generated_token_ids, skip_special_tokens=True)
```

问题出在**逐 token decode** 上。对于中文和日文等多字节字符，一个汉字往往对应多个 token ID（或者说，一个 token ID 只代表一个字节），单独 decode 这个 token ID 会得到空字符串或乱码：

```python
# Qwen tokenizer 中，"大" 可能编码为多个 token
tokenizer.decode([token_id_for_part_of_大])  # → "" 或 "<0xe5>"
```

这不是 Qwen 特有的问题，是所有基于 BPE/SentencePiece 的 tokenizer 的共同特性：**token boundary 不等于字符 boundary**。逐 token decode 对多字节字符是不安全的。

### 为什么测试没发现

所有测试用 `dry_run=True`，走的是 `_StubTokenizer`：

```python
class _StubTokenizer:
    def decode(self, token_ids, skip_special_tokens=True) -> str:
        return "".join(f" [{tid}]" for tid in token_ids)
```

stub tokenizer 没有多字节字符的问题，decode `[1]` 返回 `" [1]"`，decode `[1, 2]` 返回 `" [1] [2]"`，永远非空。

测试全部通过，但真实 GPU 路径的 bug 被完全掩盖。

### 修复：增量全序列 decode

正确的流式 decode 方法是**增量全序列 decode**：不是 decode 单个 token，而是 decode 从头到当前位置的所有 token，再与上一步的结果比较，取差值。

```python
# engine.step() 收集新 token 部分
for state in self.scheduler.get_running_states():
    rid = state.request.request_id
    pre = pre_lens.get(rid, 0)          # 这一步之前已有 pre 个 token
    curr = len(state.generated_token_ids)
    if curr > pre:
        tok_ids = state.generated_token_ids
        old_text = (
            tokenizer.decode(tok_ids[:pre], skip_special_tokens=True)
            if pre > 0 else ""
        )
        new_text = tokenizer.decode(tok_ids[:curr], skip_special_tokens=True)
        delta = new_text[len(old_text):]
        if delta:
            new_tokens[rid] = [delta]
```

修复后：

```python
# step() 返回正确的增量文本
step 0: {'rid': [' 大语言']}   # 多个 token 合并后第一个可见字符边界
step 1: {'rid': ['模型']}
step 2: {'rid': ['是一种']}
step 3: {'rid': ['能够']}
```

`AsyncEngine.generate()` 现在能正确返回：

```python
result = await engine.generate("请介绍大语言模型。", max_new_tokens=10)
# → ' 大语言模型是一种能够生成自然语言文本'
```

与直接调用 `LLMEngine.generate()` 结果完全一致。

---

## 实验结果（真实 GPU 数据）

**环境**：Ubuntu 24.04 + RTX 4090，Qwen2.5-7B-Instruct，block_size=256，max_tokens=64，ASGI transport

| 指标 | 值 | 口径说明 |
|------|----|---------|
| 整体延迟（单请求） | ~1157ms | ASGITransport 缓冲完整响应，等于单请求总时间；真实流式 TTFT 需独立 server + curl |
| TPOT（近似） | ~18.5ms/tok | 总时间 / 输出 token 数，与 Phase 6 model_forward ~17.9ms/step 一致 |
| 峰值显存 | 23.34 GB | 与 Phase 6/7 完全相同，HTTP 层无额外显存开销 |
| 吞吐（并发=1） | 55.7 tok/s | tokenizer.encode 精确计算 |
| 吞吐（并发=8） | 351.4 tok/s | — |

Phase 6 直接 LLMEngine 参考（batch=8，max=128）：~406 tok/s。两者差距主要来自 workload 参数差异（64 vs 128 tokens），HTTP 层本身开销可忽略。

---

## 测试策略反思

这个 bug 揭示了一个更普遍的问题：

**dry_run / stub 测试验证的是逻辑流程，不能验证真实路径的数据质量。**

stub tokenizer 和真实 tokenizer 的行为差异太大：
- stub：任意 token_id → 非空字符串，字符边界总是对齐的
- 真实：token_id 可能对应 UTF-8 的一个字节，必须攒够字节才能显示字符

对于任何涉及真实 tokenizer 行为的功能（streaming decode、token 计数、chat template 格式），dry_run 测试是不够的。应该在 CI 中加入一个轻量级真实模型路径的集成测试（比如 GPT-2 这样的小模型），或者至少在 review 阶段明确标注"此功能仅 dry_run 测试，真实路径未验证"。

---

## 总结

Phase 8 把 mini-infer 从推理库变成了一个可以直接用 OpenAI 客户端对接的 HTTP 服务。过程中三个值得记录的设计点：

1. **单后台线程 step loop** 是在 HTTP 层保留 continuous batching 的关键，每个 HTTP 请求只提交 prompt + 订阅 queue，不独占 GPU。

2. **竞态条件修复**：queue 注册必须在 `add_request()` 之前，否则后台线程可能在 queue 注册前就投递 token。

3. **增量全序列 decode**：流式 LLM 推理中，逐 token decode 对多字节字符不安全，正确做法是 `decode(all[:curr])[len(decode(all[:pre])):]`。

第三点被所有基于 dry_run 的测试掩盖，只有跑真实 GPU benchmark 才能发现。这提醒我们：测试 stub 的覆盖率高，不等于真实路径是对的。

---

## 附录：关键代码片段

### AsyncEngine 后台 step loop

```python
def _engine_loop(self) -> None:
    while self._running:
        if self._engine.has_unfinished_requests():
            new_tokens = self._engine.step()
            for rid, tokens in new_tokens.items():
                for text in tokens:
                    self._put(rid, text)
                if self._engine.is_finished(rid):
                    self._put(rid, _DONE)
                    self._engine.cleanup_request(rid)
        else:
            time.sleep(0.001)

def _put(self, rid: str, item: object) -> None:
    queue = self._token_queues.get(rid)
    if queue is not None and self._loop is not None:
        self._loop.call_soon_threadsafe(queue.put_nowait, item)
```

### 增量 tokenizer decode（engine.step()）

```python
for state in self.scheduler.get_running_states():
    pre = pre_lens.get(state.request.request_id, 0)
    curr = len(state.generated_token_ids)
    if curr > pre:
        tok_ids = state.generated_token_ids
        old_text = tokenizer.decode(tok_ids[:pre], skip_special_tokens=True) if pre > 0 else ""
        new_text = tokenizer.decode(tok_ids[:curr], skip_special_tokens=True)
        delta = new_text[len(old_text):]
        if delta:
            new_tokens[state.request.request_id] = [delta]
```
