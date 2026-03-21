# 知识专题：异步 LLM 服务架构与流式解码

## 主题

如何将同步 LLM 推理引擎包装为支持 continuous batching 的异步 HTTP 服务，以及流式 token 解码的正确实现方式。

---

## 一、问题定义

**输入**：一个同步批量推理接口 `LLMEngine.generate(prompts: list[str]) -> list[str]`，内部实现了 continuous batching。

**目标**：
1. 提供 HTTP 接口，兼容 OpenAI Chat Completions API 格式
2. 多个并发 HTTP 请求仍能被 continuous batching 合并到同一 decode batch
3. 支持 SSE 流式推送，客户端逐 token 接收

**挑战**：HTTP 请求是异步的（每个请求独立到达），但 continuous batching 要求多个请求同时进入同一个 decode batch。两者天然矛盾。

---

## 二、核心原理

### 为什么"每请求独立 generate()"不行

```
HTTP 请求 A → 线程1 → engine.generate(["A"])  ← 独占 GPU，耗时 T
HTTP 请求 B → 线程2 → 等待线程1 → engine.generate(["B"])  ← 耗时 2T
```

两个请求串行执行，没有 batching，GPU 利用率低。

### 共享 step loop 的设计

将推理循环从每个请求中剥离：

```
[后台线程] 持续运行 step loop，不等请求凑齐：
  while True:
    new_tokens = engine.step()     ← 处理所有 running 请求
    for rid, tokens in new_tokens:
      投递 tokens 到对应请求的 asyncio.Queue

[HTTP 请求处理协程]
  1. 注册 asyncio.Queue
  2. engine.add_request(prompt, request_id=rid)  ← 加入调度队列
  3. await queue.get()  ← 等待 token
  4. yield token (SSE)
```

这样，HTTP 请求 A 和 B 同时调用 `add_request()`，后台 `step()` 下一次迭代时会把 A 和 B 一起放入 `decode_batch()`。

### 跨线程的 asyncio.Queue 写入

后台线程（非 event loop 线程）向 asyncio.Queue 写入必须通过 `call_soon_threadsafe`：

```python
loop.call_soon_threadsafe(queue.put_nowait, token)
```

不能直接调用 `await queue.put(token)`（后台线程没有 event loop），也不能用 `asyncio.run_coroutine_threadsafe`（可以但复杂，且不需要等待结果）。

### 完成哨兵

使用独立 sentinel 对象标记流结束，避免与正常 token（包括空字符串）冲突：

```python
_DONE = object()  # 唯一对象，用 `is` 比较
```

---

## 三、工程实现方式

### AsyncEngine（mini-infer Phase 8 实现）

```python
class AsyncEngine:
    def __init__(self, config):
        self._engine = LLMEngine(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._token_queues: dict[str, asyncio.Queue] = {}

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._thread.start()

    async def generate_stream(self, prompt, max_new_tokens=128):
        rid = str(uuid4())
        queue = asyncio.Queue()
        self._token_queues[rid] = queue           # 先注册
        try:
            self._engine.add_request(prompt, max_new_tokens, request_id=rid)  # 再提交
            while True:
                token = await asyncio.wait_for(queue.get(), timeout=60.0)
                if token is _DONE:
                    return
                yield token
        finally:
            self._token_queues.pop(rid, None)
```

### 增量全序列 decode

```python
# engine.step() 中收集新 token 文本
for state in running_states:
    pre = pre_lens[state.request_id]    # step 前的 token 数
    curr = len(state.generated_token_ids)  # step 后的 token 数
    if curr > pre:
        old_text = tokenizer.decode(state.generated_token_ids[:pre], skip_special_tokens=True)
        new_text = tokenizer.decode(state.generated_token_ids[:curr], skip_special_tokens=True)
        delta = new_text[len(old_text):]
        if delta:
            new_tokens[state.request_id] = [delta]
```

---

## 四、设计取舍

### 单线程 step loop vs 多线程

- **单线程**（mini-infer 选择）：简单，天然线程安全，充分利用 continuous batching
- **多线程**：可能在多 GPU 场景有优势，但复杂度高，LLMEngine 内部状态管理困难

### ASGITransport vs 真实 uvicorn

- **ASGITransport（测试用）**：直接调用 ASGI app，缓冲完整响应，TTFT 无法测量，适合功能测试
- **uvicorn（生产/真实延迟测量）**：真实流式，需要独立进程 + curl/流式客户端

### streaming decode：逐 token vs 增量全序列

| 方案 | 优点 | 缺点 |
|------|------|------|
| 逐 token decode | O(1) per step | 多字节字符返回空串，不安全 |
| 增量全序列 decode | 结果正确，处理字符边界 | 每步 2 次 tokenizer.decode，O(seq_len) 时间 |

实践中选增量全序列：tokenizer decode 是 CPU 字典查找，即使 seq_len=512 也只需几毫秒，远小于 GPU forward。

---

## 五、常见误区

### 误区 1：逐 token decode 是正确的

```python
# ❌ 看起来合理，但中文等多字节字符会返回空串
token_text = tokenizer.decode([token_id])
```

例：假设汉字"大"对应 token ID 99546，但 tokenizer 内部用 BPE，"大"可能拆分为多个字节 token，每个单独 decode 返回空字符串。

### 误区 2：queue 注册晚了没关系

"刚提交请求，后台线程还没开始处理，queue 注册稍微晚一点没问题。"

实际上：后台线程持续在轮询，`add_request()` 调用后立刻可能被处理。在多核 CPU 上，线程调度是不确定的，微秒级窗口足以触发竞态。

### 误区 3：FastAPI lifespan 在测试中自动触发

`httpx.AsyncClient(transport=httpx.ASGITransport(...))` 默认不触发 FastAPI 的 lifespan 事件。必须使用 `asgi_lifespan.LifespanManager` 显式触发。

### 误区 4：dry_run 测试通过 = 真实路径正确

dry_run 的 stub tokenizer 行为与真实 tokenizer 差异极大。涉及 tokenizer.decode() 的功能（streaming decode、token 计数、chat template）必须用真实路径测试。

---

## 六、和 mini-infer 的关系

Phase 8 是 mini-infer 的最后阶段，将前 7 个 Phase 积累的推理能力（Paged KV Cache、True PagedAttention、Preemption）包装为标准 HTTP 服务：

- `AsyncEngine`：LLMEngine 的异步包装，维护后台 step loop
- `server.py`：FastAPI 路由，调用 AsyncEngine 的流式接口
- `engine.py`：新增 `add_request/step/cleanup_request` 单步接口，`step()` 使用增量 decode

关键 benchmark 数据：8 并发 351.4 tok/s（= 单并发 6.3×），证明 continuous batching 通过 HTTP 层正常工作。

---

## 七、进一步阅读

- vLLM AsyncLLMEngine 源码：对比更完整的生产实现
- Python threading + asyncio 通信模式：`loop.call_soon_threadsafe` vs `run_coroutine_threadsafe`
- HuggingFace Text Generation Inference（TGI）：类似架构的工业级实现
- FastAPI lifespan events 文档：`@asynccontextmanager` lifespan 的正确用法
