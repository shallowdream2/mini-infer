# Phase 8 知识笔记：Async HTTP API + 流式 LLM 推理

## 核心概念

### Continuous Batching 在 HTTP 服务中的保留

要让 HTTP 服务保留 continuous batching，关键是**不能让每个请求独占推理循环**。正确模式：

- 单后台线程持续运行 `step()` 循环
- HTTP 请求通过 `add_request()` 加入调度队列
- 每个请求通过 `asyncio.Queue` 异步等待 token
- 多个并发 HTTP 请求的 prompt 被同一次 `decode_batch()` 处理

### 跨线程 asyncio 通信

后台线程（非 event loop 线程）向 asyncio 队列投递数据：

```python
# ✅ 正确
loop.call_soon_threadsafe(queue.put_nowait, item)

# ❌ 错误：在非 event loop 线程里直接调用 coroutine
await queue.put(item)
```

### SSE（Server-Sent Events）格式

```
data: {"id": "...", "object": "chat.completion.chunk", "choices": [{"delta": {"content": "..."}}]}\n\n
data: [DONE]\n\n
```

每条消息以 `data: ` 开头，以 `\n\n` 结束。FastAPI 用 `StreamingResponse` + async generator 实现。

## 流式解码：增量全序列 decode

**问题**：逐 token decode 对多字节字符（中文、日文等）不安全。单个 token ID 可能对应 UTF-8 的一个字节，decode 返回空字符串。

**正确方式**：维护全序列，每步计算差值：

```python
old_text = tokenizer.decode(all_ids[:pre], skip_special_tokens=True)
new_text = tokenizer.decode(all_ids[:curr], skip_special_tokens=True)
delta = new_text[len(old_text):]  # 只取新增部分
```

这是流式 LLM 推理的标准做法（vLLM、TGI 等系统都用类似的增量 decode 方案）。

## 测试 FastAPI 的 lifespan

FastAPI 的 lifespan 在 `httpx.AsyncClient(transport=httpx.ASGITransport(...))` 下默认不触发。需要用 `asgi_lifespan.LifespanManager`：

```python
async with LifespanManager(app) as manager:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=manager.app)) as client:
        ...
```

## 竞态条件的根本形式

任何"生产者-消费者"场景中，**消费者的接收端（queue/channel）必须在生产者开始之前注册**：

```python
# ❌ 竞态：生产者 (add_request) 先启动，消费者 (queue) 后注册
rid = engine.add_request(prompt)
self._queues[rid] = asyncio.Queue()  # 生产者可能已经在投递 token

# ✅ 正确：消费者先注册
rid = str(uuid4())
self._queues[rid] = asyncio.Queue()  # 先注册
engine.add_request(prompt, request_id=rid)  # 再启动生产者
```

## Pydantic v1 vs v2

项目环境 Pydantic 1.10.x：
- 序列化用 `.json()`，不是 `.model_dump_json()`
- 字段默认值用 `Field(default_factory=...)` 可以
- `Literal` 类型正常支持

## 数字

| 指标 | 值 |
|------|----|
| 并发=8 throughput | 351.4 tok/s |
| TPOT（近似） | ~18.5ms/tok |
| 峰值显存 | 23.34 GB（无额外开销） |
