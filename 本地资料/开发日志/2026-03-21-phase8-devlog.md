# Phase 8 开发日志（2026-03-21）

## 阶段目标

将 mini-infer 包装成 OpenAI-compatible HTTP 服务。

## 关键决策

### 架构选型：单后台线程 step loop

最初的方案是每个 HTTP 请求在独立线程里调用 `engine.generate()`，但这样无法保留 continuous batching。改用单后台线程持续运行 `engine.step()`，HTTP 请求通过 `add_request()` 加入调度队列，通过 `asyncio.Queue` 接收 token。

这个架构需要新增 `LLMEngine.add_request()` / `step()` / `cleanup_request()` 等单步接口，并实现 `AsyncEngine` 包装类。

### 跨线程通信：loop.call_soon_threadsafe

后台线程（sync）向前台 asyncio 协程传递 token，必须使用 `loop.call_soon_threadsafe(queue.put_nowait, token)`。直接调用 `queue.put_nowait` 或 `asyncio.run_coroutine_threadsafe` 都不合适。

## 踩到的坑

### 坑 1：竞态条件

第一版 `generate_stream()` 在 `add_request()` 之后才注册 queue。后台线程可能在注册之前就产出 token，`_put()` 找不到 queue，token 丢弃。

修复：先生成 `rid`，注册 `_token_queues[rid]`，再调用 `add_request(request_id=rid)`。

### 坑 2：serve.py 默认 block_size=16

argparse 的 `--block-size default=16` 覆盖了 `EngineConfig.block_size=256` 的默认值。真实模型启动即报 ValueError。改为 `default=256`。

### 坑 3：step() 空字符串 Bug（在 benchmark 阶段发现）

这是整个 Phase 8 最严重的 bug，测试全通过但真实路径完全错误：

**现象**：benchmark 首次运行，throughput = 0 tok/s。诊断：`step()` 返回 `['', '']`。

**根因**：`model_runner.py` 在真实 GPU 路径的 `prefill()` 和 `decode_batch()` 中，始终调用 `append_generated(token_id, "")`，token 文本始终为空字符串。`generate()` 不受影响，因为它在末尾做批量 decode。`step()` 依赖 `generated_text_parts` 这个空数组。

**为什么测试没发现**：所有测试用 `dry_run=True`，stub tokenizer 返回 `" [1] [2]..."` 格式，永远非空。

**修复**：`step()` 改用增量全序列 decode：
```python
delta = decode(ids[:curr])[len(decode(ids[:pre])):]
```

**教训**：干跑测试≠真实路径正确。涉及 tokenizer decode 的功能需要真实路径集成测试。

## 时间线

- 设计 AsyncEngine 架构 + 实现 add_request/step 接口
- 实现 server.py + openai_schema.py + serve.py
- 编写 test_server.py，调试 LifespanManager + ASGI transport
- review 发现竞态条件 + serve.py block_size bug，修复
- benchmark 发现 step() 空字符串 bug，修复增量 decode
- re-review + re-benchmark，通过
