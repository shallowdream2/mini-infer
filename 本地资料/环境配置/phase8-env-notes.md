# Phase 8 环境配置笔记

这个文件记录 Phase 8 新增的 Python 包依赖和配置注意事项。

## 新增依赖

Phase 8 引入了 HTTP server 能力，需要以下额外包：

| 包 | 用途 | 是否已装 |
|----|------|---------|
| `fastapi` | HTTP 路由框架 | ✅ 已在 ai-infra |
| `uvicorn` | ASGI server，serve.py 使用 | ✅ 已在 ai-infra |
| `httpx` | 异步 HTTP 客户端，test_server.py 使用 | ✅ 已在 ai-infra |
| `pytest-asyncio` | 异步测试支持 | ✅ 已在 ai-infra |
| `asgi-lifespan` | 触发 FastAPI lifespan 的测试工具 | ✅ 已在 ai-infra |
| `pydantic` | 数据模型，1.10.13（非 v2）| ✅ 已在 ai-infra |

验证命令（Phase 8 前置条件）：

```bash
conda run -n ai-infra python -c "import fastapi, uvicorn, httpx, pytest_asyncio, asgi_lifespan; print('ok')"
```

## Pydantic v1 兼容性

ai-infra 环境的 Pydantic 版本是 **1.10.13**（非 v2）。Phase 8 代码对应使用 v1 API：

- 序列化：`.json()` 而非 `.model_dump_json()`
- 字段定义：`Field(default_factory=...)` 正常支持
- `Literal` 类型正常支持

如果未来升级 Pydantic 到 v2，需要同步更新 `mini_infer/openai_schema.py`。

## pytest-asyncio 配置

`pyproject.toml` 中已配置：

```toml
[tool.pytest.ini_options]
asyncio_mode = "strict"
```

在 `strict` 模式下，所有异步测试函数必须加 `@pytest.mark.asyncio` 装饰器。

## serve.py 关键参数

- `--block-size`：默认值 **256**（Phase 8 修复，原来是 16，导致非 dry_run 启动即报 ValueError）
- `--dry-run`：无需模型权重，快速验证 API 结构
- `--port`：默认 8000

## 代理环境注意事项

服务器环境存在代理变量（如 `ALL_PROXY` / `HTTP_PROXY` / `HTTPS_PROXY`），会影响 OpenAI SDK 客户端直连本地服务。

已验证可行的调用方式：

```bash
env -u ALL_PROXY -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u http_proxy -u https_proxy \
  conda run -n ai-infra python -c "from openai import OpenAI; client = OpenAI(base_url='http://127.0.0.1:8000/v1', api_key='none'); resp = client.chat.completions.create(model='mini-infer', messages=[{'role':'user','content':'hello'}], max_tokens=8); print(resp.choices[0].message.content)"
```

结论：问题在本机代理环境，而不是 API 格式本身。Phase 8 的 OpenAI SDK 基础调用现已验证通过，但仍需遵守当前接口是 Chat Completions 受限子集这一边界。
