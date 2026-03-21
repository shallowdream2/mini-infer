"""
Phase 8 启动脚本。

用法：
  python serve.py --model /path/to/Qwen2.5-0.5B-Instruct
  python serve.py --model /path/to/Qwen2.5-0.5B-Instruct --port 8000 --host 0.0.0.0
  python serve.py --dry-run   # 无模型权重时测试服务器

参数：
  --model    模型目录路径（与 tokenizer 相同目录）
  --host     监听地址（默认 0.0.0.0）
  --port     监听端口（默认 8000）
  --dry-run  使用 stub tokenizer + 随机 forward，无需真实模型
  --device   推理设备（默认 cuda:0）
  --dtype    权重精度（默认 float16）
"""

from __future__ import annotations

import argparse

import uvicorn

from mini_infer.config import EngineConfig
from mini_infer.server import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini-infer OpenAI-compatible server")
    parser.add_argument("--model", type=str, default="", help="模型目录路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="使用 stub tokenizer，无需真实模型")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run and not args.model:
        raise SystemExit("请指定 --model 或 --dry-run")

    model_name = args.model if args.model else "dry"
    config = EngineConfig(
        model_name=model_name,
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
        max_batch_size=args.max_batch_size,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
    )

    # 将 config 注入 app.state，lifespan 中读取
    app.state.engine_config = config  # type: ignore[attr-defined]

    print(f"[mini-infer] model={model_name!r}  device={args.device}  port={args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
