"""mini_infer.cli.serve — console script: mini-infer-serve

等价于：python serve.py [args]

用法：
  mini-infer-serve --dry-run --port 8000
  mini-infer-serve --model /path/to/Qwen2.5-7B-Instruct --port 8000
  mini-infer-serve --model /path/to/model --chunk-prefill-size 256 --port 8000
"""
from __future__ import annotations

import argparse

import uvicorn

from mini_infer.core.config import EngineConfig
from mini_infer.serving.server import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini-infer — OpenAI Chat Completions 兼容服务")
    parser.add_argument("--model", type=str, default="", help="模型目录路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="使用 stub tokenizer，无需真实模型")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--chunk-prefill-size", type=int, default=0,
                        help="Phase 9：每步 prefill 的 token 上限（0=禁用）")
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
        chunk_prefill_size=args.chunk_prefill_size,
    )
    app.state.engine_config = config  # type: ignore[attr-defined]

    print(f"[mini-infer] model={model_name!r}  device={args.device}  port={args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
