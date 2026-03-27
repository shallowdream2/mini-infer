"""examples/local_chat.py — 不启动 HTTP 服务，直接使用 Python API 与 mini-infer 交互。

使用方法
--------
方式 A：dry-run（无需模型权重）

    python examples/local_chat.py --dry-run

方式 B：真实模型

    python examples/local_chat.py --model /path/to/Qwen2.5-1.5B-Instruct

方式 C：自定义参数

    python examples/local_chat.py \\
        --model /path/to/Qwen2.5-1.5B-Instruct \\
        --max-tokens 256 \\
        --temperature 0.7

与 examples/openai_client.py 的区别
-----------------------------------
- local_chat.py：直接调用 Python API（LLMEngine），无需 HTTP 服务，无网络延迟
- openai_client.py：通过 OpenAI 兼容接口，需要先启动 serve.py
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini-infer local chat example")
    parser.add_argument("--model", type=str, default="", help="模型目录路径")
    parser.add_argument("--dry-run", action="store_true", help="使用 stub model，无需真实权重")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-tokens", type=int, default=128, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（0.0 = greedy）")
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def build_engine(args: argparse.Namespace):
    """初始化 LLMEngine。"""
    from mini_infer.config import EngineConfig
    from mini_infer.engine import LLMEngine

    if not args.dry_run and not args.model:
        raise SystemExit("请指定 --model 或 --dry-run")

    config = EngineConfig(
        model_name=args.model if args.model else "dry",
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
        block_size=args.block_size,
    )
    print(f"[local_chat] 初始化引擎：model={config.model_name!r}  dry_run={config.dry_run}")
    engine = LLMEngine(config)
    return engine


def generate(engine, prompt: str, max_new_tokens: int, temperature: float) -> tuple[str, float]:
    """单次生成，返回（生成文本, 耗时）。"""
    from mini_infer.request import Request, SamplingParams

    req = Request(
        request_id=f"req-{int(time.time() * 1000)}",
        prompt=prompt,
        sampling_params=SamplingParams(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        ),
    )

    t0 = time.perf_counter()
    engine.add_request(req)

    output_tokens: list[int] = []
    while True:
        finished = engine.step()
        # 收集本步产出的 token
        if req.output_token_ids:
            output_tokens = list(req.output_token_ids)
        if req.is_finished():
            break
    elapsed = time.perf_counter() - t0

    # 解码输出
    text = ""
    if hasattr(engine, "tokenizer") and engine.tokenizer is not None:
        text = engine.tokenizer.decode(output_tokens, skip_special_tokens=True)
    elif output_tokens:
        # dry-run：直接展示 token id 列表
        text = f"[dry-run token ids] {output_tokens[:20]}{'...' if len(output_tokens) > 20 else ''}"
    return text, elapsed


def main() -> int:
    args = parse_args()
    engine = build_engine(args)

    print("\n=== mini-infer local chat ===")
    print("输入 'quit' 或 'exit' 退出，'clear' 重置对话历史\n")

    history: list[dict] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if user_input.lower() in {"quit", "exit"}:
            print("再见！")
            break

        if user_input.lower() == "clear":
            history.clear()
            print("[对话历史已清空]\n")
            continue

        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        # 构建完整 prompt（简单拼接，不做 chat template）
        prompt = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        ) + "\nASSISTANT:"

        text, elapsed = generate(engine, prompt, args.max_tokens, args.temperature)

        print(f"Assistant: {text}")
        print(f"[{elapsed:.2f}s, {len(text.split()):.0f} words]\n")

        history.append({"role": "assistant", "content": text})

    return 0


if __name__ == "__main__":
    sys.exit(main())
