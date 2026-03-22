"""One-command quick chat entrypoint for mini-infer."""

from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Quick chat entrypoint for mini-infer",
        add_help=True,
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use a real local model instead of the temporary dry-run server",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="override the local model path used by --real",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device used by the temporary real-model server",
    )
    return parser.parse_known_args(argv)


def resolve_real_model_path(args: argparse.Namespace) -> str:
    path = args.model_path or os.getenv("MODEL") or DEFAULT_MODEL_PATH
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        raise SystemExit(
            "找不到本地模型目录。\n"
            f"默认查找路径：{DEFAULT_MODEL_PATH}\n"
            "可用以下方式之一：\n"
            "  python quick_chat.py --real --model-path /path/to/Qwen2.5-7B-Instruct\n"
            "  MODEL=/path/to/Qwen2.5-7B-Instruct python quick_chat.py --real"
        )
    return path


def main() -> int:
    args, rest = parse_args(sys.argv[1:])
    from mini_infer.clients.chat_client import main as chat_main

    if args.real or args.model_path:
        argv = [
            "--quick-model-path",
            resolve_real_model_path(args),
            "--device",
            args.device,
            "--max-tokens",
            "64",
            *rest,
        ]
    elif rest:
        argv = rest
    else:
        argv = ["--quick-dry-run", "--max-tokens", "64"]
    return chat_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
