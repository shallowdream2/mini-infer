"""Interactive chat client for the local mini-infer HTTP server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive chat client for mini-infer")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", type=str, default="mini-infer")
    parser.add_argument("--system", type=str, default="你是一个简洁、专业的中文助手。")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--no-stream", action="store_true", help="disable SSE streaming output")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="respect HTTP(S)_PROXY env vars even for localhost",
    )
    return parser.parse_args()


def _build_opener(base_url: str, use_env_proxy: bool) -> urllib.request.OpenerDirector:
    if use_env_proxy:
        return urllib.request.build_opener()
    host = urllib.parse.urlparse(base_url).hostname
    if host in {"127.0.0.1", "localhost"}:
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post_json(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc


def _stream_chat(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[str, str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    parts: list[str] = []
    finish_reason = "stop"
    try:
        with opener.open(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                choice = chunk["choices"][0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                if content:
                    print(content, end="", flush=True)
                    parts.append(content)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc
    print()
    return "".join(parts), finish_reason


def _non_stream_chat(
    opener: urllib.request.OpenerDirector,
    url: str,
    payload: dict[str, Any],
    timeout: float,
) -> tuple[str, str]:
    response = _post_json(opener, url, payload, timeout)
    choice = response["choices"][0]
    text = choice["message"]["content"]
    print(text)
    return text, choice.get("finish_reason") or "stop"


def _print_help() -> None:
    print("commands: /clear reset history, /history show history, /help show help, exit quit")


def main() -> int:
    args = parse_args()
    opener = _build_opener(args.base_url, args.use_env_proxy)
    chat_url = args.base_url.rstrip("/") + "/chat/completions"
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print(f"mini-infer chat -> {args.base_url}  model={args.model}")
    _print_help()

    while True:
        try:
            user_text = input("\n你: ").strip()
        except EOFError:
            print()
            return 0

        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            return 0
        if user_text == "/help":
            _print_help()
            continue
        if user_text == "/clear":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("[history cleared]")
            continue
        if user_text == "/history":
            for message in messages:
                print(f"{message['role']}: {message['content']}")
            continue

        messages.append({"role": "user", "content": user_text})
        payload = {
            "model": args.model,
            "messages": messages,
            "stream": not args.no_stream,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        }

        print("模型: ", end="", flush=True)
        try:
            if args.no_stream:
                answer, _ = _non_stream_chat(opener, chat_url, payload, args.timeout)
            else:
                answer, _ = _stream_chat(opener, chat_url, payload, args.timeout)
        except KeyboardInterrupt:
            print("\n[interrupted]")
            messages.pop()
            continue
        except Exception as exc:  # noqa: BLE001 - CLI should show concise failure.
            print(f"\n[error] {exc}")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    raise SystemExit(main())
