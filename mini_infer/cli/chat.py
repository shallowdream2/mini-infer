"""mini_infer.cli.chat — console script: mini-infer-chat

等价于：python quick_chat.py [args]

用法：
  mini-infer-chat                           # dry-run 模式，自动启动临时服务
  mini-infer-chat --real --model-path /path/to/model  # 真实模型
"""
from __future__ import annotations


def main() -> None:
    import sys
    # Delegate to quick_chat.main()
    # quick_chat.py is installed as a top-level script; import via importlib
    import importlib.util
    import pathlib

    # Resolve quick_chat.py relative to this package's install root
    pkg_root = pathlib.Path(__file__).parent.parent.parent
    spec = importlib.util.spec_from_file_location("quick_chat", pkg_root / "quick_chat.py")
    if spec is None or spec.loader is None:
        raise SystemExit("quick_chat.py not found. Run mini-infer-chat from the project root.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    sys.exit(module.main())


if __name__ == "__main__":
    main()
