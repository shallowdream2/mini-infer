"""mini_infer.cli.demo — console script: mini-infer-demo

等价于：python demo.py [args]

用法：
  mini-infer-demo --model /path/to/Qwen2.5-1.5B-Instruct --mode quant
  mini-infer-demo --model /path/to/Qwen2.5-1.5B-Instruct --mode cuda-graph
  mini-infer-demo --model /path/to/Qwen2.5-1.5B-Instruct --mode prefix-cache
  mini-infer-demo --model /path/to/Qwen2.5-1.5B-Instruct --mode all
"""
from __future__ import annotations


def main() -> None:
    import sys
    import importlib.util
    import pathlib

    pkg_root = pathlib.Path(__file__).parent.parent.parent
    spec = importlib.util.spec_from_file_location("demo", pkg_root / "demo.py")
    if spec is None or spec.loader is None:
        raise SystemExit("demo.py not found. Run mini-infer-demo from the project root.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    sys.exit(module.main())


if __name__ == "__main__":
    main()
