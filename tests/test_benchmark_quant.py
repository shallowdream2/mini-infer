"""Phase 16 benchmark 口径测试。"""

from __future__ import annotations

import importlib.util
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_quant.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_quant", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_quant = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_quant)


def test_compute_match_metrics_exact_match() -> None:
    metrics = benchmark_quant.compute_match_metrics(
        [[1, 2, 3], [4, 5]],
        [[1, 2, 3], [4, 5]],
    )
    assert metrics["token_match_rate"] == 1.0
    assert metrics["sequence_exact_rate"] == 1.0


def test_compute_match_metrics_counts_length_mismatch_as_mismatch() -> None:
    metrics = benchmark_quant.compute_match_metrics(
        [[1, 2, 3], [7, 8]],
        [[1, 9], [7, 8, 10]],
    )
    assert metrics["token_matches"] == 3.0
    assert metrics["token_total"] == 6.0
    assert metrics["token_match_rate"] == 0.5
    assert metrics["sequence_exact_rate"] == 0.0
