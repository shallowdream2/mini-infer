"""Shared pytest options for integration-style tests."""

from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register options used by real-model tests."""
    parser.addoption(
        "--model",
        action="store",
        default="",
        help="Local HuggingFace model directory for tests marked real_model.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "real_model: tests that require a local real model directory passed by --model",
    )


@pytest.fixture
def real_model_path(pytestconfig: pytest.Config) -> str:
    """Return and validate the --model path for real-model tests."""
    model_path = pytestconfig.getoption("--model")
    if not model_path:
        pytest.skip("需要通过 --model 指定本地模型目录")
    if not Path(model_path).exists():
        pytest.skip(f"模型目录不存在: {model_path}")
    return model_path
