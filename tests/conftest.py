"""Pytest configuration shared by the test suite."""


def pytest_addoption(parser):
    parser.addoption(
        "--model",
        action="store",
        default=None,
        help="Path or HuggingFace repo id for tests marked with real_model.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_model: tests that require a real local or HuggingFace model",
    )
