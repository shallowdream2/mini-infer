def pytest_addoption(parser):
    parser.addoption("--model", default=None, help="真实模型路径，用于 real_model 测试")
