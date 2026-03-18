"""这个文件覆盖引擎的最小 smoke test，使用 dry_run=True 在无 GPU 环境下验证 generate 主链路的状态流转。"""

from mini_infer import EngineConfig, LLMEngine


def test_engine_generate_smoke() -> None:
    config = EngineConfig(model_name="stub-model", max_batch_size=2, block_size=4, dry_run=True)
    engine = LLMEngine(config)
    outputs = engine.generate(["hello", "world"], max_new_tokens=2)

    assert len(outputs) == 2
    assert outputs[0] == " [1] [2]"
    assert outputs[1] == " [1] [2]"
