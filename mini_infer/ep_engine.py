"""Backward-compat shim — 真实模块已移至 mini_infer.parallel.ep_engine。"""
import time  # noqa: F401 — tests access ep_engine_mod.time for monkeypatching
import torch  # noqa: F401 — tests access ep_engine_mod.torch.cuda for monkeypatching
from mini_infer.parallel.ep_engine import *  # noqa: F401, F403
from mini_infer.parallel.ep_engine import (  # explicit (including private names used in tests)
    EPEngine,
    _dump_rank_state_dicts,
    _rank_state_dict_path,
    _prepare_packed_control_plane,
)
