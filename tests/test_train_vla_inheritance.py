from __future__ import annotations

import pytest


def test_act_requires_m2_checkpoint(monkeypatch):
    from argparse import Namespace
    from idv_agent.scripts.train_vla import train

    args = Namespace(
        data="missing.jsonl", model_path="base", init_checkpoint="",
        allow_base_init=False, device="cpu", batch_size=1, steps=1,
        max_samples=1, temporal_dim=16, lr=1e-3, checkpoint="tmp/x.pt",
    )
    with pytest.raises(ValueError, match="init-checkpoint M2_VG"):
        train(args)
