from __future__ import annotations

import pytest


def test_stratified_chunk_sampling_keeps_action_strata():
    from idv_agent.scripts.train_vla import _action_stratum, _stratified_subset

    class TinyDataset:
        def __init__(self):
            self.rows = []
            for episode in ("e1", "e2", "e3", "e4"):
                for kind in ("interact", "move", "other_key", "stop"):
                    row = {
                        "episode_id": episode,
                        "move_target": __import__("torch").tensor([1 if kind == "move" else 0]),
                        "button_target": __import__("torch").tensor([[1, 0, 0, 0, 0, 0]] if kind == "interact" else
                                                                     [[0, 1, 0, 0, 0, 0]] if kind == "other_key" else
                                                                     [[0, 0, 0, 0, 0, 0]]),
                    }
                    self.rows.append(row)

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    dataset = TinyDataset()
    sampled = _stratified_subset(dataset, 8, seed=3)
    from collections import Counter
    counts = Counter(_action_stratum(sampled[i]) for i in range(len(sampled)))
    assert counts == {"interact": 3, "move": 3, "other_key": 1, "stop": 1}


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
