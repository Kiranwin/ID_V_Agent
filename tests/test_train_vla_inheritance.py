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


def test_direction_balance_loss_is_finite():
    import torch
    from idv_agent.model.vla_heads import FastVLAOutput
    from idv_agent.training.vla_loss import VLALossWeights, compute_vla_loss
    fast = FastVLAOutput(
        move_logits=torch.randn(1, 4, 9), camera_dx_logits=torch.randn(1, 4, 5),
        camera_dy_logits=torch.randn(1, 4, 5), button_logits=torch.randn(1, 4, 6),
        duration=torch.ones(1, 4), confidence=torch.ones(1), stop_or_replan=torch.ones(1),
        intent_context_logits=torch.randn(1, 8))
    batch = {"move_target": torch.tensor([[0, 1, 8, 0]]), "camera_dx_target": torch.zeros(1, 4, dtype=torch.long),
             "camera_dy_target": torch.zeros(1, 4, dtype=torch.long), "button_target": torch.zeros(1, 4, 6),
             "duration_target": torch.ones(1, 4), "fast_loss_mask": torch.ones(1),
             "slow_loss_mask": torch.zeros(1), "intent_target": torch.tensor([-100]),
             "subgoal_target": torch.tensor([-100]), "subgoal_weight": torch.zeros(1)}
    assert torch.isfinite(compute_vla_loss(fast, None, batch,
                                           VLALossWeights(move_direction_balance=True))["total"])


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
