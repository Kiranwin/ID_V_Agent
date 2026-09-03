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


def test_direction_balance_preserves_stop_weight():
    import torch
    from idv_agent.training.vla_loss import _move_class_weights

    counts = torch.tensor([100.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    weights = _move_class_weights(
        counts, move_stop_weight=0.5, balance=True,
    )
    # Global inverse-sqrt balancing must still apply the configured stop
    # multiplier; otherwise enabling balancing silently re-enables stop.
    assert torch.isclose(weights[0] / weights[1], torch.tensor(0.1), atol=1e-6)


def test_mvp_finds_transitional_v4_chunk_filename(tmp_path):
    from idv_agent.scripts.prepare_mvp_data import _find_v4_chunks

    session = tmp_path / "session"
    session.mkdir()
    alternate = session / "train_vla_chunks_v4.jsonl"
    alternate.write_text("", encoding="utf-8")
    assert _find_v4_chunks(session) == alternate
    canonical = session / "vla_chunks_v4.jsonl"
    canonical.write_text("", encoding="utf-8")
    assert _find_v4_chunks(session) == canonical


def test_mvp_split_keeps_train_nonempty_with_two_mixed_sessions():
    from idv_agent.scripts.prepare_mvp_data import _split

    def rows():
        return [{"slow_label": {"intent": "travel"}},
                {"slow_label": {"intent": "decipher"}}]

    train, val = _split([("s1", rows()), ("s2", rows())], 0.2, 7, None)
    assert len(train) == 1 and len(val) == 1
    assert {r["slow_label"]["intent"] for r in train[0][1]} == {"travel", "decipher"}
    assert {r["slow_label"]["intent"] for r in val[0][1]} == {"travel", "decipher"}


def test_teacher_forcing_ratio_schedule_and_prediction_fallback():
    from argparse import Namespace
    import torch
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.scripts.train_vla import _scheduled_condition, _teacher_forcing_ratio

    args = Namespace(steps=100, teacher_forcing_start=1.0,
                     teacher_forcing_end=0.0, teacher_forcing_decay_steps=100)
    assert _teacher_forcing_ratio(args, 0) == 1.0
    assert _teacher_forcing_ratio(args, 50) == 0.5
    assert _teacher_forcing_ratio(args, 100) == 0.0

    model = SharedFastSlowVLA(frame_feature_dim=8, temporal_dim=12)
    slow = model.slow_head(torch.randn(2, 12))
    batch = {"mode_id": torch.zeros(2, dtype=torch.long),
             "intent_target": torch.tensor([2, -100]),
             "subgoal_target": torch.tensor([3, -100])}
    forced = _scheduled_condition(model, slow, batch, teacher_forcing_ratio=1.0)
    predicted = _scheduled_condition(model, slow, batch, teacher_forcing_ratio=0.0)
    assert forced.intent_id[0].item() == 2 and forced.subgoal_id[0].item() == 3
    assert forced.intent_id[1].item() == slow.intent_id[1].item()
    assert torch.equal(predicted.intent_id, slow.intent_id)


def test_act_requires_m2_checkpoint(monkeypatch):
    from argparse import Namespace
    from idv_agent.scripts.train_vla import train

    args = Namespace(
        data="missing.jsonl", model_path="base", init_checkpoint="",
        allow_base_init=False, device="cpu", batch_size=1, steps=1,
        max_samples=1, temporal_dim=16, lr=1e-3, checkpoint="tmp/x.pt",
        seed=123,
    )
    called = []
    monkeypatch.setattr("idv_agent.scripts.train_vla.set_seed", lambda value: called.append(value))
    with pytest.raises(ValueError, match="init-checkpoint M2_VG"):
        train(args)
    assert called == [123]
