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


def test_class_balance_clamps_and_normalizes_each_head_independently():
    import torch
    from idv_agent.training.vla_loss import balanced_class_weights

    weights = balanced_class_weights(torch.tensor([10_000.0, 1.0, 0.0]))
    observed = weights[:2]
    assert torch.all(observed >= 0.35)
    assert torch.all(observed <= 3.0)
    assert torch.isclose(observed.mean(), torch.tensor(1.0), atol=1e-5)
    assert weights[0] < weights[1]
    assert weights[2] == 1.0  # absent classes do not affect normalization


def test_class_balance_rejects_wrong_global_histogram_shapes():
    import torch
    from idv_agent.training.vla_loss import VLALossWeights, compute_vla_loss
    from idv_agent.model.vla_heads import FastVLAOutput

    fast = FastVLAOutput(
        move_logits=torch.randn(1, 1, 9), camera_dx_logits=torch.randn(1, 1, 5),
        camera_dy_logits=torch.randn(1, 1, 5), button_logits=torch.randn(1, 1, 6),
        duration=torch.ones(1, 1), confidence=torch.ones(1), stop_or_replan=torch.ones(1),
        intent_context_logits=torch.randn(1, 8),
    )
    batch = {"move_target": torch.zeros(1, 1, dtype=torch.long),
             "camera_dx_target": torch.zeros(1, 1, dtype=torch.long),
             "camera_dy_target": torch.zeros(1, 1, dtype=torch.long),
             "button_target": torch.zeros(1, 1, 6), "duration_target": torch.ones(1, 1),
             "fast_loss_mask": torch.ones(1), "slow_loss_mask": torch.zeros(1),
             "intent_target": torch.tensor([-100]), "subgoal_target": torch.tensor([-100]),
             "subgoal_weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="move_global_counts"):
        compute_vla_loss(fast, None, batch, VLALossWeights(
            class_balance=True, move_global_counts=torch.ones(3)))


def test_class_balance_uses_global_move_histogram_not_batch_histogram():
    import torch
    from idv_agent.training.vla_loss import VLALossWeights, compute_vla_loss
    from idv_agent.model.vla_heads import FastVLAOutput

    logits = torch.zeros(1, 1, 9)
    logits[0, 0, 1] = 2.0
    fast = FastVLAOutput(
        move_logits=logits, camera_dx_logits=torch.zeros(1, 1, 5),
        camera_dy_logits=torch.zeros(1, 1, 5), button_logits=torch.zeros(1, 1, 6),
        duration=torch.ones(1, 1), confidence=torch.ones(1), stop_or_replan=torch.ones(1),
        intent_context_logits=torch.zeros(1, 8),
    )
    batch = {"move_target": torch.tensor([[1]]), "camera_dx_target": torch.zeros(1, 1, dtype=torch.long),
             "camera_dy_target": torch.zeros(1, 1, dtype=torch.long), "button_target": torch.zeros(1, 1, 6),
             "duration_target": torch.ones(1, 1), "fast_loss_mask": torch.ones(1),
             "slow_loss_mask": torch.zeros(1), "intent_target": torch.tensor([-100]),
             "subgoal_target": torch.tensor([-100]), "subgoal_weight": torch.zeros(1)}
    global_loss = compute_vla_loss(fast, None, batch, VLALossWeights(
        class_balance=True, move_global_counts=torch.tensor([1000., 1.] + [0.] * 7)))['fast_move']
    batch_loss = compute_vla_loss(fast, None, batch, VLALossWeights(
        class_balance=True, move_global_counts=torch.tensor([0., 1.] + [0.] * 7)))['fast_move']
    assert global_loss > batch_loss


def test_evaluation_metrics_include_recalls_and_zero_bucket_false_turn_rate():
    from idv_agent.scripts.train_vla import _classification_metrics

    metrics = _classification_metrics(
        [[2, 0, 0, 0, 0], [0, 1, 0, 0, 0], [1, 0, 3, 0, 0],
         [0, 0, 0, 2, 0], [0, 0, 0, 0, 1]],
        zero_class=2,
    )
    assert metrics["recall"][0] == 1.0
    assert metrics["zero_false_turn_rate"] == 1 / 4


def test_interact_event_metrics_report_precision_recall_and_logits():
    import torch
    from idv_agent.scripts.train_vla import _interact_event_metrics

    metrics = _interact_event_metrics(
        torch.tensor([[2.0, -1.0], [-2.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.ones(2, 2, dtype=torch.bool),
    )
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["per_step_target_positive"] == [1.0, 1.0]


def test_interact_event_threshold_fits_separated_scores_with_fp_limit():
    import torch
    from idv_agent.scripts.train_vla import _fit_interact_event_threshold

    threshold, report = _fit_interact_event_threshold(
        torch.tensor([-8.0, -7.0, -6.0, -1.0, -0.5]),
        torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]),
        max_false_positive_rate=0.15,
    )
    assert -6.0 < threshold < -1.0
    assert report["false_positive_rate"] == 0.0
    assert report["recall"] == 1.0


def test_interact_event_threshold_can_use_strict_event_fpr():
    import torch
    from idv_agent.scripts.train_vla import _fit_interact_event_threshold

    threshold, report = _fit_interact_event_threshold(
        torch.tensor([-8.0, -7.0, -6.0, -3.0, -1.0]),
        torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]),
        max_false_positive_rate=0.02,
    )
    assert threshold > -3.0
    assert report["false_positive_rate"] == 0.0
    assert report["recall"] == 1.0


def test_ordinary_button_bias_init_returns_only_five_held_channels():
    import torch
    from idv_agent.scripts.train_vla import _ordinary_button_bias_init

    dataset = [
        {"button_target": torch.tensor([
            [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)},
        {"button_target": torch.zeros(4, 6)},
    ]
    bias = _ordinary_button_bias_init(dataset)
    assert bias.shape == (4, 5)
    assert bias[0, 0].item() == pytest.approx(0.0)
    assert bias[1, 0].item() < -8.0


def test_interact_event_metrics_uses_explicit_threshold():
    import torch
    from idv_agent.scripts.train_vla import _interact_event_metrics

    metrics = _interact_event_metrics(
        torch.tensor([[-2.0, -5.0], [-4.0, -5.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        torch.ones(2, 2, dtype=torch.bool),
        threshold=-3.0,
    )
    assert metrics["threshold_logit"] == -3.0
    assert metrics["tp"] == 1
    assert metrics["fp"] == 0


def test_evaluate_event_threshold_supports_batched_future_steps(monkeypatch):
    import torch
    from types import SimpleNamespace
    import idv_agent.scripts.train_vla as train_vla

    fast = SimpleNamespace(
        move_logits=torch.zeros(2, 2, 9), camera_dx_logits=torch.zeros(2, 2, 5),
        camera_dy_logits=torch.zeros(2, 2, 5), button_logits=torch.zeros(2, 2, 6),
        interact_event_logits=torch.tensor([[-2.0, -3.0], [-4.0, -5.0]]),
    )
    slow = SimpleNamespace(intent_logits=torch.zeros(2, 8))
    output = SimpleNamespace(fast=fast, slow=slow)
    batch = {
        "fast_loss_mask": torch.ones(2), "move_target": torch.zeros(2, 2, dtype=torch.long),
        "camera_dx_target": torch.zeros(2, 2, dtype=torch.long),
        "camera_dy_target": torch.zeros(2, 2, dtype=torch.long),
        "button_target": torch.zeros(2, 2, 6), "intent_target": torch.zeros(2, dtype=torch.long),
    }
    monkeypatch.setattr(train_vla, "forward_loss", lambda *args, **kwargs: ({"total": torch.tensor(1.)}, output))
    monkeypatch.setattr(train_vla, "_slow_accuracy", lambda *args, **kwargs: 1.0)
    result = train_vla._evaluate(torch.nn.Module(), torch.nn.Module(), [batch],
                                 torch.device("cpu"), False, event_threshold=-3.0)
    assert result["interact_pred_positive"] == 2


def test_apply_interact_event_threshold_reuses_evaluation_without_reencoding():
    import torch
    from idv_agent.scripts.train_vla import _apply_interact_event_threshold

    result = {
        "button_pred_positive_counts": [99, 3, 0, 0, 0, 0],
        "button_target_positive_counts": [2, 3, 0, 0, 0, 0],
        "interact_pred_positive": 99,
        "interact_target_positive": 2,
    }
    updated = _apply_interact_event_threshold(
        result,
        torch.tensor([[-2.0, -4.0], [-5.0, -6.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        threshold=-3.0,
    )
    assert updated["button_pred_positive_counts"] == [1, 3, 0, 0, 0, 0]
    assert updated["interact_pred_positive"] == 1
    assert updated["interact_target_positive"] == 1
    assert updated["interact_event"]["recall"] == 1.0


def test_balance_acceptance_requires_baseline_and_enforces_zero_turn_limit():
    from idv_agent.scripts.train_vla import _balance_acceptance

    current = {"classification": {
        "move": {"recall": [0.90] + [None] * 8},
        "camera_dx": {"recall": [0.70, 0.70, 0.90, 0.70, 0.70], "zero_false_turn_rate": 0.10},
        "camera_dy": {"recall": [0.70, 0.70, 0.90, 0.70, 0.70], "zero_false_turn_rate": 0.10},
        "intent": {"recall": [0.70, None, None, None, 0.90, None, None, None]},
    }}
    assert _balance_acceptance(current)["status"] == "not_comparable"
    baseline = {"classification": {
        "move": {"recall": [0.90] + [None] * 8},
        "camera_dx": {"recall": [0.60, 0.60, 0.90, 0.60, 0.60], "zero_false_turn_rate": 0.10},
        "camera_dy": {"recall": [0.60, 0.60, 0.90, 0.60, 0.60], "zero_false_turn_rate": 0.10},
        "intent": {"recall": [0.60, None, None, None, 0.90, None, None, None]},
    }}
    accepted = _balance_acceptance(current, baseline)
    assert accepted["status"] == "pass"
    too_active = {**current, "classification": {**current["classification"],
        "camera_dx": {**current["classification"]["camera_dx"], "zero_false_turn_rate": 0.16}}}
    assert _balance_acceptance(too_active, baseline)["status"] == "fail"


def test_class_balance_manifest_records_sampled_data_provenance(tmp_path):
    import torch
    from idv_agent.scripts.train_vla import _class_balance_manifest

    source = tmp_path / "vla_chunks_v5.jsonl"
    source.write_text("", encoding="utf-8")
    manifest = _class_balance_manifest(
        {"move": torch.ones(9), "camera_dx": torch.ones(5),
         "camera_dy": torch.ones(5), "intent": torch.ones(8)},
        data_paths=[str(source)], sample_count=17, sampling="stratified", max_samples=17,
    )
    assert manifest["source"]["sample_count"] == 17
    assert manifest["source"]["sampling"] == "stratified"
    assert manifest["source"]["paths"] == [str(source.resolve())]


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


def test_interact_stage_counts_group_predictions_by_intent_and_subgoal():
    import torch
    from idv_agent.scripts.train_vla import _interact_stage_counts

    predicted = torch.tensor([[[5.0] * 6] * 4, [[-5.0] * 6] * 4])
    target = torch.tensor([[[1, 0, 0, 0, 0, 0]] * 4, [[0, 0, 0, 0, 0, 0]] * 4], dtype=torch.float32)
    intent = torch.tensor([4, 0])  # travel, decipher
    subgoal = torch.tensor([16, 2])
    result = _interact_stage_counts(predicted, target, intent, subgoal)
    assert result["travel/move_to_target"] == {"pred": 4, "target": 4}
    assert result["decipher/start_decoding"] == {"pred": 0, "target": 0}


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


def test_act_training_no_longer_requires_m2_checkpoint(monkeypatch):
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
    with pytest.raises(ValueError, match="VLA JSONL 不存在"):
        train(args)
    assert called == [123]


def test_event_centered_merge_removes_periodic_future_overlap():
    from idv_agent.scripts.prepare_mvp_data import _merge_event_centered_rows

    def row(start, mode):
        return {"episode_id": "ep", "anchor_frame": start - 2,
                "alignment": {"action_start_frame": start,
                               "action_end_frame": start + 23},
                "auxiliary": {"sampling_mode": mode}}

    merged = _merge_event_centered_rows(
        [row(23, "periodic"), row(47, "periodic"), row(71, "periodic")],
        [row(40, "event_centered")],
    )

    assert [item["auxiliary"]["sampling_mode"] for item in merged] == [
        "event_centered", "periodic"
    ]
    assert [item["alignment"]["action_start_frame"] for item in merged] == [40, 71]


def test_event_centered_merge_drops_periodic_interact_when_event_window_unavailable():
    from idv_agent.scripts.prepare_mvp_data import _merge_event_centered_rows

    periodic = {
        "episode_id": "ep", "anchor_frame": 50,
        "alignment": {"action_start_frame": 52, "action_end_frame": 75},
        "action_chunk": [
            {"buttons": [0, 0, 0, 0, 0, 0]},
            {"buttons": [0, 0, 0, 0, 0, 0]},
            {"buttons": [0, 0, 0, 0, 0, 0]},
            {"buttons": [1, 0, 0, 0, 0, 0]},
        ],
        "auxiliary": {"sampling_mode": "periodic"},
    }

    assert _merge_event_centered_rows([periodic], []) == []


def test_interact_event_bias_init_returns_only_future_event_logits():
    import torch
    from idv_agent.scripts.train_vla import _interact_event_bias_init

    dataset = [
        {"button_target": torch.tensor([
            [1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
        ], dtype=torch.float32)},
        {"button_target": torch.zeros(4, 6)},
    ]

    prior = _interact_event_bias_init(dataset)
    assert prior.shape == (4,)
    assert prior[0].item() == pytest.approx(0.0)
    assert prior[1].item() < -8.0
