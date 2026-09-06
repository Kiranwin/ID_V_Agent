"""Training-only same-frame visual-grounding contracts for ACT."""

from __future__ import annotations

import json

import pytest
import torch


def _annotation(*, session: str, frame: int, bbox, side: str, prompt: int, reachable: int) -> dict:
    return {
        "schema_version": "act.grounding_auxiliary.v1", "id": f"{session}_{frame:08d}",
        "split": "train", "session": session, "frame": frame,
        "cipher_bbox_xyxy_norm": bbox, "cipher_reachable": reachable,
        "target_side": side, "interact_prompt": prompt,
    }


def test_grounding_index_resolves_observation_end_frame_not_future_action(tmp_path):
    from idv_agent.training.vla_dataset import GroundingAnnotationIndex

    path = tmp_path / "grounding.jsonl"
    path.write_text("\n".join((
        json.dumps(_annotation(session="ep", frame=21, bbox=[.1, .2, .5, .9],
                               side="left", prompt=0, reachable=0)),
        json.dumps(_annotation(session="ep", frame=23, bbox=[.4, .2, .6, .9],
                               side="center", prompt=1, reachable=1)),
    )) + "\n", encoding="utf-8")

    index = GroundingAnnotationIndex(path)
    row = index.lookup("ep", 21)
    assert row["grounding_mask"].item() == 1
    assert row["grounding_side_target"].item() == 1  # left
    assert torch.allclose(row["grounding_bbox_target"], torch.tensor([.1, .2, .5, .9]))
    assert index.lookup("ep", 22)["grounding_mask"].item() == 0


def test_grounding_index_rejects_prompt_without_cipher_box(tmp_path):
    from idv_agent.training.vla_dataset import GroundingAnnotationIndex

    path = tmp_path / "invalid.jsonl"
    path.write_text(json.dumps(_annotation(session="ep", frame=21, bbox=None,
                                           side="none", prompt=1, reachable=1)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="interact_prompt"):
        GroundingAnnotationIndex(path)


def test_visual_expert_grounding_heads_have_direct_visual_gradients():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.training.vla_loss import VLALossWeights, compute_grounding_loss

    torch.manual_seed(9)
    core = SharedFastSlowVLA(8, temporal_dim=12)
    condition = core.initial_condition(2)
    output = core(torch.randn(2, 3, 8), condition, run_slow=False)
    assert output.visual is not None and output.visual.grounding is not None
    batch = {
        "grounding_mask": torch.tensor([1., 0.]),
        "grounding_present_target": torch.tensor([1., 0.]),
        "grounding_bbox_mask": torch.tensor([1., 0.]),
        "grounding_bbox_target": torch.tensor([[.1, .2, .7, .8], [0., 0., 0., 0.]]),
        "grounding_side_target": torch.tensor([2, 0]),
        "grounding_prompt_target": torch.tensor([1., 0.]),
        "grounding_reachable_target": torch.tensor([1., 0.]),
    }
    losses = compute_grounding_loss(output.visual.grounding, batch, VLALossWeights(grounding=1.0))
    assert torch.isfinite(losses["grounding_total"])
    losses["grounding_total"].backward()
    assert core.visual_expert.grounding_side.weight.grad is not None
    assert torch.isfinite(core.visual_expert.grounding_side.weight.grad).all()


def test_grounding_loss_ignores_unannotated_rows():
    from idv_agent.model.vla_heads import GroundingOutput
    from idv_agent.training.vla_loss import VLALossWeights, compute_grounding_loss

    out = GroundingOutput(torch.tensor([1., 99.], requires_grad=True),
                          torch.zeros(2, 4, requires_grad=True),
                          torch.zeros(2, 4, requires_grad=True),
                          torch.tensor([1., 99.], requires_grad=True),
                          torch.tensor([1., 99.], requires_grad=True))
    batch = {
        "grounding_mask": torch.tensor([1., 0.]), "grounding_present_target": torch.tensor([1., 0.]),
        "grounding_bbox_mask": torch.tensor([0., 0.]), "grounding_bbox_target": torch.zeros(2, 4),
        "grounding_side_target": torch.tensor([0, 0]), "grounding_prompt_target": torch.tensor([1., 0.]),
        "grounding_reachable_target": torch.tensor([1., 0.]),
    }
    first = compute_grounding_loss(out, batch, VLALossWeights())
    changed = GroundingOutput(out.present_logits[:1].detach().clone().requires_grad_().repeat(2),
                              out.bbox[:1].detach().clone().requires_grad_().repeat(2, 1),
                              out.side_logits[:1].detach().clone().requires_grad_().repeat(2, 1),
                              out.prompt_logits[:1].detach().clone().requires_grad_().repeat(2),
                              out.reachable_logits[:1].detach().clone().requires_grad_().repeat(2))
    second = compute_grounding_loss(changed, batch, VLALossWeights())
    assert torch.allclose(first["grounding_total"], second["grounding_total"])


def test_vla_dataset_attaches_grounding_only_at_observation_end_frame(tmp_path):
    from idv_agent.training.vla_dataset import VLASequenceDataset

    session = tmp_path / "raw" / "ep"
    (session / "frames").mkdir(parents=True)
    frames = tuple(range(0, 22, 3))
    for frame in frames:
        (session / "frames" / f"{frame:08d}.jpg").write_bytes(b"x")
    record = {
        "schema_version": "vla.action_chunk.v5", "episode_id": "ep", "anchor_frame": 21,
        "source_root": str(session), "mode": "standard",
        "task": {"instruction": "find", "mode_token": "<mode:standard>"},
        "observations": {"frames": [{"path": f"frames/{frame:08d}.jpg", "frame_index": frame,
                                        "timestamp_ns": frame,
                                        "slow_label": {"valid": True, "segment_id": "s", "intent": "travel",
                                                       "subgoal": "move_to_target", "subgoal_source": "rule",
                                                       "subgoal_rule_version": "subgoal.v1"}} for frame in frames],
                             "history_actions": [{"move_dir": 0, "camera_dx": 0, "camera_dy": 0,
                                                  "camera_dx_px": 0., "camera_dy_px": 0.,
                                                  "buttons": [0] * 6} for _ in range(8)], "fps": 30.0},
            "action_chunk": [{"move_dir": 0, "camera_dx": 0, "camera_dy": 0,
                              "camera_dx_px": 0., "camera_dy_px": 0.,
                              "buttons": [0] * 6, "duration_frames": 6}] * 4,
        "alignment": {"mode": "causal_future", "action_delay_frames": 1,
                      "observation_end_frame": 21, "action_start_frame": 23,
                      "action_end_frame": 46, "observation_end_timestamp_ns": 21,
                      "action_start_timestamp_ns": 23, "action_end_timestamp_ns": 46},
            "slow_label": {"valid": True, "intent": "travel", "subgoal": "move_to_target",
                               "subgoal_source": "rule", "subgoal_rule_version": "subgoal.v1"},
        "quality": {"source": "human", "outcome": "success"},
        "loss_mask": {"slow": 1, "fast": 1},
    }
    data = tmp_path / "train.jsonl"
    data.write_text(json.dumps(record) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.jsonl"
    labels.write_text(json.dumps(_annotation(session="ep", frame=21, bbox=[.1, .2, .5, .9],
                                             side="left", prompt=0, reachable=0)) + "\n", encoding="utf-8")
    item = VLASequenceDataset(data, grounding_annotations=labels)[0]
    assert item["grounding_mask"].item() == 1
    assert item["grounding_side_target"].item() == 1


def test_model_inputs_carries_grounding_tensors_to_training_loss_device():
    from idv_agent.scripts.train_vla import _model_inputs

    batch = {
        "mode_id": torch.zeros(1, dtype=torch.long), "intent_target": torch.zeros(1, dtype=torch.long),
        "subgoal_target": torch.zeros(1, dtype=torch.long), "subgoal_weight": torch.ones(1),
        "move_target": torch.zeros(1, 4, dtype=torch.long), "camera_dx_target": torch.zeros(1, 4, dtype=torch.long),
        "camera_dy_target": torch.zeros(1, 4, dtype=torch.long), "button_target": torch.zeros(1, 4, 6),
        "duration_target": torch.ones(1, 4), "slow_loss_mask": torch.ones(1), "fast_loss_mask": torch.ones(1),
        "frame_valid_mask": torch.ones(1, 8, dtype=torch.bool), "history_actions": torch.zeros(1, 72),
        "grounding_mask": torch.ones(1), "grounding_present_target": torch.ones(1),
        "grounding_bbox_mask": torch.ones(1), "grounding_bbox_target": torch.zeros(1, 4),
        "grounding_side_target": torch.zeros(1, dtype=torch.long), "grounding_prompt_target": torch.zeros(1),
        "grounding_reachable_target": torch.zeros(1),
    }
    result = _model_inputs(batch, torch.device("cpu"))
    assert set(key for key in batch if key.startswith("grounding_")) <= set(result)
    assert result["grounding_bbox_target"].shape == (1, 4)


def test_grounding_metrics_score_only_annotated_rows():
    from idv_agent.scripts.train_vla import _grounding_metrics

    metrics = _grounding_metrics(
        present_logits=torch.tensor([10., -10., 99.]),
        bbox=torch.tensor([[.1, .1, .5, .5], [0., 0., 0., 0.], [1., 1., 1., 1.]]),
        side_logits=torch.tensor([[0., 10., 0., 0.], [10., 0., 0., 0.], [0., 0., 0., 99.]]),
        prompt_logits=torch.tensor([-10., -10., 99.]),
        reachable_logits=torch.tensor([-10., -10., 99.]),
        mask=torch.tensor([1., 1., 0.]), present_target=torch.tensor([1., 0., 0.]),
        bbox_mask=torch.tensor([1., 0., 0.]), bbox_target=torch.tensor([[.1, .1, .5, .5], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
        side_target=torch.tensor([1, 0, 0]), prompt_target=torch.tensor([0., 0., 0.]),
        reachable_target=torch.tensor([0., 0., 0.]),
    )
    assert metrics["annotated"] == 2
    assert metrics["presence"]["accuracy"] == 1.0
    assert metrics["side_accuracy"] == 1.0
    assert metrics["bbox_l1"] == 0.0
