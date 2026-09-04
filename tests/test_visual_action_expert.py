"""Contracts for the history-free visual action expert."""

from __future__ import annotations

import torch


def _condition(core, batch: int):
    condition = core.initial_condition(batch)
    condition.mode_id = torch.zeros(batch, dtype=torch.long)
    return condition


def test_visual_expert_logits_are_history_independent():
    """The visual expert must never receive the action-history shortcut."""
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(7)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=72).eval()
    frames = torch.randn(2, 3, 8)
    with torch.no_grad():
        without_history = core(frames, _condition(core, 2),
                               history_actions=torch.zeros(2, 72), run_slow=True)
        with_history = core(frames, _condition(core, 2),
                            history_actions=torch.ones(2, 72), run_slow=True)

    assert without_history.visual is not None
    assert with_history.visual is not None
    assert torch.equal(without_history.visual.fast.move_logits,
                       with_history.visual.fast.move_logits)
    assert torch.equal(without_history.visual.slow.intent_logits,
                       with_history.visual.slow.intent_logits)


def test_final_logits_are_visual_plus_bounded_prior():
    """Prior logits may refine but cannot replace the visual logits."""
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(11)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=72).eval()
    frames = torch.randn(2, 3, 8)
    with torch.no_grad():
        output = core(frames, _condition(core, 2),
                      history_actions=torch.randn(2, 72), run_slow=True)

    assert output.visual is not None
    assert output.prior_fast is not None
    assert output.prior_slow is not None
    assert torch.allclose(output.fast.move_logits,
                          output.visual.fast.move_logits + 0.5 * output.prior_fast.move_logits)
    assert torch.allclose(output.fast.camera_dx_logits,
                          output.visual.fast.camera_dx_logits + 0.5 * output.prior_fast.camera_dx_logits)
    assert torch.allclose(output.slow.intent_logits,
                          output.visual.slow.intent_logits + 0.5 * output.prior_slow.intent_logits)


def test_visual_expert_uses_first_to_last_change():
    """A last-frame-identical sequence with a different first frame is visible."""
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(13)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=72).eval()
    last = torch.randn(1, 8)
    baseline = torch.stack((torch.zeros_like(last), last, last), dim=1)
    changed = baseline.clone()
    changed[:, 0] = 2.0
    with torch.no_grad():
        first = core(baseline, _condition(core, 1), history_actions=torch.zeros(1, 72), run_slow=False)
        second = core(changed, _condition(core, 1), history_actions=torch.zeros(1, 72), run_slow=False)

    assert not torch.allclose(first.visual.feature, second.visual.feature)


def test_visual_losses_are_explicit_and_weighted_separately():
    """A fused loss alone is insufficient: direct visual logits need gradients."""
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.training.vla_loss import VLALossWeights, compute_vla_loss

    torch.manual_seed(17)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=72)
    output = core(torch.randn(2, 3, 8), _condition(core, 2),
                  history_actions=torch.randn(2, 72), run_slow=True)
    batch = {
        "fast_loss_mask": torch.ones(2),
        "move_target": torch.zeros(2, 4, dtype=torch.long),
        "camera_dx_target": torch.full((2, 4), 2, dtype=torch.long),
        "camera_dy_target": torch.full((2, 4), 2, dtype=torch.long),
        "button_target": torch.zeros(2, 4, 6),
        "duration_target": torch.full((2, 4), 6.0),
        "slow_loss_mask": torch.ones(2),
        "intent_target": torch.full((2,), 4, dtype=torch.long),
        "subgoal_target": torch.ones(2, dtype=torch.long),
        "subgoal_weight": torch.ones(2),
    }
    fused_only = compute_vla_loss(output.fast, output.slow, batch,
                                  weights=VLALossWeights(visual_aux=0.0),
                                  visual_fast=output.visual.fast,
                                  visual_slow=output.visual.slow)
    supervised = compute_vla_loss(output.fast, output.slow, batch,
                                  weights=VLALossWeights(visual_aux=1.0),
                                  visual_fast=output.visual.fast,
                                  visual_slow=output.visual.slow)

    visual_total = (
        supervised["visual_fast_move"]
        + supervised["visual_fast_camera"]
        + supervised["visual_fast_buttons"]
        + 0.25 * supervised["visual_fast_duration"]
        + supervised["visual_slow_intent"]
        + 0.5 * supervised["visual_slow_subgoal"]
    )
    assert torch.allclose(supervised["total"] - fused_only["total"], visual_total)
