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
    assert torch.equal(output.fast.move_logits, output.visual.fast.move_logits)
    assert torch.allclose(
        output.fast.camera_dx_logits,
        output.visual.fast.camera_dx_logits + core.camera_prior_scale *
        torch.tanh(output.prior_fast.camera_dx_logits),
    )
    assert torch.equal(output.slow.intent_logits, output.visual.slow.intent_logits)


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
        + supervised["visual_fast_interact_event"]
        + 0.25 * supervised["visual_fast_duration"]
        + supervised["visual_slow_intent"]
        + 0.5 * supervised["visual_slow_subgoal"]
    )
    assert torch.allclose(supervised["total"] - fused_only["total"], visual_total)


def test_visual_expert_can_use_pure_visual_features_separate_from_conditioned_prior():
    """Task condition must not be part of the history-free visual branch."""
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(23)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=0).eval()
    pure = torch.randn(2, 3, 8)
    conditioned_a = pure + 3.0
    conditioned_b = pure - 7.0
    condition = _condition(core, 2)
    with torch.no_grad():
        first = core(conditioned_a, condition, visual_frame_features=pure, run_slow=False)
        second = core(conditioned_b, condition, visual_frame_features=pure, run_slow=False)
    assert torch.equal(first.visual.fast.move_logits, second.visual.fast.move_logits)
    assert torch.equal(first.visual.slow.intent_logits, second.visual.slow.intent_logits)


def test_visual_expert_changes_when_only_pure_visual_features_change():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(29)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=0).eval()
    conditioned = torch.randn(2, 3, 8)
    pure_a = torch.randn(2, 3, 8)
    pure_b = pure_a + 2.0
    condition = _condition(core, 2)
    with torch.no_grad():
        first = core(conditioned, condition, visual_frame_features=pure_a, run_slow=False)
        second = core(conditioned, condition, visual_frame_features=pure_b, run_slow=False)
    assert not torch.equal(first.visual.fast.move_logits, second.visual.fast.move_logits)


def test_visual_expert_uses_middle_frames_not_only_first_and_last():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(31)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=0).eval()
    frames = torch.randn(2, 8, 8)
    changed = frames.clone()
    changed[:, 3] += 2.0
    condition = _condition(core, 2)
    with torch.no_grad():
        first = core(frames, condition, visual_frame_features=frames, run_slow=False)
        second = core(frames, condition, visual_frame_features=changed, run_slow=False)
    assert not torch.equal(first.visual.fast.move_logits, second.visual.fast.move_logits)


def test_prior_correction_is_hard_bounded_when_fused_with_visual_logits():
    from idv_agent.model.fast_slow_vla import fuse_fast_outputs
    from idv_agent.model.vla_heads import FastVLAOutput

    shape = (1, 4, 3)
    visual = FastVLAOutput(
        move_logits=torch.zeros(shape), camera_dx_logits=torch.zeros(shape),
        camera_dy_logits=torch.zeros(shape), button_logits=torch.zeros((1, 4, 6)),
        duration=torch.full((1, 4), 6.0), confidence=torch.full((1,), 0.5),
        stop_or_replan=torch.full((1,), 0.5), intent_context_logits=torch.zeros((1, 8)),
    )
    prior = FastVLAOutput(
        move_logits=torch.full(shape, 1000.0), camera_dx_logits=torch.full(shape, -1000.0),
        camera_dy_logits=torch.full(shape, 1000.0), button_logits=torch.full((1, 4, 6), -1000.0),
        duration=torch.full((1, 4), 30.0), confidence=torch.ones((1,)),
        stop_or_replan=torch.zeros((1,)), intent_context_logits=torch.full((1, 8), 1000.0),
    )
    fused = fuse_fast_outputs(visual, prior, prior_scale=0.1, bound_prior=True)
    assert float(fused.move_logits.abs().max()) <= 0.100001
    assert float(fused.camera_dx_logits.abs().max()) <= 0.100001
    assert float(fused.button_logits.abs().max()) <= 0.100001


def test_visual_expert_has_no_constant_class_bias_and_normalizes_visual_pair():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(8, 12)
    assert expert.input_norm.elementwise_affine is False
    assert expert.fast.move.bias is None
    assert expert.slow.intent.bias is None
    zero = torch.zeros(2, 8)
    output = expert(torch.stack((zero, zero), dim=1))
    assert torch.equal(output.fast.move_logits[0], torch.zeros_like(output.fast.move_logits[0]))
    assert torch.equal(output.slow.intent_logits[0], torch.zeros_like(output.slow.intent_logits[0]))


def test_visual_expert_exposes_interact_event_as_independent_head():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(2, 4).eval()
    assert expert.fast.interact_event is not expert.fast.buttons
    with torch.no_grad():
        output = expert(torch.zeros(1, 2, 2))
    assert output.fast.interact_event_logits.shape == (1, 4)
    assert torch.equal(output.fast.button_logits[..., 0],
                       output.fast.interact_event_logits)


def test_visual_expert_keeps_ordinary_buttons_on_an_independent_negative_baseline():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(2, 4).eval()
    bias = torch.full((4, 5), -4.0)
    expert.set_ordinary_button_bias(bias)
    with torch.no_grad():
        output = expert(torch.zeros(1, 2, 2))
    assert torch.allclose(output.fast.button_logits[0, :, 1:], bias)


def test_fast_output_decodes_q_with_calibrated_threshold_only():
    from idv_agent.model.vla_heads import FastVLAOutput

    output = FastVLAOutput(
        move_logits=torch.zeros(1, 1, 2), camera_dx_logits=torch.zeros(1, 1, 2),
        camera_dy_logits=torch.zeros(1, 1, 2),
        button_logits=torch.zeros(1, 1, 6), duration=torch.ones(1, 1),
        confidence=torch.zeros(1), stop_or_replan=torch.zeros(1),
        intent_context_logits=torch.zeros(1, 8),
        interact_event_logits=torch.tensor([[-2.0]]),
    )
    assert not bool(output.button_predictions(event_threshold=-1.0)[0, 0, 0])
    assert bool(output.button_predictions(event_threshold=-3.0)[0, 0, 0])


def test_interact_event_loss_is_separate_from_ordinary_button_loss():
    from idv_agent.model.vla_heads import FastVLAOutput
    from idv_agent.training.vla_loss import VLALossWeights, compute_vla_loss

    shape = (1, 2)
    base = FastVLAOutput(
        move_logits=torch.zeros(1, 2, 2), camera_dx_logits=torch.zeros(1, 2, 2),
        camera_dy_logits=torch.zeros(1, 2, 2), button_logits=torch.zeros(1, 2, 6),
        interact_event_logits=torch.zeros(shape), duration=torch.ones(shape),
        confidence=torch.zeros(1), stop_or_replan=torch.zeros(1),
        intent_context_logits=torch.zeros(1, 8),
    )
    batch = {
        "fast_loss_mask": torch.ones(1), "move_target": torch.zeros(1, 2, dtype=torch.long),
        "camera_dx_target": torch.zeros(1, 2, dtype=torch.long),
        "camera_dy_target": torch.zeros(1, 2, dtype=torch.long),
        "button_target": torch.tensor([[[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]], dtype=torch.float32),
        "duration_target": torch.ones(1, 2),
    }
    changed = FastVLAOutput(**{**base.__dict__, "button_logits": base.button_logits.clone()})
    changed.interact_event_logits = torch.full(shape, 100.0)
    first = compute_vla_loss(base, None, batch, VLALossWeights())
    second = compute_vla_loss(changed, None, batch, VLALossWeights())
    assert torch.equal(first["fast_buttons"], second["fast_buttons"])
    assert not torch.equal(first["fast_interact_event"], second["fast_interact_event"])


def test_visual_expert_class_heads_read_normalized_raw_pair_without_hidden_bottleneck():
    """The held-out raw-grid linear signal must not be compressed before logits."""
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(8, 12)
    assert expert.fast.move.in_features == 72
    assert expert.slow.intent.in_features == 72
    assert isinstance(expert.fast.trunk, torch.nn.Identity)


def test_deployment_logits_are_visual_only_and_prior_is_diagnostic():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    torch.manual_seed(41)
    core = SharedFastSlowVLA(8, temporal_dim=12, history_action_dim=72).eval()
    frames = torch.randn(2, 3, 8)
    with torch.no_grad():
        output = core(frames, _condition(core, 2), history_actions=torch.randn(2, 72), run_slow=True)
    assert output.visual is not None
    assert output.prior_fast is not None
    assert torch.equal(output.fast.move_logits, output.visual.fast.move_logits)
    assert torch.equal(output.fast.camera_dx_logits,
                       output.visual.fast.camera_dx_logits + core.camera_prior_scale *
                       torch.tanh(output.prior_fast.camera_dx_logits))
    assert torch.equal(output.slow.intent_logits, output.visual.slow.intent_logits)


def test_visual_expert_uses_persistent_training_center_before_normalization():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(2, 4)
    frames = torch.tensor([[[1.0, 2.0], [3.0, 6.0]]])
    raw_pair = expert.pair_features(frames)
    expert.set_input_center(raw_pair[0])
    centered = expert.normalized_pair_features(frames)

    assert expert.input_center.shape == raw_pair[0].shape
    assert torch.equal(expert.input_center, raw_pair[0])
    assert torch.equal(centered, torch.zeros_like(centered))


def test_visual_expert_keeps_large_raw_grid_forward_finite_under_autocast():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(8, 12).float()
    frames = torch.full((2, 3, 8), 1e4, dtype=torch.float32)
    with torch.autocast(device_type="cpu", dtype=torch.float16):
        output = expert(frames)
    assert bool(torch.isfinite(output.feature).all())
    assert bool(torch.isfinite(output.slow.intent_logits).all())


def test_full_sequence_visual_feature_has_explicit_stable_scale():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(8, 12)
    assert expert.visual_feature_scale == 0.25
    frames = torch.randn(1, 8, 8)
    normalized = expert.normalized_pair_features(frames)
    assert torch.allclose(expert(frames).feature, normalized * 0.25)


def test_camera_visual_head_has_independent_temporal_summary():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(8, 12)
    assert expert.camera_summary_dim == 8 * 5
    assert expert.camera_visual_dx.in_features == expert.camera_summary_dim
    assert expert.camera_visual_dy.in_features == expert.camera_summary_dim


def test_camera_visual_logits_respond_to_middle_frame_trajectory():
    from idv_agent.model.vla_heads import VisualActionExpert

    torch.manual_seed(53)
    expert = VisualActionExpert(8, 12).eval()
    frames = torch.randn(1, 8, 8)
    changed = frames.clone()
    changed[:, 3] += 4.0
    with torch.no_grad():
        first = expert(frames)
        second = expert(changed)
    assert not torch.equal(first.fast.camera_dx_logits, second.fast.camera_dx_logits)


def test_camera_visual_logits_have_a_direct_full_pair_path():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(2, 4).eval()
    assert expert.camera_visual_pair_dx.in_features == expert.pair_dim
    assert expert.camera_visual_pair_dy.in_features == expert.pair_dim
    expert.camera_visual_dx.weight.data.zero_()
    expert.camera_visual_dy.weight.data.zero_()
    first = torch.zeros(1, 8, 2)
    second = first.clone()
    second[:, -1] = 1.0
    with torch.no_grad():
        first_out = expert(first)
        second_out = expert(second)
    assert not torch.equal(first_out.fast.camera_dx_logits, second_out.fast.camera_dx_logits)


def test_camera_summary_uses_centered_scaled_visual_features():
    from idv_agent.model.vla_heads import VisualActionExpert

    expert = VisualActionExpert(8, 12)
    frames = torch.full((1, 8, 8), 1000.0)
    center = expert.pair_features(frames)[0]
    expert.set_input_center(center)
    summary = expert.camera_summary(frames)
    assert float(summary.abs().max()) == 0.0
    assert expert.camera_summary_scale == 0.25
