"""Temporal contracts that prevent a GRU gate from erasing visual features."""

from __future__ import annotations

import torch


def test_temporal_visual_residual_keeps_last_valid_frame_distinguishable():
    from idv_agent.model.temporal import SharedTemporalEncoder

    encoder = SharedTemporalEncoder(input_dim=3, hidden_dim=3)
    for parameter in encoder.gru.parameters():
        parameter.data.zero_()
    encoder.delta_proj.weight.data.zero_()
    encoder.delta_proj.bias.data.zero_()
    encoder.input_residual.weight.data.copy_(torch.eye(3))
    encoder.input_residual.bias.data.zero_()

    first = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    second = torch.tensor([[[0., 0., 0.], [0., 1., 0.]]])
    valid = torch.tensor([[True, True]])
    output_first = encoder(first, valid_mask=valid)
    output_second = encoder(second, valid_mask=valid)

    assert torch.equal(output_first, first[:, -1])
    assert torch.equal(output_second, second[:, -1])
    assert not torch.equal(output_first, output_second)


def test_fast_decision_residual_bypasses_zeroed_gru_with_current_visual_frame():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    core = SharedFastSlowVLA(frame_feature_dim=3, temporal_dim=3, history_action_dim=0)
    for parameter in core.fast_temporal.gru.parameters():
        parameter.data.zero_()
    core.fast_temporal.input_residual.weight.data.zero_()
    core.fast_temporal.input_residual.bias.data.zero_()
    core.fast_visual_residual.weight.data.copy_(torch.eye(3))
    core.fast_visual_residual.bias.data.zero_()

    first = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]])
    second = torch.tensor([[[0., 0., 0.], [0., 1., 0.]]])
    condition = core.initial_condition(1)
    valid = torch.tensor([[True, True]])

    output_first = core(first, condition, valid_mask=valid, run_slow=False)
    output_second = core(second, condition, valid_mask=valid, run_slow=False)

    assert torch.equal(output_first.prior_temporal_feature, first[:, -1])
    assert torch.equal(output_second.prior_temporal_feature, second[:, -1])
    assert not torch.equal(output_first.temporal_feature, output_second.temporal_feature)
