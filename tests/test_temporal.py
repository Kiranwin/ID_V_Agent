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
