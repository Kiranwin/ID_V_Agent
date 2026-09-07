from __future__ import annotations

import pytest
import torch


def test_fit_zero_bias_prefers_nonzero_recall_with_zero_turn_constraint():
    from idv_agent.scripts.diagnose_camera_zero_calibration import _fit_zero_bias

    logits = torch.tensor([
        [1., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0.],
        [0., 1.5, 0., 0., 0.],
        [0., 1.5, 0., 0., 0.],
    ])
    targets = torch.tensor([2, 2, 0, 1])
    bias, report = _fit_zero_bias(logits, targets, max_zero_false_turn_rate=0.15)
    assert bias > 1.0
    assert report["selected"]["zero_false_turn_rate"] == 0.0


def test_fit_zero_bias_rejects_invalid_shapes_and_impossible_limit():
    from idv_agent.scripts.diagnose_camera_zero_calibration import _fit_zero_bias

    with pytest.raises(ValueError, match=r"\[N,5\]"):
        _fit_zero_bias(torch.zeros(2, 4), torch.zeros(2))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        _fit_zero_bias(torch.zeros(2, 5), torch.tensor([0, 1]), max_zero_false_turn_rate=1.1)
