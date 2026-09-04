from __future__ import annotations

import torch


def test_training_behavior_gate_requires_decreasing_finite_loss_and_gradients():
    from idv_agent.training.feature_activity import training_behavior_gate

    report = training_behavior_gate({
        "steps": 30,
        "loss_history": [4.0, 3.0, 2.0],
        "component_loss_history": [{"fast_move": 2.0, "fast_camera": 1.0}],
        "gradient_norm_history": [0.8, 0.7],
    })
    assert report["pass"] is True
    failed = training_behavior_gate({
        "steps": 30,
        "loss_history": [4.0, float("nan"), 4.1],
        "component_loss_history": [{"fast_move": 2.0}],
        "gradient_norm_history": [0.8],
    })
    assert failed["pass"] is False
    assert "loss_nonfinite" in failed["failures"]
    oversized = training_behavior_gate({
        "steps": 30, "loss_history": [4.0, 3.0],
        "component_loss_history": [{"fast_move": 101.0}],
        "gradient_norm_history": [0.8],
    })
    assert "component_loss_out_of_range" in oversized["failures"]


def test_feature_activity_gate_rejects_cosine_collapse_and_accepts_spatial_signal():
    from idv_agent.training.feature_activity import feature_activity_gate, summarize_feature_activity

    normal = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    repeated = normal.clone()
    left = torch.tensor([[1., 0.1, 0.], [0., 1., 0.1], [0.1, 0., 1.]])
    right = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    outputs = torch.zeros(3, 4)
    left_outputs = torch.ones(3, 4)
    right_outputs = torch.full((3, 4), 2.)
    report = summarize_feature_activity(normal, repeated, left, right,
                                        outputs, left_outputs, right_outputs)
    assert report["same_frame_cosine_mean"] == 1.0
    assert report["different_scene_cosine_mean"] < 0.99
    assert report["left_right_output_l1_mean"] > 0
    assert feature_activity_gate(report)["pass"] is True

    collapsed = summarize_feature_activity(torch.ones(3, 3), torch.ones(3, 3),
                                           torch.ones(3, 3), torch.ones(3, 3),
                                           outputs, outputs, outputs)
    assert feature_activity_gate(collapsed)["pass"] is False


def test_feature_activity_gate_requires_repeat_determinism():
    from idv_agent.training.feature_activity import feature_activity_gate

    report = {"same_frame_cosine_mean": 0.98, "different_scene_cosine_mean": 0.5,
              "left_right_cosine_mean": 0.5, "left_right_output_l1_mean": 0.2}
    result = feature_activity_gate(report)
    assert result["pass"] is False
    assert "same_frame_not_deterministic" in result["failures"]


def test_left_right_occlusion_masks_only_the_requested_half():
    from PIL import Image
    from idv_agent.scripts.evaluate_feature_activity import _occlusion

    image = Image.new("RGB", (8, 4), (10, 20, 30))
    left = _occlusion("left")(image)
    right = _occlusion("right")(image)
    assert left.getpixel((1, 1)) == (0, 0, 0)
    assert left.getpixel((6, 1)) == (10, 20, 30)
    assert right.getpixel((1, 1)) == (10, 20, 30)
    assert right.getpixel((6, 1)) == (0, 0, 0)
