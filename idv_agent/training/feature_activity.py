"""Pure acceptance metrics for short VLA training and spatial activity."""

from __future__ import annotations

from typing import Any

import torch


def _finite(values: list[float]) -> bool:
    return bool(values) and all(torch.isfinite(torch.tensor(values)))


def training_behavior_gate(metrics: dict[str, Any], *, min_steps: int = 30,
                           max_steps: int = 60, max_gradient_norm: float = 100.0,
                           max_component_loss: float = 100.0) -> dict[str, Any]:
    """Check the short-run training invariants without hiding missing data."""
    losses = [float(value) for value in metrics.get("loss_history", [])]
    gradients = [float(value) for value in metrics.get("gradient_norm_history", [])]
    components = metrics.get("component_loss_history", [])
    failures: list[str] = []
    steps = int(metrics.get("steps", 0))
    if not min_steps <= steps <= max_steps:
        failures.append("steps_out_of_range")
    if len(losses) < 2 or not _finite(losses):
        failures.append("loss_nonfinite")
    elif not losses[-1] < losses[0]:
        failures.append("loss_not_decreasing")
    if not gradients or not _finite(gradients):
        failures.append("gradient_nonfinite")
    elif max(gradients) > max_gradient_norm:
        failures.append("gradient_explosion")
    flat_components = [float(value) for row in components for value in row.values()]
    if not flat_components or not _finite(flat_components):
        failures.append("component_loss_nonfinite")
    elif max(flat_components) > max_component_loss:
        failures.append("component_loss_out_of_range")
    return {"pass": not failures, "failures": failures,
            "rules": {"steps": [min_steps, max_steps],
                      "loss_final_lt_initial": True,
                      "max_gradient_norm": max_gradient_norm,
                      "max_component_loss": max_component_loss,
                      "finite_component_losses": True}}


def _cosine_mean(left: torch.Tensor, right: torch.Tensor) -> float:
    values = torch.nn.functional.cosine_similarity(left.float(), right.float(), dim=-1)
    return float(values.mean())


def summarize_feature_activity(normal: torch.Tensor, repeated: torch.Tensor,
                               left: torch.Tensor, right: torch.Tensor,
                               normal_outputs: torch.Tensor,
                               left_outputs: torch.Tensor,
                               right_outputs: torch.Tensor) -> dict[str, float]:
    """Summarize deterministic, cross-scene and left/right spatial signals."""
    normal = normal.reshape(normal.shape[0], -1)
    repeated = repeated.reshape(repeated.shape[0], -1)
    left = left.reshape(left.shape[0], -1)
    right = right.reshape(right.shape[0], -1)
    if normal.shape != repeated.shape or left.shape != right.shape or normal.shape != left.shape:
        raise ValueError("feature activity 的样本数和维度必须一致")
    if normal.shape[0] < 2:
        raise ValueError("feature activity 至少需要两个场景样本")
    pairwise = []
    for i in range(normal.shape[0]):
        for j in range(i + 1, normal.shape[0]):
            pairwise.append(torch.nn.functional.cosine_similarity(normal[i:i + 1], normal[j:j + 1], dim=-1)[0])
    pairwise_cos = torch.stack(pairwise)
    left_right_cos = torch.nn.functional.cosine_similarity(left, right, dim=-1)
    feature_l2 = (left - right).norm(dim=-1)
    output_l1 = (left_outputs.float().reshape(left.shape[0], -1) -
                 right_outputs.float().reshape(left.shape[0], -1)).abs().mean(dim=-1)
    return {
        "same_frame_cosine_mean": _cosine_mean(normal, repeated),
        "different_scene_cosine_mean": float(pairwise_cos.mean()),
        "left_right_cosine_mean": float(left_right_cos.mean()),
        "left_right_feature_l2_mean": float(feature_l2.mean()),
        "left_right_output_l1_mean": float(output_l1.mean()),
        "samples": float(normal.shape[0]),
    }


def feature_activity_gate(report: dict[str, Any], *, same_frame_min: float = 0.999,
                          different_scene_max: float = 0.99,
                          min_spatial_feature_l2: float = 1e-4,
                          min_spatial_output_l1: float = 1e-4) -> dict[str, Any]:
    """Require k=8 signal to survive the temporal decision path."""
    failures: list[str] = []
    same = report.get("same_frame_cosine_mean")
    different = report.get("different_scene_cosine_mean")
    feature_l2 = report.get("left_right_feature_l2_mean")
    output_l1 = report.get("left_right_output_l1_mean")
    values = {"same_frame_cosine_mean": same, "different_scene_cosine_mean": different,
              "left_right_feature_l2_mean": feature_l2, "left_right_output_l1_mean": output_l1}
    for name, value in values.items():
        if value is None or not torch.isfinite(torch.tensor(float(value))):
            failures.append(f"{name}_nonfinite")
    if same is not None and torch.isfinite(torch.tensor(float(same))) and float(same) < same_frame_min:
        failures.append("same_frame_not_deterministic")
    if different is not None and torch.isfinite(torch.tensor(float(different))) and float(different) >= different_scene_max:
        failures.append("different_scene_collapsed")
    if feature_l2 is not None and torch.isfinite(torch.tensor(float(feature_l2))) and float(feature_l2) <= min_spatial_feature_l2:
        failures.append("left_right_feature_no_change")
    if output_l1 is not None and torch.isfinite(torch.tensor(float(output_l1))) and float(output_l1) <= min_spatial_output_l1:
        failures.append("left_right_output_no_change")
    return {"pass": not failures, "failures": failures,
            "rules": {"same_frame_cosine_min": same_frame_min,
                      "different_scene_cosine_max": different_scene_max,
                      "min_spatial_feature_l2": min_spatial_feature_l2,
                      "min_spatial_output_l1": min_spatial_output_l1}}
