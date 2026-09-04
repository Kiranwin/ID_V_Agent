"""Joint supervised losses for the shared fast/slow VLA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from idv_agent.model.vla_heads import FastVLAOutput, SlowVLAOutput


@dataclass(frozen=True)
class VLALossWeights:
    move: float = 1.0
    camera: float = 1.0
    buttons: float = 1.0
    duration: float = 0.25
    slow_intent: float = 1.0
    slow_subgoal: float = 0.5
    consistency: float = 0.1
    # Recorded trajectories contain many more stop than moving steps.  Keep
    # stop supervised, but prevent it from dominating the categorical head.
    move_stop_weight: float = 0.5
    move_direction_balance: bool = False
    class_balance: bool = False
    # Global counts are computed once from the sampled training subset.  They
    # are used for small batches where per-batch counts are statistically
    # meaningless; large batches retain the original dynamic weighting.
    move_global_counts: Optional[torch.Tensor] = None
    camera_dx_global_counts: Optional[torch.Tensor] = None
    camera_dy_global_counts: Optional[torch.Tensor] = None
    intent_global_counts: Optional[torch.Tensor] = None
    button_positive_weight: float = 4.0
    # Optional [num_buttons, 2] histogram: columns are negative/positive
    # counts.  When supplied, BCE uses neg/pos per button instead of a fixed
    # scalar, preventing a majority interaction label from collapsing to all
    # ones (or a rare key from collapsing to all zeros).
    button_global_counts: Optional[torch.Tensor] = None


def balanced_class_weights(counts: torch.Tensor, *, min_weight: float = 0.35,
                           max_weight: float = 3.0) -> torch.Tensor:
    """Return per-class weights with bounded inverse-sqrt balancing.

    Positive-count classes have mean weight one independently of other heads.
    Zero-count classes are set to one and excluded from the normalization.
    The bounded projection preserves the configured ``[min_weight, max_weight]``
    interval while keeping the mean of observed classes exactly one.
    """
    if counts.ndim != 1 or counts.numel() < 1:
        raise ValueError("class counts 必须是一维非空 Tensor")
    if not 0 < min_weight <= 1 <= max_weight:
        raise ValueError("class weight clamp 必须满足 0 < min <= 1 <= max")
    counts = counts.to(dtype=torch.float32)
    weights = torch.ones_like(counts)
    active = counts > 0
    if not bool(active.any()):
        return weights
    raw = counts[active].clamp_min(1.0).rsqrt()
    raw = raw / raw.mean().clamp_min(1e-12)
    result = raw.clone()
    free = torch.ones_like(raw, dtype=torch.bool)
    target_sum = float(raw.numel())
    while bool(free.any()):
        candidate = raw[free] * ((target_sum - float(result[~free].sum())) /
                                 raw[free].sum().clamp_min(1e-12))
        low = candidate < min_weight
        high = candidate > max_weight
        if not bool(low.any()) and not bool(high.any()):
            result[free] = candidate
            break
        free_indices = torch.nonzero(free, as_tuple=False).reshape(-1)
        if bool(low.any()):
            result[free_indices[low]] = min_weight
            free[free_indices[low]] = False
        if bool(high.any()):
            result[free_indices[high]] = max_weight
            free[free_indices[high]] = False
        if not bool(free.any()):
            break
    weights[active] = result
    return weights


def _validated_global_counts(counts: torch.Tensor | None, classes: int, name: str,
                            fallback: torch.Tensor) -> torch.Tensor:
    if counts is None:
        return fallback
    if counts.ndim != 1 or counts.numel() != classes:
        raise ValueError(f"{name} 必须是 [{classes}] Tensor")
    if bool((counts < 0).any()) or not bool(torch.isfinite(counts).all()):
        raise ValueError(f"{name} 必须包含非负有限计数")
    return counts


def _move_class_weights(counts: torch.Tensor, *, move_stop_weight: float,
                        balance: bool) -> torch.Tensor:
    """Build movement CE weights while preserving the explicit stop policy."""
    if counts.ndim != 1 or counts.numel() < 1:
        raise ValueError("move counts 必须是一维非空 Tensor")
    weights = torch.ones(counts.numel(), dtype=counts.dtype, device=counts.device)
    if balance:
        weights = counts.clamp_min(1.0).rsqrt()
        weights = weights / weights.mean().clamp_min(1e-12)
    weights[0] *= float(move_stop_weight)
    return weights


def _masked_mean(values: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    while sample_mask.ndim < values.ndim:
        sample_mask = sample_mask.unsqueeze(-1)
    sample_mask = sample_mask.to(values.dtype).expand_as(values)
    return (values * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)


def compute_vla_loss(fast: FastVLAOutput, slow: Optional[SlowVLAOutput],
                     batch: dict, weights: VLALossWeights = VLALossWeights()) -> dict[str, torch.Tensor]:
    fast_mask = batch["fast_loss_mask"].to(fast.move_logits.device)
    move_class_weight = torch.ones(fast.move_logits.shape[-1], device=fast.move_logits.device,
                                   dtype=fast.move_logits.dtype)
    if weights.class_balance:
        counts = _validated_global_counts(
            weights.move_global_counts, fast.move_logits.shape[-1], "move_global_counts",
            torch.bincount(batch["move_target"].reshape(-1), minlength=fast.move_logits.shape[-1]))
        move_class_weight = balanced_class_weights(counts).to(
            device=fast.move_logits.device, dtype=fast.move_logits.dtype)
    elif weights.move_direction_balance:
        # Inverse-square-root frequency weighting is less unstable than pure
        # inverse frequency while still preventing the dominant north class.
        # Counts are computed once from the sampled training set.  Never fall
        # back to a single-batch histogram: that made batch=1/4 runs highly
        # sensitive to whichever directions happened to be in the batch.
        if weights.move_global_counts is not None:
            counts = weights.move_global_counts.to(device=move_class_weight.device,
                                                   dtype=move_class_weight.dtype)
        else:
            # Compatibility for callers that do not provide a precomputed
            # histogram; training entry points always provide one when
            # move_direction_balance is enabled.
            counts = torch.bincount(batch["move_target"].reshape(-1), minlength=fast.move_logits.shape[-1]).to(move_class_weight.dtype)
        move_class_weight = _move_class_weights(counts, move_stop_weight=weights.move_stop_weight, balance=True)
    else:
        move_class_weight[0] = weights.move_stop_weight
    move = F.cross_entropy(fast.move_logits.transpose(1, 2), batch["move_target"],
                           weight=move_class_weight, reduction="none")
    camera_dx_weight = None
    camera_dy_weight = None
    if weights.class_balance:
        camera_dx_counts = _validated_global_counts(
            weights.camera_dx_global_counts, fast.camera_dx_logits.shape[-1],
            "camera_dx_global_counts",
            torch.bincount(batch["camera_dx_target"].reshape(-1), minlength=fast.camera_dx_logits.shape[-1]))
        camera_dy_counts = _validated_global_counts(
            weights.camera_dy_global_counts, fast.camera_dy_logits.shape[-1],
            "camera_dy_global_counts",
            torch.bincount(batch["camera_dy_target"].reshape(-1), minlength=fast.camera_dy_logits.shape[-1]))
        camera_dx_weight = balanced_class_weights(camera_dx_counts).to(
            device=fast.camera_dx_logits.device, dtype=fast.camera_dx_logits.dtype)
        camera_dy_weight = balanced_class_weights(camera_dy_counts).to(
            device=fast.camera_dy_logits.device, dtype=fast.camera_dy_logits.dtype)
    cam_dx = F.cross_entropy(fast.camera_dx_logits.transpose(1, 2), batch["camera_dx_target"],
                             weight=camera_dx_weight, reduction="none")
    cam_dy = F.cross_entropy(fast.camera_dy_logits.transpose(1, 2), batch["camera_dy_target"],
                             weight=camera_dy_weight, reduction="none")
    button_target = batch["button_target"].to(fast.button_logits.dtype)
    if weights.button_global_counts is not None:
        counts = weights.button_global_counts.to(device=fast.button_logits.device,
                                                 dtype=fast.button_logits.dtype)
        if counts.ndim != 2 or counts.shape[0] != fast.button_logits.shape[-1] or counts.shape[1] != 2:
            raise ValueError("button_global_counts 必须是 [num_buttons, 2]，列为 negative/positive")
        pos_weight = (counts[:, 0] / counts[:, 1].clamp_min(1.0)).clamp_min(1e-3)
        positive_weight = torch.where(button_target > 0, pos_weight, torch.ones_like(button_target))
    else:
        positive_weight = torch.where(button_target > 0,
                                      torch.as_tensor(weights.button_positive_weight, device=fast.button_logits.device,
                                                      dtype=fast.button_logits.dtype),
                                      torch.ones((), device=fast.button_logits.device,
                                                 dtype=fast.button_logits.dtype))
    buttons = F.binary_cross_entropy_with_logits(fast.button_logits, button_target,
                                                 weight=positive_weight, reduction="none")
    duration = F.smooth_l1_loss(fast.duration, batch["duration_target"], reduction="none")
    losses = {
        "fast_move": _masked_mean(move, fast_mask),
        "fast_camera": _masked_mean(cam_dx + cam_dy, fast_mask),
        "fast_buttons": _masked_mean(buttons, fast_mask),
        "fast_duration": _masked_mean(duration, fast_mask),
    }
    zero = fast.move_logits.sum() * 0.0
    losses.update({"slow_intent": zero, "slow_subgoal": zero, "consistency": zero})
    if slow is not None:
        slow_mask = batch["slow_loss_mask"].to(slow.intent_logits.device)
        intent_target = batch["intent_target"].to(slow.intent_logits.device)
        subgoal_target = batch["subgoal_target"].to(slow.subgoal_logits.device)
        valid = (intent_target != -100).to(slow_mask.dtype)
        supervised_mask = slow_mask * valid
        intent_weight = None
        if weights.class_balance:
            intent_counts = _validated_global_counts(
                weights.intent_global_counts, slow.intent_logits.shape[-1], "intent_global_counts",
                torch.bincount(intent_target[intent_target >= 0], minlength=slow.intent_logits.shape[-1]))
            intent_weight = balanced_class_weights(intent_counts).to(
                device=slow.intent_logits.device, dtype=slow.intent_logits.dtype)
        intent = F.cross_entropy(slow.intent_logits, intent_target, weight=intent_weight,
                                 ignore_index=-100, reduction="none")
        subgoal = F.cross_entropy(slow.subgoal_logits, subgoal_target, ignore_index=-100, reduction="none")
        subgoal_weight = batch["subgoal_weight"].to(subgoal.device)
        losses["slow_intent"] = _masked_mean(intent, supervised_mask)
        losses["slow_subgoal"] = _masked_mean(subgoal * subgoal_weight, supervised_mask)

        slow_prob = F.softmax(slow.intent_logits, dim=-1)
        fast_log_prob = F.log_softmax(fast.intent_context_logits, dim=-1)
        consistency = F.kl_div(fast_log_prob, slow_prob.detach(), reduction="none").sum(dim=-1)
        losses["consistency"] = _masked_mean(consistency, supervised_mask)

    total = (
        weights.move * losses["fast_move"]
        + weights.camera * losses["fast_camera"]
        + weights.buttons * losses["fast_buttons"]
        + weights.duration * losses["fast_duration"]
        + weights.slow_intent * losses["slow_intent"]
        + weights.slow_subgoal * losses["slow_subgoal"]
        + weights.consistency * losses["consistency"]
    )
    return {"total": total, **losses}
