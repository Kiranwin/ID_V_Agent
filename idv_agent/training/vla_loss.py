"""Joint supervised losses for the shared fast/slow VLA."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F

from idv_agent.model.vla_heads import FastVLAOutput, GroundingOutput, SlowVLAOutput


@dataclass(frozen=True)
class VLALossWeights:
    move: float = 1.0
    camera: float = 1.0
    buttons: float = 1.0
    duration: float = 0.25
    slow_intent: float = 1.0
    slow_subgoal: float = 0.5
    consistency: float = 0.1
    # Supervise the history-free visual action expert with the same head
    # weights and masks as the fused deployment outputs.
    visual_aux: float = 1.0
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
    interact_event_positive_weight: float = 3.0
    grounding: float = 1.0
    # Grounding annotations are sparse and their binary targets have different
    # base rates (for example prompt/reachable).  These tables are calculated
    # only from annotated training frames and normalized independently per
    # head, so they cannot change the relative lambda between grounding heads.
    grounding_class_balance: bool = False
    grounding_present_global_counts: Optional[torch.Tensor] = None
    grounding_side_global_counts: Optional[torch.Tensor] = None
    grounding_prompt_global_counts: Optional[torch.Tensor] = None
    grounding_reachable_global_counts: Optional[torch.Tensor] = None


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


def _binary_target_weights(target: torch.Tensor, counts: torch.Tensor | None,
                           *, name: str) -> torch.Tensor | None:
    """Return negative/positive class weights selected per binary target."""
    if counts is None:
        return None
    checked = _validated_global_counts(counts, 2, name, target.new_zeros(2))
    table = balanced_class_weights(checked).to(device=target.device, dtype=target.dtype)
    return table[(target > 0).to(torch.long)]


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


def compute_grounding_loss(grounding: GroundingOutput, batch: dict,
                           weights: VLALossWeights = VLALossWeights()) -> dict[str, torch.Tensor]:
    """Current-frame visual grounding loss, masked outside the annotation pool."""
    mask = batch["grounding_mask"].to(grounding.present_logits.device)
    bbox_mask = batch["grounding_bbox_mask"].to(grounding.bbox.device) * mask
    present_target = batch["grounding_present_target"].to(grounding.present_logits.dtype)
    prompt_target = batch["grounding_prompt_target"].to(grounding.prompt_logits.dtype)
    reachable_target = batch["grounding_reachable_target"].to(grounding.reachable_logits.dtype)
    present_weight = prompt_weight = reachable_weight = side_weight = None
    if weights.grounding_class_balance:
        present_weight = _binary_target_weights(
            present_target, weights.grounding_present_global_counts,
            name="grounding_present_global_counts")
        prompt_weight = _binary_target_weights(
            prompt_target, weights.grounding_prompt_global_counts,
            name="grounding_prompt_global_counts")
        reachable_weight = _binary_target_weights(
            reachable_target, weights.grounding_reachable_global_counts,
            name="grounding_reachable_global_counts")
        side_counts = _validated_global_counts(
            weights.grounding_side_global_counts, grounding.side_logits.shape[-1],
            "grounding_side_global_counts", torch.bincount(
                batch["grounding_side_target"].reshape(-1), minlength=grounding.side_logits.shape[-1]))
        side_weight = balanced_class_weights(side_counts).to(
            device=grounding.side_logits.device, dtype=grounding.side_logits.dtype)
    present = F.binary_cross_entropy_with_logits(
        grounding.present_logits, present_target, weight=present_weight, reduction="none")
    prompt = F.binary_cross_entropy_with_logits(
        grounding.prompt_logits, prompt_target, weight=prompt_weight, reduction="none")
    reachable = F.binary_cross_entropy_with_logits(
        grounding.reachable_logits, reachable_target, weight=reachable_weight, reduction="none")
    side = F.cross_entropy(grounding.side_logits, batch["grounding_side_target"].to(torch.long),
                           weight=side_weight, reduction="none")
    bbox = F.smooth_l1_loss(grounding.bbox, batch["grounding_bbox_target"].to(grounding.bbox.dtype), reduction="none").mean(dim=-1)
    losses = {
        "grounding_present": _masked_mean(present, mask),
        "grounding_bbox": _masked_mean(bbox, bbox_mask),
        "grounding_side": _masked_mean(side, mask),
        "grounding_prompt": _masked_mean(prompt, mask),
        "grounding_reachable": _masked_mean(reachable, mask),
    }
    return {"grounding_total": sum(losses.values()), **losses}


def compute_vla_loss(fast: FastVLAOutput, slow: Optional[SlowVLAOutput],
                     batch: dict, weights: VLALossWeights = VLALossWeights(), *,
                     visual_fast: Optional[FastVLAOutput] = None,
                     visual_slow: Optional[SlowVLAOutput] = None,
                     grounding: Optional[GroundingOutput] = None) -> dict[str, torch.Tensor]:
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
    event_target = button_target[..., 0]
    ordinary_button_target = button_target[..., 1:]
    if weights.button_global_counts is not None:
        counts = weights.button_global_counts.to(device=fast.button_logits.device,
                                                 dtype=fast.button_logits.dtype)
        if counts.ndim != 2 or counts.shape[0] != fast.button_logits.shape[-1] or counts.shape[1] != 2:
            raise ValueError("button_global_counts 必须是 [num_buttons, 2]，列为 negative/positive")
        pos_weight = (counts[:, 0] / counts[:, 1].clamp_min(1.0)).clamp_min(1e-3)
        positive_weight = torch.where(ordinary_button_target > 0, pos_weight[1:],
                                      torch.ones_like(ordinary_button_target))
    else:
        positive_weight = torch.where(ordinary_button_target > 0,
                                      torch.as_tensor(weights.button_positive_weight, device=fast.button_logits.device,
                                                      dtype=fast.button_logits.dtype),
                                      torch.ones((), device=fast.button_logits.device,
                                                 dtype=fast.button_logits.dtype))
    buttons = F.binary_cross_entropy_with_logits(fast.button_logits[..., 1:], ordinary_button_target,
                                                 weight=positive_weight, reduction="none")
    event_logits = (fast.interact_event_logits if fast.interact_event_logits is not None
                    else fast.button_logits[..., 0])
    event_weight = torch.where(
        event_target > 0,
        torch.as_tensor(weights.interact_event_positive_weight,
                        device=event_logits.device, dtype=event_logits.dtype),
        torch.ones_like(event_target),
    )
    interact_event = F.binary_cross_entropy_with_logits(
        event_logits, event_target, weight=event_weight, reduction="none")
    duration = F.smooth_l1_loss(fast.duration, batch["duration_target"], reduction="none")
    losses = {
        "fast_move": _masked_mean(move, fast_mask),
        "fast_camera": _masked_mean(cam_dx + cam_dy, fast_mask),
        "fast_buttons": _masked_mean(buttons, fast_mask),
        "fast_interact_event": _masked_mean(interact_event, fast_mask),
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
        + weights.buttons * losses["fast_interact_event"]
        + weights.duration * losses["fast_duration"]
        + weights.slow_intent * losses["slow_intent"]
        + weights.slow_subgoal * losses["slow_subgoal"]
        + weights.consistency * losses["consistency"]
    )
    if (visual_fast is None) != (visual_slow is None):
        raise ValueError("visual_fast 与 visual_slow 必须同时提供或同时省略")
    if visual_fast is not None:
        # Reuse the exact class weights and masking rules for the visual-only
        # branch.  The recursive call has no visual outputs, so it terminates;
        # fast/slow consistency belongs to the fused decision only.
        visual_base = compute_vla_loss(
            visual_fast, visual_slow, batch,
            weights=replace(weights, consistency=0.0, visual_aux=0.0),
        )
        names = ("fast_move", "fast_camera", "fast_buttons", "fast_interact_event", "fast_duration",
                 "slow_intent", "slow_subgoal")
        for name in names:
            losses[f"visual_{name}"] = visual_base[name]
        visual_total = visual_base["total"]
        total = total + weights.visual_aux * visual_total
    else:
        zero = fast.move_logits.sum() * 0.0
        for name in ("fast_move", "fast_camera", "fast_buttons", "fast_interact_event", "fast_duration",
                     "slow_intent", "slow_subgoal"):
            losses[f"visual_{name}"] = zero
    if grounding is not None:
        grounding_losses = compute_grounding_loss(grounding, batch, weights)
        losses.update(grounding_losses)
        total = total + weights.grounding * grounding_losses["grounding_total"]
    else:
        zero = fast.move_logits.sum() * 0.0
        losses.update({"grounding_total": zero, "grounding_present": zero, "grounding_bbox": zero,
                       "grounding_side": zero, "grounding_prompt": zero, "grounding_reachable": zero})
    return {"total": total, **losses}
