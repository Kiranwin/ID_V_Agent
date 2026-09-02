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
    move_stop_weight: float = 0.25
    button_positive_weight: float = 4.0


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
    move_class_weight[0] = weights.move_stop_weight
    move = F.cross_entropy(fast.move_logits.transpose(1, 2), batch["move_target"],
                           weight=move_class_weight, reduction="none")
    cam_dx = F.cross_entropy(fast.camera_dx_logits.transpose(1, 2), batch["camera_dx_target"], reduction="none")
    cam_dy = F.cross_entropy(fast.camera_dy_logits.transpose(1, 2), batch["camera_dy_target"], reduction="none")
    button_target = batch["button_target"].to(fast.button_logits.dtype)
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
        intent = F.cross_entropy(slow.intent_logits, intent_target, ignore_index=-100, reduction="none")
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
