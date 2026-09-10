"""Masked multi-task losses for M29 predicted-state hierarchy."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from idv_agent.training.prompt_q_loss import prompt_q_loss


@dataclass(frozen=True)
class M29LossWeights:
    navigation: float = 1.0
    phase: float = 0.25
    facts: float = 0.25
    route: float = 0.25
    q: float = 1.0


def masked_ce(logits, target, mask, weight=None):
    mask = mask.bool()
    return F.cross_entropy(logits[mask], target[mask].long(), weight=weight) if bool(mask.any()) else logits.sum()*0


def compute_m29_loss(output, targets, masks, weights=M29LossWeights(), class_weights=None):
    class_weights = class_weights or {}
    facts, decision, nav = output.facts, output.decision, output.navigation
    losses = {"q": prompt_q_loss(output.q_logits, targets["q"], masks["q"],
                                  pos_weight=class_weights.get("q")),
              "decoding": prompt_q_loss(facts.decoding_logits, targets["decoding"], masks["decoding"],
                                         pos_weight=class_weights.get("decoding")),
              "visibility": masked_ce(facts.visibility_logits, targets["visibility"], masks["visibility"], class_weights.get("visibility")),
              "phase": masked_ce(decision.phase_logits, targets["phase"], masks["phase"], class_weights.get("phase")),
              "steering": masked_ce(decision.steering_logits, targets["steering"], masks["steering"], class_weights.get("steering")),
              "path": masked_ce(decision.path_logits, targets["path"], masks["path"], class_weights.get("path")),
              "move": masked_ce(nav.move_logits, targets["move"], masks["navigation"], class_weights.get("move")),
              "camera_dx": masked_ce(nav.camera_dx_logits, targets["camera_dx"], masks["navigation"], class_weights.get("camera_dx")),
              "camera_dy": masked_ce(nav.camera_dy_logits, targets["camera_dy"], masks["navigation"], class_weights.get("camera_dy"))}
    mask = masks["bbox"].bool()
    losses["bbox"] = F.smooth_l1_loss(facts.bbox[mask], targets["bbox"][mask]) if bool(mask.any()) else facts.bbox.sum()*0
    prompt_mask = masks["prompt_bbox"].bool()
    losses["prompt_bbox"] = (F.smooth_l1_loss(facts.prompt_bbox[prompt_mask], targets["prompt_bbox"][prompt_mask])
                             if bool(prompt_mask.any()) else facts.prompt_bbox.sum()*0)
    losses["total"] = (weights.q*losses["q"] + weights.navigation*(losses["move"]+losses["camera_dx"]+losses["camera_dy"])/3
                       + weights.phase*losses["phase"] + weights.route*(losses["steering"]+losses["path"])/2
                       + weights.facts*(losses["visibility"]+losses["decoding"]+losses["bbox"]+losses["prompt_bbox"])/4)
    return losses
