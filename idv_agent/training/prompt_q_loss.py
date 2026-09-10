"""Supervise the actual Q-token logit from current-frame prompt annotations."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def prompt_q_loss(q_logits: torch.Tensor, targets: torch.Tensor,
                  known_mask: torch.Tensor, *, pos_weight: torch.Tensor | None = None) -> torch.Tensor:
    """One current Q logit per image; mask unknowns independently of navigation.

    Callers pass the logit used for emitting Q, not a disconnected auxiliary
    classifier. Replay timing, phase, previous Q and decoding outcomes do not
    enter this loss. This is a v6 component, not wired into legacy v5 training.
    """
    if q_logits.ndim != 1 or targets.shape != q_logits.shape or known_mask.shape != q_logits.shape:
        raise ValueError("Q logits/targets/known_mask must all be [B]")
    mask = known_mask.to(device=q_logits.device, dtype=torch.bool)
    selected = targets.to(q_logits)[mask]
    if not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError("known prompt-Q targets must be 0/1")
    if not bool(mask.any()):
        return q_logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(q_logits[mask], selected, pos_weight=pos_weight)
