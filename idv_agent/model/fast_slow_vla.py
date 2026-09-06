"""Composable fast/slow VLA core operating on cached frame features.

Architecture change (2026-09-03): the shared temporal GRU was split into
``slow_temporal``/``fast_temporal``.  This removes competition over the
*recurrent* parameters between the slow (discrete classification) and fast
(continuous control) objectives.  It does **not** fully decouple the two
losses: when the caller passes ``detach_slow_condition=False`` (the training
default in ``train_vla.py``'s second pass), the fast loss still reaches
``slow_head``/``slow_temporal`` through the FiLM ``context_embedding`` path,
since that conditioning is applied before the temporal split.  That remaining
channel is deliberate (it is how the fast head learns to use the slow head's
context) and is controlled explicitly via ``detach_slow_condition``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from idv_agent.model.temporal import SharedTemporalEncoder
from idv_agent.model.vla_heads import (
    CONTEXT_EMBEDDING_DIM,
    FastVLAHead,
    FastVLAOutput,
    FiLMConditioner,
    SlowVLAHead,
    SlowVLAOutput,
    VisualActionExpert,
    VisualExpertOutput,
)
from idv_agent.vla.action_chunk import INTENTS
from idv_agent.configs.subgoal import SUBGOAL_NAMES


@dataclass
class SlowCondition:
    """Last valid slow output consumed by the next fast tick."""

    intent_id: torch.Tensor
    subgoal_id: torch.Tensor
    context_embedding: torch.Tensor
    mode_id: torch.Tensor
    created_timestamp_ns: Optional[int] = None

    def detached(self) -> "SlowCondition":
        return SlowCondition(
            intent_id=self.intent_id.detach(),
            subgoal_id=self.subgoal_id.detach(),
            context_embedding=self.context_embedding.detach(),
            mode_id=self.mode_id.detach(),
            created_timestamp_ns=self.created_timestamp_ns,
        )


@dataclass
class FastSlowVLAOutput:
    temporal_feature: torch.Tensor
    fast: FastVLAOutput
    slow: Optional[SlowVLAOutput]
    prior_temporal_feature: Optional[torch.Tensor] = None
    visual: Optional[VisualExpertOutput] = None
    prior_fast: Optional[FastVLAOutput] = None
    prior_slow: Optional[SlowVLAOutput] = None


def fuse_fast_outputs(visual: FastVLAOutput, prior: FastVLAOutput,
                      prior_scale: float = 0.5, *, bound_prior: bool = False) -> FastVLAOutput:
    """Bound history/temporal prior so it cannot erase visual logits."""
    scale = float(prior_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("prior_scale 必须在 [0,1]")
    prior_logits = (lambda value: torch.tanh(value) if bound_prior else value)
    visual_event = (visual.interact_event_logits if visual.interact_event_logits is not None
                    else visual.button_logits[..., 0])
    prior_event = (prior.interact_event_logits if prior.interact_event_logits is not None
                   else prior.button_logits[..., 0])
    event_logits = visual_event + scale * prior_logits(prior_event)
    button_logits = visual.button_logits + scale * prior_logits(prior.button_logits)
    button_logits = button_logits.clone()
    button_logits[..., 0] = event_logits
    return FastVLAOutput(
        move_logits=visual.move_logits + scale * prior_logits(prior.move_logits),
        camera_dx_logits=visual.camera_dx_logits + scale * prior_logits(prior.camera_dx_logits),
        camera_dy_logits=visual.camera_dy_logits + scale * prior_logits(prior.camera_dy_logits),
        button_logits=button_logits,
        duration=(visual.duration + scale * (prior.duration - 6.0)).clamp(1.0, 30.0),
        confidence=(visual.confidence + scale * (prior.confidence - 0.5)).clamp(0.0, 1.0),
        stop_or_replan=(visual.stop_or_replan + scale * (prior.stop_or_replan - 0.5)).clamp(0.0, 1.0),
        intent_context_logits=visual.intent_context_logits + scale * prior_logits(prior.intent_context_logits),
        interact_event_logits=event_logits,
    )


def fuse_slow_outputs(visual: SlowVLAOutput, prior: SlowVLAOutput,
                      prior_scale: float = 0.5, *, bound_prior: bool = False) -> SlowVLAOutput:
    """Fuse tactical logits while retaining visual evidence as the base path."""
    scale = float(prior_scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("prior_scale 必须在 [0,1]")
    prior_logits = (lambda value: torch.tanh(value) if bound_prior else value)
    return SlowVLAOutput(
        intent_logits=visual.intent_logits + scale * prior_logits(prior.intent_logits),
        subgoal_logits=visual.subgoal_logits + scale * prior_logits(prior.subgoal_logits),
        context_embedding=visual.context_embedding + scale * prior.context_embedding,
        refresh_logits=visual.refresh_logits + scale * prior_logits(prior.refresh_logits),
    )


class SharedFastSlowVLA(nn.Module):
    """Separated temporal encoders plus slow and fast heads.

    Architecture change (2026-09-03): slow and fast heads now use independent
    GRU temporal encoders instead of one shared GRU, so the two objectives no
    longer compete for the same recurrent weights.  This increases parameters
    by ~1.2M (per ``temporal_dim``-sized GRU; negligible next to the frozen
    vision-language backbone).  The FiLM conditioning path (``context_embedding``
    -> ``FiLMConditioner``) still runs *before* the slow/fast split, so it
    remains a shared, intentional coupling point: with
    ``detach_slow_condition=False`` the fast loss can still update
    ``slow_head``/``slow_temporal`` through that path.  See
    ``docs/18-架构变更历史.md`` for the measured effect and remaining caveats.

    The visual backbone deliberately remains outside this module.  Runtime
    supplies cached per-frame features, ensuring the two heads cannot trigger
    duplicate image encoding.
    """

    def __init__(self, frame_feature_dim: int, temporal_dim: int = 512,
                 history_action_dim: int = 0):
        super().__init__()
        self.conditioner = FiLMConditioner(frame_feature_dim)
        self.slow_temporal = SharedTemporalEncoder(frame_feature_dim, temporal_dim)
        self.fast_temporal = SharedTemporalEncoder(frame_feature_dim, temporal_dim)
        # A direct current-frame path keeps visual evidence available to the
        # decision heads even when FiLM/GRU dynamics learn a strong prior.
        # This path is intentionally fed from the unconditioned frame feature
        # and is added after each temporal encoder.
        self.slow_visual_residual = nn.Linear(frame_feature_dim, temporal_dim)
        self.fast_visual_residual = nn.Linear(frame_feature_dim, temporal_dim)
        self.slow_head = SlowVLAHead(temporal_dim)
        self.fast_head = FastVLAHead(temporal_dim, history_action_dim=history_action_dim)
        self.visual_expert = VisualActionExpert(frame_feature_dim, temporal_dim)
        self.prior_scale = 0.1
        self.camera_prior_scale = 0.5
        # ACT deployment consumes the directly supervised visual branch.  The
        # temporal/history branch remains exposed for diagnostics, but does
        # not form a second copy of the deployed logits.
        self.visual_only_deployment = True

    @staticmethod
    def initial_condition(batch_size: int, device: torch.device | str = "cpu",
                          mode_id: int = 0, intent_id: int = 7,
                          subgoal_id: int | None = 1) -> SlowCondition:
        if batch_size < 1:
            raise ValueError("batch_size 必须为正数")
        if not 0 <= intent_id < len(INTENTS):
            raise ValueError("intent_id 超出范围")
        if subgoal_id is None:
            subgoal_id = SUBGOAL_NAMES.index("observe")
        return SlowCondition(
            intent_id=torch.full((batch_size,), intent_id, dtype=torch.long, device=device),
            subgoal_id=torch.full((batch_size,), subgoal_id, dtype=torch.long, device=device),
            context_embedding=torch.zeros((batch_size, CONTEXT_EMBEDDING_DIM), device=device),
            mode_id=torch.full((batch_size,), mode_id, dtype=torch.long, device=device),
        )

    def forward(
        self,
        frame_features: torch.Tensor,
        slow_condition: SlowCondition,
        *,
        valid_mask: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
        history_actions: Optional[torch.Tensor] = None,
        visual_frame_features: Optional[torch.Tensor] = None,
        run_slow: bool = False,
        detach_slow_condition: bool = True,
    ) -> FastSlowVLAOutput:
        """Forward pass with separated temporal encoders.

        ``detach_slow_condition`` still controls a real gradient path: FiLM
        conditioning happens before the slow/fast split, so an attached
        ``context_embedding`` lets the fast loss update ``slow_temporal``/
        ``slow_head`` through the conditioner even though the two GRUs are
        otherwise independent.
        """
        if frame_features.ndim == 2:
            frame_features = frame_features.unsqueeze(0)
        if visual_frame_features is None:
            visual_frame_features = frame_features
        elif visual_frame_features.ndim == 2:
            visual_frame_features = visual_frame_features.unsqueeze(0)
        if visual_frame_features.shape != frame_features.shape:
            raise ValueError("visual_frame_features shape 必须与 frame_features 一致")
        condition = slow_condition.detached() if detach_slow_condition else slow_condition
        conditioned = self.conditioner(
            frame_features,
            condition.intent_id,
            condition.subgoal_id,
            condition.context_embedding,
            condition.mode_id,
        )
        # Separated paths: slow and fast each run their own GRU.
        indices = self._last_valid_indices(frame_features, valid_mask)
        rows = torch.arange(frame_features.shape[0], device=frame_features.device)
        current_visual = frame_features[rows, indices]
        visual = self.visual_expert(visual_frame_features, valid_mask=valid_mask)
        slow_temporal = None
        if run_slow:
            slow_temporal = (self.slow_temporal(conditioned, valid_mask=valid_mask,
                                                time_deltas=time_deltas)
                             + self.slow_visual_residual(current_visual))
        fast_temporal = (self.fast_temporal(conditioned, valid_mask=valid_mask,
                                            time_deltas=time_deltas)
                         + self.fast_visual_residual(current_visual))

        prior_fast = self.fast_head(fast_temporal, history_actions=history_actions)
        # m11 deployment is deliberately visual-only.  The temporal/history
        # prior remains observable for diagnostics but cannot replace image
        # evidence at inference time.
        camera_prior = float(self.camera_prior_scale)
        fast = FastVLAOutput(
            move_logits=visual.fast.move_logits,
            camera_dx_logits=visual.fast.camera_dx_logits + camera_prior *
                              torch.tanh(prior_fast.camera_dx_logits),
            camera_dy_logits=visual.fast.camera_dy_logits + camera_prior *
                              torch.tanh(prior_fast.camera_dy_logits),
            button_logits=visual.fast.button_logits,
            duration=visual.fast.duration,
            confidence=visual.fast.confidence,
            stop_or_replan=visual.fast.stop_or_replan,
            intent_context_logits=visual.fast.intent_context_logits,
            interact_event_logits=visual.fast.interact_event_logits,
        )
        prior_slow = self.slow_head(slow_temporal) if run_slow and slow_temporal is not None else None
        slow = visual.slow if prior_slow is not None else None

        # The deployment decision feature must include the independent visual
        # expert; diagnostics therefore cannot certify a prior-only GRU path.
        decision_feature = visual.feature
        return FastSlowVLAOutput(temporal_feature=decision_feature, fast=fast, slow=slow,
                                 prior_temporal_feature=fast_temporal,
                                 visual=visual, prior_fast=prior_fast, prior_slow=prior_slow)

    @staticmethod
    def _last_valid_indices(frame_features: torch.Tensor,
                            valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
        if valid_mask is None:
            return torch.full((frame_features.shape[0],), frame_features.shape[1] - 1,
                              dtype=torch.long, device=frame_features.device)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != frame_features.shape[:2]:
            raise ValueError("valid_mask shape 必须与 frame_features 的 [B,L] 一致")
        return valid_mask.long().sum(dim=1).clamp_min(1) - 1

    @staticmethod
    def condition_from_slow_output(
        slow: SlowVLAOutput,
        mode_id: torch.Tensor,
        *,
        intent_id: Optional[torch.Tensor] = None,
        subgoal_id: Optional[torch.Tensor] = None,
    ) -> SlowCondition:
        """Build the next fast-loop condition from a slow prediction.

        ``intent_id``/``subgoal_id`` may be supplied for scheduled-sampling
        teacher forcing.  The continuous context always comes from the slow
        head, so gradients can flow from the fast loss back into that head
        when ``detach_slow_condition=False`` is used by the caller.
        """
        if not isinstance(slow, SlowVLAOutput):
            raise TypeError("slow 必须是 SlowVLAOutput")
        predicted_intent = slow.intent_id
        predicted_subgoal = slow.subgoal_id
        return SlowCondition(
            intent_id=predicted_intent if intent_id is None else intent_id.to(predicted_intent.device),
            subgoal_id=predicted_subgoal if subgoal_id is None else subgoal_id.to(predicted_subgoal.device),
            context_embedding=slow.context_embedding,
            mode_id=mode_id.to(predicted_intent.device),
        )


class FixedRateTrigger:
    """Absolute-deadline trigger independent of producer/capture frequency."""

    def __init__(self, hz: float, start_timestamp_ns: int = 0):
        if hz <= 0:
            raise ValueError("hz 必须为正数")
        self.period_ns = max(1, round(1_000_000_000 / float(hz)))
        self.next_deadline_ns = int(start_timestamp_ns)

    def due(self, timestamp_ns: int) -> bool:
        timestamp_ns = int(timestamp_ns)
        if timestamp_ns < self.next_deadline_ns:
            return False
        skipped = (timestamp_ns - self.next_deadline_ns) // self.period_ns
        self.next_deadline_ns += (skipped + 1) * self.period_ns
        return True
