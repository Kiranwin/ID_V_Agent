"""Composable fast/slow VLA core operating on cached frame features.

Architecture change (2026-09-03): Separated temporal encoders to prevent
gradient conflicts between slow (discrete classification) and fast (continuous
control) objectives.  The two heads now have independent GRU paths.
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


class SharedFastSlowVLA(nn.Module):
    """Separated temporal encoders plus slow and fast heads.

    Architecture change (2026-09-03): Slow and fast heads now use independent
    GRU temporal encoders to prevent gradient conflicts.  The slow head learns
    long-horizon intent transitions; the fast head learns short-horizon action
    responses.  This increases parameters by ~2M (512-dim GRU × 2) but eliminates
    the gradient tug-of-war between discrete classification and continuous control.

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
        self.slow_head = SlowVLAHead(temporal_dim)
        self.fast_head = FastVLAHead(temporal_dim, history_action_dim=history_action_dim)

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
        run_slow: bool = False,
        detach_slow_condition: bool = True,
    ) -> FastSlowVLAOutput:
        """Forward pass with separated temporal encoders.

        ``detach_slow_condition`` is now less critical because the two heads
        no longer share temporal parameters.  It remains available for optional
        ablation experiments.
        """
        if frame_features.ndim == 2:
            frame_features = frame_features.unsqueeze(0)
        condition = slow_condition.detached() if detach_slow_condition else slow_condition
        conditioned = self.conditioner(
            frame_features,
            condition.intent_id,
            condition.subgoal_id,
            condition.context_embedding,
            condition.mode_id,
        )
        # Separated paths: slow and fast each run their own GRU.
        slow_temporal = self.slow_temporal(conditioned, valid_mask=valid_mask, time_deltas=time_deltas) if run_slow else None
        fast_temporal = self.fast_temporal(conditioned, valid_mask=valid_mask, time_deltas=time_deltas)

        fast = self.fast_head(fast_temporal, history_actions=history_actions)
        slow = self.slow_head(slow_temporal) if run_slow and slow_temporal is not None else None

        # Return fast_temporal as the canonical temporal feature for compatibility
        return FastSlowVLAOutput(temporal_feature=fast_temporal, fast=fast, slow=slow)

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
