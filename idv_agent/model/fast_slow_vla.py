"""Composable fast/slow VLA core operating on cached frame features."""

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
    """Shared temporal core plus slow and fast heads.

    The visual backbone deliberately remains outside this module.  Runtime
    supplies cached per-frame features, ensuring the two heads cannot trigger
    duplicate image encoding.
    """

    def __init__(self, frame_feature_dim: int, temporal_dim: int = 512,
                 history_action_dim: int = 0):
        super().__init__()
        self.conditioner = FiLMConditioner(frame_feature_dim)
        self.temporal = SharedTemporalEncoder(frame_feature_dim, temporal_dim)
        self.slow_head = SlowVLAHead(temporal_dim)
        self.fast_head = FastVLAHead(temporal_dim, history_action_dim=history_action_dim)

    @staticmethod
    def initial_condition(batch_size: int, device: torch.device | str = "cpu",
                          mode_id: int = 0) -> SlowCondition:
        if batch_size < 1:
            raise ValueError("batch_size 必须为正数")
        # idle=7 and observe=1 are the safe cold-start condition.
        return SlowCondition(
            intent_id=torch.full((batch_size,), 7, dtype=torch.long, device=device),
            subgoal_id=torch.full((batch_size,), 1, dtype=torch.long, device=device),
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
        temporal = self.temporal(conditioned, valid_mask=valid_mask, time_deltas=time_deltas)
        fast = self.fast_head(temporal, history_actions=history_actions)
        slow = self.slow_head(temporal) if run_slow else None
        return FastSlowVLAOutput(temporal_feature=temporal, fast=fast, slow=slow)


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

