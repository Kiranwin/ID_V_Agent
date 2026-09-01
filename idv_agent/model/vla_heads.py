"""Shared-base VLA condition fusion and fast/slow prediction heads.

The heads are intentionally backbone-agnostic: they consume the output of a
cached per-frame encoder and can therefore be smoke-tested without model
weights or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from idv_agent.configs.subgoal import SUBGOAL_NAMES
from idv_agent.vla.action_chunk import ACTION_CHUNK_HORIZON, CAMERA_BUCKETS, INTENTS, MOVE_DIRECTIONS


CONTEXT_EMBEDDING_DIM = 256
INTENT_EMBEDDING_DIM = 64
SUBGOAL_EMBEDDING_DIM = 64
MODE_EMBEDDING_DIM = 32


@dataclass
class SlowVLAOutput:
    intent_logits: torch.Tensor
    subgoal_logits: torch.Tensor
    context_embedding: torch.Tensor
    refresh_logits: torch.Tensor

    @property
    def intent_id(self) -> torch.Tensor:
        return self.intent_logits.argmax(dim=-1)

    @property
    def subgoal_id(self) -> torch.Tensor:
        return self.subgoal_logits.argmax(dim=-1)


@dataclass
class FastVLAOutput:
    move_logits: torch.Tensor
    camera_dx_logits: torch.Tensor
    camera_dy_logits: torch.Tensor
    button_logits: torch.Tensor
    duration: torch.Tensor
    confidence: torch.Tensor
    stop_or_replan: torch.Tensor
    intent_context_logits: torch.Tensor


class FiLMConditioner(nn.Module):
    """Turn slow discrete/continuous context into per-channel FiLM values."""

    def __init__(self, feature_dim: int, mode_count: int = 3):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim 必须为正数")
        self.feature_dim = int(feature_dim)
        self.intent_embedding = nn.Embedding(len(INTENTS), INTENT_EMBEDDING_DIM)
        self.subgoal_embedding = nn.Embedding(len(SUBGOAL_NAMES), SUBGOAL_EMBEDDING_DIM)
        self.mode_embedding = nn.Embedding(mode_count, MODE_EMBEDDING_DIM)
        self.context_norm = nn.LayerNorm(CONTEXT_EMBEDDING_DIM)
        self.condition = nn.Sequential(
            nn.Linear(INTENT_EMBEDDING_DIM + SUBGOAL_EMBEDDING_DIM + CONTEXT_EMBEDDING_DIM + MODE_EMBEDDING_DIM,
                      max(128, feature_dim // 2)),
            nn.GELU(),
            nn.Linear(max(128, feature_dim // 2), feature_dim * 2),
        )
        self.output_norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        frame_features: torch.Tensor,
        intent_id: torch.Tensor,
        subgoal_id: torch.Tensor,
        context_embedding: torch.Tensor,
        mode_id: torch.Tensor,
    ) -> torch.Tensor:
        """Apply one condition to every frame in ``[B,L,D]`` features."""
        if frame_features.ndim != 3 or frame_features.shape[-1] != self.feature_dim:
            raise ValueError(f"frame_features 必须是 [B,L,{self.feature_dim}]")
        batch = frame_features.shape[0]
        context = context_embedding
        if context.ndim == 1:
            context = context.unsqueeze(0)
        if context.shape != (batch, CONTEXT_EMBEDDING_DIM):
            raise ValueError(f"context_embedding 必须是 [B,{CONTEXT_EMBEDDING_DIM}]")
        intent_id = intent_id.reshape(-1).to(torch.long)
        subgoal_id = subgoal_id.reshape(-1).to(torch.long)
        mode_id = mode_id.reshape(-1).to(torch.long)
        if not (len(intent_id) == len(subgoal_id) == len(mode_id) == batch):
            raise ValueError("条件 batch 维度不一致")
        condition = self.condition(torch.cat((
            self.intent_embedding(intent_id),
            self.subgoal_embedding(subgoal_id),
            self.context_norm(context),
            self.mode_embedding(mode_id),
        ), dim=-1))
        gamma, beta = condition.chunk(2, dim=-1)
        return self.output_norm((1.0 + torch.tanh(gamma).unsqueeze(1)) * frame_features
                                + beta.unsqueeze(1))


class SlowVLAHead(nn.Module):
    """Low-frequency tactical intent/subgoal/context head."""

    def __init__(self, feature_dim: int, context_dim: int = CONTEXT_EMBEDDING_DIM,
                 refresh_classes: int = 3):
        super().__init__()
        if feature_dim < 1 or context_dim != CONTEXT_EMBEDDING_DIM:
            raise ValueError("feature_dim 必须为正数且 context_dim 必须为 256")
        self.intent = nn.Linear(feature_dim, len(INTENTS))
        self.subgoal = nn.Linear(feature_dim, len(SUBGOAL_NAMES))
        self.context = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.GELU(),
                                     nn.Linear(feature_dim, context_dim))
        self.refresh = nn.Linear(feature_dim, refresh_classes)

    def forward(self, temporal_feature: torch.Tensor) -> SlowVLAOutput:
        if temporal_feature.ndim == 1:
            temporal_feature = temporal_feature.unsqueeze(0)
        if temporal_feature.ndim != 2:
            raise ValueError("temporal_feature 必须是 [B,D]")
        return SlowVLAOutput(
            intent_logits=self.intent(temporal_feature),
            subgoal_logits=self.subgoal(temporal_feature),
            context_embedding=self.context(temporal_feature),
            refresh_logits=self.refresh(temporal_feature),
        )


class FastVLAHead(nn.Module):
    """High-frequency action-chunk head with fixed v3/v4 action dimensions."""

    def __init__(self, feature_dim: int, history_action_dim: int = 0,
                 horizon: int = ACTION_CHUNK_HORIZON):
        super().__init__()
        if feature_dim < 1 or horizon < 1:
            raise ValueError("feature_dim/horizon 必须为正数")
        self.horizon = int(horizon)
        input_dim = feature_dim + max(0, int(history_action_dim))
        self.trunk = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.GELU(),
                                   nn.Linear(feature_dim, feature_dim), nn.GELU())
        self.move = nn.Linear(feature_dim, self.horizon * len(MOVE_DIRECTIONS))
        self.camera_dx = nn.Linear(feature_dim, self.horizon * len(CAMERA_BUCKETS))
        self.camera_dy = nn.Linear(feature_dim, self.horizon * len(CAMERA_BUCKETS))
        self.buttons = nn.Linear(feature_dim, self.horizon * 6)
        # Duration is a bounded scalar in frames; sigmoid is mapped to 1..30.
        self.duration = nn.Linear(feature_dim, self.horizon)
        self.confidence = nn.Linear(feature_dim, 1)
        self.stop = nn.Linear(feature_dim, 1)
        # Auxiliary intent prediction used only for fast/slow consistency.
        self.intent_context = nn.Linear(feature_dim, len(INTENTS))
        self.history_action_dim = max(0, int(history_action_dim))

    def forward(self, temporal_feature: torch.Tensor,
                history_actions: Optional[torch.Tensor] = None) -> FastVLAOutput:
        if temporal_feature.ndim == 1:
            temporal_feature = temporal_feature.unsqueeze(0)
        if temporal_feature.ndim != 2:
            raise ValueError("temporal_feature 必须是 [B,D]")
        if self.history_action_dim:
            if history_actions is None or history_actions.ndim != 2:
                raise ValueError("配置了 history_action_dim 时必须提供 [B,H] history_actions")
            if history_actions.shape != (temporal_feature.shape[0], self.history_action_dim):
                raise ValueError("history_actions shape 不匹配")
            temporal_feature = torch.cat((temporal_feature, history_actions.to(temporal_feature.dtype)), dim=-1)
        hidden = self.trunk(temporal_feature)
        batch = hidden.shape[0]
        reshape = lambda value, width: value.view(batch, self.horizon, width)
        return FastVLAOutput(
            move_logits=reshape(self.move(hidden), len(MOVE_DIRECTIONS)),
            camera_dx_logits=reshape(self.camera_dx(hidden), len(CAMERA_BUCKETS)),
            camera_dy_logits=reshape(self.camera_dy(hidden), len(CAMERA_BUCKETS)),
            button_logits=reshape(self.buttons(hidden), 6),
            duration=1.0 + 29.0 * torch.sigmoid(self.duration(hidden)),
            confidence=torch.sigmoid(self.confidence(hidden)).squeeze(-1),
            stop_or_replan=torch.sigmoid(self.stop(hidden)).squeeze(-1),
            intent_context_logits=self.intent_context(hidden),
        )
