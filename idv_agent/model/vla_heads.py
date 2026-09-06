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
    # Independent one-shot interaction event (Q) logits.  The first column of
    # ``button_logits`` mirrors this tensor for wire compatibility, but is not
    # part of the ordinary held-button loss.
    interact_event_logits: Optional[torch.Tensor] = None

    def button_predictions(self, *, event_threshold: float = 0.0) -> torch.Tensor:
        """Decode wire-format buttons with the calibrated one-shot Q threshold."""
        threshold = torch.as_tensor(event_threshold)
        if not bool(torch.isfinite(threshold)):
            raise ValueError("event_threshold 必须是有限数")
        predictions = self.button_logits.sigmoid() >= 0.5
        event_logits = (self.interact_event_logits
                        if self.interact_event_logits is not None
                        else self.button_logits[..., 0])
        predictions = predictions.clone()
        predictions[..., 0] = event_logits >= float(event_threshold)
        return predictions


@dataclass
class VisualExpertOutput:
    """History-free visual predictions and their decision representation."""

    fast: FastVLAOutput
    slow: SlowVLAOutput
    feature: torch.Tensor
    grounding: Optional["GroundingOutput"] = None


@dataclass
class GroundingOutput:
    """Training-only visual grounding predictions for the current input frame."""
    present_logits: torch.Tensor
    bbox: torch.Tensor
    side_logits: torch.Tensor
    prompt_logits: torch.Tensor
    reachable_logits: torch.Tensor


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
                 refresh_classes: int = 3, *, bias: bool = True, direct: bool = False):
        super().__init__()
        if feature_dim < 1 or context_dim != CONTEXT_EMBEDDING_DIM:
            raise ValueError("feature_dim 必须为正数且 context_dim 必须为 256")
        self.intent = nn.Linear(feature_dim, len(INTENTS), bias=bias)
        self.subgoal = nn.Linear(feature_dim, len(SUBGOAL_NAMES), bias=bias)
        self.context = (nn.Linear(feature_dim, context_dim, bias=bias) if direct else
                        nn.Sequential(nn.Linear(feature_dim, feature_dim, bias=bias), nn.GELU(),
                                      nn.Linear(feature_dim, context_dim, bias=bias)))
        self.refresh = nn.Linear(feature_dim, refresh_classes, bias=bias)

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
                 horizon: int = ACTION_CHUNK_HORIZON, *, bias: bool = True, direct: bool = False):
        super().__init__()
        if feature_dim < 1 or horizon < 1:
            raise ValueError("feature_dim/horizon 必须为正数")
        self.horizon = int(horizon)
        input_dim = feature_dim + max(0, int(history_action_dim))
        if direct:
            if history_action_dim:
                raise ValueError("direct FastVLAHead 不允许 history actions")
            self.trunk = nn.Identity()
        else:
            self.trunk = nn.Sequential(nn.Linear(input_dim, feature_dim, bias=bias), nn.GELU(),
                                       nn.Linear(feature_dim, feature_dim, bias=bias), nn.GELU())
        self.move = nn.Linear(feature_dim, self.horizon * len(MOVE_DIRECTIONS), bias=bias)
        self.camera_dx = nn.Linear(feature_dim, self.horizon * len(CAMERA_BUCKETS), bias=bias)
        self.camera_dy = nn.Linear(feature_dim, self.horizon * len(CAMERA_BUCKETS), bias=bias)
        # Five ordinary button channels; the one-shot Q event has its own
        # classifier below and is mirrored into button_logits[..., 0].
        self.buttons = nn.Linear(feature_dim, self.horizon * 5, bias=bias)
        self.interact_event = nn.Linear(feature_dim, self.horizon, bias=bias)
        # Duration is a bounded scalar in frames; sigmoid is mapped to 1..30.
        self.duration = nn.Linear(feature_dim, self.horizon, bias=bias)
        self.confidence = nn.Linear(feature_dim, 1, bias=bias)
        self.stop = nn.Linear(feature_dim, 1, bias=bias)
        # Auxiliary intent prediction used only for fast/slow consistency.
        self.intent_context = nn.Linear(feature_dim, len(INTENTS), bias=bias)
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
        ordinary_button_logits = reshape(self.buttons(hidden), 5)
        interact_event_logits = self.interact_event(hidden).view(batch, self.horizon)
        button_logits = torch.cat((interact_event_logits.unsqueeze(-1), ordinary_button_logits), dim=-1)
        return FastVLAOutput(
            move_logits=reshape(self.move(hidden), len(MOVE_DIRECTIONS)),
            camera_dx_logits=reshape(self.camera_dx(hidden), len(CAMERA_BUCKETS)),
            camera_dy_logits=reshape(self.camera_dy(hidden), len(CAMERA_BUCKETS)),
            button_logits=button_logits,
            duration=1.0 + 29.0 * torch.sigmoid(self.duration(hidden)),
            confidence=torch.sigmoid(self.confidence(hidden)).squeeze(-1),
            stop_or_replan=torch.sigmoid(self.stop(hidden)).squeeze(-1),
            intent_context_logits=self.intent_context(hidden),
            interact_event_logits=interact_event_logits,
        )


class VisualActionExpert(nn.Module):
    """Predict action and tactical outputs from image evidence only.

    The input explicitly contains the latest visual state and its change from
    the first valid observation.  It intentionally has no history-action or
    slow-condition input, making the visual path inspectable and deployable.
    """

    def __init__(self, frame_feature_dim: int, temporal_dim: int,
                 horizon: int = ACTION_CHUNK_HORIZON, *, max_frames: int = 8):
        super().__init__()
        if frame_feature_dim < 1 or temporal_dim < 1 or horizon < 1:
            raise ValueError("frame_feature_dim/temporal_dim/horizon 必须为正数")
        self.frame_feature_dim = int(frame_feature_dim)
        self.temporal_dim = int(temporal_dim)
        self.max_frames = int(max_frames)
        if self.max_frames < 1:
            raise ValueError("max_frames 必须为正数")
        # Preserve the whole visual window.  The previous expert only kept
        # first/last frames, which discarded the camera trajectory in the
        # middle of the observation window.
        self.sequence_dim = self.frame_feature_dim * self.max_frames
        self.pair_dim = self.sequence_dim + self.frame_feature_dim
        self.visual_feature_scale = 0.25
        self.register_buffer("input_center", torch.zeros(self.pair_dim), persistent=True)
        self.input_norm = nn.LayerNorm(self.pair_dim, elementwise_affine=False)
        self.fast = FastVLAHead(self.pair_dim, history_action_dim=0, horizon=horizon,
                                bias=False, direct=True)
        self.slow = SlowVLAHead(self.pair_dim, bias=False, direct=True)
        # Sparse event channels need a calibrated negative baseline.  The
        # parameter is initialized from the train-subset event rate and
        # remains trainable so visual evidence can move it away from prior.
        self.interact_event_bias = nn.Parameter(torch.zeros(horizon))
        # Ordinary held buttons use an independent five-channel baseline;
        # Q is intentionally excluded because it is an edge-triggered event.
        self.ordinary_button_bias = nn.Parameter(torch.zeros(horizon, 5))
        self.camera_summary_dim = self.frame_feature_dim * 5
        self.camera_summary_scale = 0.25
        # Keep the direct full-window visual signal in the camera path and
        # fuse it with the trajectory summary instead of replacing it.
        self.camera_visual_pair_dx = nn.Linear(self.pair_dim,
                                               self.fast.horizon * len(CAMERA_BUCKETS),
                                               bias=False)
        self.camera_visual_pair_dy = nn.Linear(self.pair_dim,
                                               self.fast.horizon * len(CAMERA_BUCKETS),
                                               bias=False)
        self.camera_visual_dx = nn.Linear(self.camera_summary_dim,
                                          self.fast.horizon * len(CAMERA_BUCKETS), bias=False)
        self.camera_visual_dy = nn.Linear(self.camera_summary_dim,
                                          self.fast.horizon * len(CAMERA_BUCKETS), bias=False)
        # These heads consume the direct full visual pair, never history,
        # temporal prior, slow condition, YOLO detections, or runtime rules.
        self.grounding_present = nn.Linear(self.pair_dim, 1, bias=False)
        self.grounding_bbox = nn.Linear(self.pair_dim, 4, bias=False)
        self.grounding_side = nn.Linear(self.pair_dim, 4, bias=False)
        self.grounding_prompt = nn.Linear(self.pair_dim, 1, bias=False)
        self.grounding_reachable = nn.Linear(self.pair_dim, 1, bias=False)

    @staticmethod
    def _valid_indices(frame_features: torch.Tensor,
                       valid_mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        batch, steps, _ = frame_features.shape
        if valid_mask is None:
            first = torch.zeros(batch, dtype=torch.long, device=frame_features.device)
            last = torch.full((batch,), steps - 1, dtype=torch.long, device=frame_features.device)
            return first, last
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != (batch, steps):
            raise ValueError("valid_mask shape 必须为 [B,L]")
        valid_mask = valid_mask.bool()
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("每个视觉窗口至少需要一帧有效画面")
        first = valid_mask.long().argmax(dim=1)
        last = valid_mask.long().sum(dim=1).clamp_min(1) - 1
        return first, last

    def forward(self, frame_features: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> VisualExpertOutput:
        if frame_features.ndim == 2:
            frame_features = frame_features.unsqueeze(0)
        if frame_features.ndim != 3 or frame_features.shape[-1] != self.frame_feature_dim:
            raise ValueError(f"frame_features 必须是 [B,L,{self.frame_feature_dim}]")
        feature = (self.normalized_pair_features(frame_features, valid_mask=valid_mask) *
                   self.visual_feature_scale).float()
        # The direct 8192 -> logits path is intentionally kept in FP32.  On
        # Turing GPUs, FP16 autocast can overflow its reduction/linear
        # gradients even though LayerNorm's output is finite.
        with torch.autocast(device_type=feature.device.type, enabled=False):
            fast = self.fast(feature)
            slow = self.slow(feature)
            event_bias = self.interact_event_bias.to(fast.button_logits).view(1, self.fast.horizon)
            ordinary_bias = self.ordinary_button_bias.to(
                fast.button_logits).view(1, self.fast.horizon, 5)
            event_logits = fast.interact_event_logits + event_bias
            fast.interact_event_logits = event_logits
            fast.button_logits = fast.button_logits.clone()
            fast.button_logits[..., 0] = event_logits
            fast.button_logits[..., 1:] = fast.button_logits[..., 1:] + ordinary_bias
        camera = self.camera_summary(frame_features, valid_mask=valid_mask)
        batch = camera.shape[0]
        pair_dx = self.camera_visual_pair_dx(feature).view(
            batch, self.fast.horizon, len(CAMERA_BUCKETS))
        pair_dy = self.camera_visual_pair_dy(feature).view(
            batch, self.fast.horizon, len(CAMERA_BUCKETS))
        summary_dx = self.camera_visual_dx(camera).view(
            batch, self.fast.horizon, len(CAMERA_BUCKETS))
        summary_dy = self.camera_visual_dy(camera).view(
            batch, self.fast.horizon, len(CAMERA_BUCKETS))
        fast.camera_dx_logits = pair_dx + summary_dx
        fast.camera_dy_logits = pair_dy + summary_dy
        grounding = GroundingOutput(
            present_logits=self.grounding_present(feature).squeeze(-1),
            bbox=torch.sigmoid(self.grounding_bbox(feature)),
            side_logits=self.grounding_side(feature),
            prompt_logits=self.grounding_prompt(feature).squeeze(-1),
            reachable_logits=self.grounding_reachable(feature).squeeze(-1),
        )
        return VisualExpertOutput(fast=fast, slow=slow, feature=feature, grounding=grounding)

    def set_interact_event_bias(self, bias: torch.Tensor) -> None:
        """Initialize only the independent one-shot interaction baseline."""
        bias = bias.detach().to(device=self.interact_event_bias.device,
                                dtype=self.interact_event_bias.dtype).reshape(-1)
        if bias.shape != self.interact_event_bias.shape or not bool(torch.isfinite(bias).all()):
            raise ValueError(f"interact event bias 必须是有限 [{self.fast.horizon}] Tensor")
        with torch.no_grad():
            self.interact_event_bias.copy_(bias)

    def set_ordinary_button_bias(self, bias: torch.Tensor) -> None:
        """Initialize the five ordinary held-button baselines."""
        bias = bias.detach().to(device=self.ordinary_button_bias.device,
                                dtype=self.ordinary_button_bias.dtype)
        if bias.shape != self.ordinary_button_bias.shape or not bool(torch.isfinite(bias).all()):
            raise ValueError(f"ordinary button bias 必须是有限 [{self.fast.horizon},5] Tensor")
        with torch.no_grad():
            self.ordinary_button_bias.copy_(bias)

    def camera_summary(self, frame_features: torch.Tensor,
                       valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Summarize camera-relevant visual trajectory without raw flattening."""
        if frame_features.ndim == 2:
            frame_features = frame_features.unsqueeze(0)
        first_index, last_index = self._valid_indices(frame_features, valid_mask)
        rows = torch.arange(frame_features.shape[0], device=frame_features.device)
        first = frame_features[rows, first_index]
        last = frame_features[rows, last_index]
        if valid_mask is None:
            valid = torch.ones(frame_features.shape[:2], dtype=torch.bool,
                               device=frame_features.device)
        else:
            valid = valid_mask.bool()
            if valid.ndim == 1:
                valid = valid.unsqueeze(0)
            if valid.shape != frame_features.shape[:2]:
                raise ValueError("valid_mask shape 必须为 [B,L]")
        mean = (frame_features * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        if frame_features.shape[1] > 1:
            delta = frame_features[:, 1:] - frame_features[:, :-1]
            delta_valid = valid[:, 1:] & valid[:, :-1]
            delta_mean = (delta * delta_valid.unsqueeze(-1)).sum(dim=1) / delta_valid.sum(dim=1, keepdim=True).clamp_min(1)
        else:
            delta_mean = torch.zeros_like(last)
        summary = torch.cat((first, last, mean, last - first, delta_mean), dim=-1)
        sequence_dim = self.frame_feature_dim * self.max_frames
        center = self.input_center[:sequence_dim]
        center_first = center[:self.frame_feature_dim]
        center_last = center[(self.max_frames - 1) * self.frame_feature_dim:self.max_frames * self.frame_feature_dim]
        center_mean = center[:self.frame_feature_dim * self.max_frames].reshape(
            self.max_frames, self.frame_feature_dim).mean(dim=0)
        center_delta = self.input_center[sequence_dim:]
        summary_center = torch.cat((center_first, center_last, center_mean,
                                    center_last - center_first, center_delta), dim=-1)
        return (summary - summary_center.to(summary)) * self.camera_summary_scale

    def pair_features(self, frame_features: torch.Tensor,
                      valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return raw ``[all_frames, last-first]`` features for calibration."""
        if frame_features.ndim == 2:
            frame_features = frame_features.unsqueeze(0)
        if frame_features.ndim != 3 or frame_features.shape[-1] != self.frame_feature_dim:
            raise ValueError(f"frame_features 必须是 [B,L,{self.frame_feature_dim}]")
        first_index, last_index = self._valid_indices(frame_features, valid_mask)
        rows = torch.arange(frame_features.shape[0], device=frame_features.device)
        first = frame_features[rows, first_index]
        last = frame_features[rows, last_index]
        if frame_features.shape[1] > self.max_frames:
            raise ValueError(f"视觉窗口长度不能超过 {self.max_frames}")
        sequence = torch.zeros((frame_features.shape[0], self.max_frames,
                                self.frame_feature_dim), device=frame_features.device,
                               dtype=frame_features.dtype)
        sequence[:, :frame_features.shape[1]] = frame_features
        return torch.cat((sequence.reshape(frame_features.shape[0], -1), last - first), dim=-1)

    def set_input_center(self, center: torch.Tensor) -> None:
        center = center.detach().reshape(-1).to(device=self.input_center.device,
                                                 dtype=self.input_center.dtype)
        if center.shape != self.input_center.shape or not bool(torch.isfinite(center).all()):
            raise ValueError(f"visual input center 必须是有限 [{self.pair_dim}] Tensor")
        self.input_center.copy_(center)

    def normalized_pair_features(self, frame_features: torch.Tensor,
                                 valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pair = self.pair_features(frame_features, valid_mask=valid_mask)
        return self.input_norm(pair - self.input_center.to(pair))
