"""Runtime caches and lightweight temporal encoder for the shared VLA base."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Optional

import torch
from torch import nn


@dataclass(frozen=True)
class TaskConditionCache:
    """Episode-constant text-side condition.

    ``instruction_embedding`` and ``mode_embedding`` are computed once at the
    beginning of an episode.  The frame encoder receives these tensors without
    tokenizing or running the text tower again.
    """

    instruction_embedding: torch.Tensor
    mode_embedding: torch.Tensor
    task_id: str = ""
    mode: str = "standard"

    def __post_init__(self) -> None:
        if self.instruction_embedding.ndim == 0 or self.mode_embedding.ndim == 0:
            raise ValueError("task embeddings 必须至少有一个维度")
        if not self.task_id:
            raise ValueError("task_id 不能为空")
        if not self.mode:
            raise ValueError("mode 不能为空")


@dataclass(frozen=True)
class FrameFeature:
    frame_index: int
    timestamp_ns: int
    feature: torch.Tensor
    valid: bool = True


class FrameFeatureCache:
    """Thread-safe ring buffer for per-frame backbone features.

    The cache stores features, not images.  A producer may append at capture
    rate while fast/slow consumers take snapshots at their own frequencies.
    """

    def __init__(self, max_length: int = 8):
        if max_length < 1:
            raise ValueError("max_length 必须为正数")
        self.max_length = int(max_length)
        self._items: deque[FrameFeature] = deque(maxlen=self.max_length)
        self._lock = RLock()

    def append(
        self,
        frame_index: int | FrameFeature,
        timestamp_ns: Optional[int] = None,
        feature: Optional[torch.Tensor] = None,
        valid: bool = True,
    ) -> None:
        if isinstance(frame_index, FrameFeature):
            item = frame_index
        else:
            if timestamp_ns is None or feature is None:
                raise ValueError("append 需要 timestamp_ns 和 feature")
            item = FrameFeature(int(frame_index), int(timestamp_ns), feature, bool(valid))
        if item.frame_index < 0 or item.timestamp_ns < 0:
            raise ValueError("frame_index/timestamp_ns 必须非负")
        if not isinstance(item.feature, torch.Tensor) or item.feature.ndim < 1:
            raise ValueError("feature 必须是至少一维 Tensor")
        with self._lock:
            if self._items:
                previous = self._items[-1]
                if item.frame_index <= previous.frame_index:
                    raise ValueError("frame_index 必须严格递增")
                if item.timestamp_ns <= previous.timestamp_ns:
                    raise ValueError("timestamp_ns 必须严格递增")
            self._items.append(item)

    def latest(self) -> Optional[FrameFeature]:
        with self._lock:
            return self._items[-1] if self._items else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def window(self, length: int = 8, stride: int = 1) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return newest items sampled every ``stride`` frames.

        Short windows are returned short (3..8 is valid by the VLA v4
        contract); callers that batch variable lengths can pad using the mask.
        The newest item is always included.  A stride of 3 and length 8
        therefore needs 22 cached frames and returns indices ``[-21,-18,...,0]``
        relative to the newest item, matching the v5 data builder.
        """
        if length < 1:
            raise ValueError("length 必须为正数")
        if stride < 1:
            raise ValueError("stride 必须为正数")
        with self._lock:
            all_items = list(self._items)
            items = list(reversed(all_items))[::int(stride)][:int(length)]
            items.reverse()
        if not items:
            raise RuntimeError("FrameFeatureCache 为空")
        shape = items[0].feature.shape
        if any(item.feature.shape != shape for item in items):
            raise ValueError("缓存中的 feature shape 必须一致")
        features = torch.stack([item.feature for item in items], dim=0)
        timestamps = torch.tensor([item.timestamp_ns for item in items], dtype=torch.long,
                                  device=features.device)
        valid = torch.tensor([item.valid for item in items], dtype=torch.bool,
                             device=features.device)
        return features, timestamps, valid


class SharedTemporalEncoder(nn.Module):
    """Causal GRU over cached per-frame features.

    This module never calls the visual-language backbone.  ``time_deltas`` are
    optional seconds from the first frame in each window and are projected into
    the feature space before the GRU.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 1):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("input_dim/hidden_dim/num_layers 必须为正数")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.delta_proj = nn.Linear(1, self.input_dim)
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=num_layers,
                          batch_first=True)

    def forward(
        self,
        features: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        time_deltas: Optional[torch.Tensor] = None,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"features 必须是 [B,L,{self.input_dim}]")
        batch, steps, _ = features.shape
        device = features.device
        if valid_mask is None:
            valid_mask = torch.ones((batch, steps), dtype=torch.bool, device=device)
        elif valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != (batch, steps):
            raise ValueError("valid_mask shape 必须为 [B,L]")
        if time_deltas is None:
            time_deltas = torch.zeros((batch, steps), dtype=features.dtype, device=device)
        elif time_deltas.ndim == 1:
            time_deltas = time_deltas.unsqueeze(0)
        if time_deltas.shape != (batch, steps):
            raise ValueError("time_deltas shape 必须为 [B,L]")

        delta = self.delta_proj(time_deltas.to(features.dtype).unsqueeze(-1))
        sequence, _ = self.gru(features + delta)
        # Select the last valid state, not necessarily the final padded slot.
        indices = valid_mask.long().sum(dim=1).clamp_min(1) - 1
        pooled = sequence[torch.arange(batch, device=device), indices]
        if return_sequence:
            return pooled, sequence
        return pooled

