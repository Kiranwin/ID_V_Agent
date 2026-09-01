"""共享的视觉语言主干输出类型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class BackboneOutput:
    """统一 Qwen/VLA 主干的可选语言输出和隐藏状态。"""

    loss: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    hidden_states: list[torch.Tensor]
