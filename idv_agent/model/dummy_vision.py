"""快系统的 dummy 视觉编码器：小型 CNN 输出 [B, L, H] 序列特征。

实际部署时换成 SigLIP2 / Qwen-VL vision tower；此处仅用于骨架验证（无 GPU/无权重也能跑）。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DummyVisionEncoder(nn.Module):
    """简单 CNN，输出 [B, L=H'×W', hidden_size] 序列特征供 mean pooling 使用。"""

    def __init__(self, hidden_size: int = 192):
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(128, hidden_size, 3, stride=2, padding=1), nn.GELU(),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(pixel_values)   # [B, hidden, H/16, W/16]
        B, C, Hf, Wf = feat.shape
        return feat.permute(0, 2, 3, 1).reshape(B, Hf * Wf, C)
