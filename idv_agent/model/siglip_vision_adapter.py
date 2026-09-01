"""把 SigLIP2-base-patch16-224 的 vision tower 包装成视觉编码器接口。

接口：
    forward(pixel_values: [B, 3, H, W]) -> [B, L, hidden_size]
L = (image_size/patch_size)^2（base/16/224 = 196），hidden_size = 768。
调用方可在外部加入时间编码和 temporal encoder。

transformers 5.x 中 SigLIP1/SigLIP2 共用 SiglipModel；只取 vision_model 子模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


def load_siglip_vision_encoder(
    model_path: str | Path,
    dtype: torch.dtype = torch.float16,
    device: str | torch.device = "cuda",
    freeze: bool = False,
):
    """加载 SigLIP2 vision tower。返回 (vision_module, image_processor, hidden_size, image_size)。"""
    from transformers import AutoModel, AutoImageProcessor

    image_processor = AutoImageProcessor.from_pretrained(str(model_path))
    full_model = AutoModel.from_pretrained(str(model_path), dtype=dtype).to(device)
    vision_model = full_model.vision_model
    if freeze:
        for p in vision_model.parameters():
            p.requires_grad = False

    cfg = full_model.config.vision_config
    hidden_size = cfg.hidden_size
    image_size = cfg.image_size
    return Siglip2VisionAdapter(vision_model), image_processor, hidden_size, image_size


class Siglip2VisionAdapter(nn.Module):
    """SigLIP2 vision tower 包装。输入 [B,3,H,W]，输出序列特征 [B,L,H]。"""

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        self.vision_model = vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.vision_model(pixel_values=pixel_values)
        return out.last_hidden_state  # [B, L, hidden]
