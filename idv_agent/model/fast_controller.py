"""快系统非自回归控制器（M2 小策略网络）。

输入：
    pixel_values:        [B, 3, H, W]  当前帧
    intent_vector:       [B, intent_dim] 慢系统输出的意图向量（16 维）
    skeleton_embedding:  [B, skel_dim] 动作骨架预计算嵌入

输出：
    category_logits:  [B, num_categories=20]
    continuous_vals:  [B, num_continuous=4]

设计：
- 视觉编码 + mean pool -> [B, vis_dim]；与 intent + skeleton 拼接过 MLP 再分两头。
- 非自回归单次前向，便于 ONNX 静态图导出。
- 训练损失：CE(分类) + λ * MSE(回归)；支持 class_weights（P4 长尾）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FastControllerOutput:
    loss: Optional[torch.Tensor]
    category_loss: Optional[torch.Tensor]
    continuous_loss: Optional[torch.Tensor]
    category_logits: torch.Tensor      # [B, num_categories]
    continuous_vals: torch.Tensor      # [B, num_continuous]


class FastController(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        vision_hidden_size: int,
        intent_dim: int = 16,
        skeleton_dim: int = 64,
        fusion_hidden_size: int = 256,
        num_categories: int = 20,
        num_continuous: int = 4,
        category_loss_weight: float = 1.0,
        continuous_loss_weight: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.intent_dim = intent_dim
        self.skeleton_dim = skeleton_dim
        self.num_categories = num_categories
        self.num_continuous = num_continuous

        fused_in = vision_hidden_size + intent_dim + skeleton_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_in, fusion_hidden_size),
            nn.GELU(),
            nn.Linear(fusion_hidden_size, fusion_hidden_size),
            nn.GELU(),
        )
        self.category_head = nn.Linear(fusion_hidden_size, num_categories)
        self.continuous_head = nn.Linear(fusion_hidden_size, num_continuous)

        self.category_loss_weight = category_loss_weight
        self.continuous_loss_weight = continuous_loss_weight
        if class_weights is not None:
            assert class_weights.numel() == num_categories
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        intent_vector: torch.Tensor,
        skeleton_embedding: torch.Tensor,
        category_labels: Optional[torch.Tensor] = None,    # [B] long
        continuous_labels: Optional[torch.Tensor] = None,  # [B, num_continuous]
    ) -> FastControllerOutput:
        vis = self.vision_encoder(pixel_values)   # [B, L, H_v]
        if vis.dim() == 3:
            vis = vis.mean(dim=1)
        cond = torch.cat([vis, intent_vector, skeleton_embedding], dim=-1)
        fused = self.fusion(cond)
        cat_logits = self.category_head(fused)
        cont_vals = self.continuous_head(fused)

        cat_loss = None
        cont_loss = None
        total_loss: Optional[torch.Tensor] = None
        if category_labels is not None:
            cat_loss = F.cross_entropy(cat_logits, category_labels, weight=self.class_weights)
            total_loss = self.category_loss_weight * cat_loss
        if continuous_labels is not None:
            cont_loss = F.mse_loss(cont_vals, continuous_labels)
            total_loss = (total_loss or 0.0) + self.continuous_loss_weight * cont_loss
        if total_loss is not None and not torch.is_tensor(total_loss):
            total_loss = torch.tensor(total_loss, device=cat_logits.device, dtype=cat_logits.dtype)

        return FastControllerOutput(
            loss=total_loss,
            category_loss=cat_loss,
            continuous_loss=cont_loss,
            category_logits=cat_logits,
            continuous_vals=cont_vals,
        )
