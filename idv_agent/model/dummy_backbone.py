"""慢系统的 dummy backbone：极小的语言 backbone，输出 hidden + loss，接口对齐 VLM。

实际部署时换成 Qwen3-VL（`model/qwen_backbone_adapter.py`）。此处仅用于骨架/训练链路验证。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    """所有 backbone（dummy / Qwen3-VL）统一返回的接口对象。"""
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: list[torch.Tensor]


class DummyBackbone(nn.Module):
    """最小可 forward 的 multimodal 语言 backbone。

    接口（与 GameActorCritic 兼容）：
        forward(pixel_values, input_ids, labels=None, attention_mask=None, output_hidden_states=True)
        返回 BackboneOutput(loss, logits, hidden_states)。
    """

    def __init__(self, vocab_size: int = 120, hidden_size: int = 64,
                 image_feature: int = 8, num_heads: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        # 图像：全局自适应池化到 [image_feature] 再线性投影到 hidden
        self.vis_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),                 # [B, 3]
            nn.Linear(3, hidden_size),
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, *, pixel_values, input_ids, labels=None,
                attention_mask=None, output_hidden_states=True, **kwargs):
        txt = self.embed(input_ids)                       # [B, T, hidden]
        vis = self.vis_proj(pixel_values)                 # [B, hidden]
        # 把图像向量加到首个 token 的嵌入（residual），保持序列长度=文本长度，便于对齐 labels
        vis_token = vis.unsqueeze(1)                      # [B, 1, hidden]
        combined = txt.clone()
        combined[:, 0:1, :] = combined[:, 0:1, :] + vis_token
        logits = self.lm_head(combined)                   # [B, T, vocab]
        hidden_states = [combined, combined]
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1).long(),
                ignore_index=-100,
            )
        return BackboneOutput(loss=loss, logits=logits, hidden_states=hidden_states)
