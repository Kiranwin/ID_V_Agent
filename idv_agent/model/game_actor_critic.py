"""慢系统三头 Actor-Critic：在 VLM backbone 上扩展数值头 + 价值头 + 复用文本头。

设计：
- backbone 鸭子类型：forward(pixel_values, input_ids, labels=?, output_hidden_states=True)
  返回带 .loss/.logits/.hidden_states 的对象（HF AutoModelForCausalLM 兼容）。
- 三个头：
    1. text head : 复用 backbone.lm_head（来自 backbone.loss）
    2. numeric_head : 连续动作均值，取最后 token hidden
    3. value_head  : V(s)，同样取最后 token hidden
- log_std 独立可学习参数，用于连续动作高斯采样（PPO 阶段用）。

BC 训练总损失：
    L = L_text + λ_num * L_num + λ_val * L_val
注意（P3）：BC 阶段 value 无监督目标；若 data 无 value 标签应传 value_targets=None，避免污染特征。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from idv_agent.model.heads import NumericHead, ValueHead


@runtime_checkable
class VLMBackboneProtocol(Protocol):
    def forward(self, *, pixel_values, input_ids, labels=None,
                output_hidden_states=True, **kwargs): ...


@dataclass
class GameActorCriticOutput:
    loss: torch.Tensor
    text_loss: Optional[torch.Tensor]
    numeric_loss: Optional[torch.Tensor]
    value_loss: Optional[torch.Tensor]
    logits: torch.Tensor
    numeric_mean: torch.Tensor
    value: torch.Tensor
    log_std: torch.Tensor


class GameActorCritic(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        hidden_size: int,
        numeric_dim: int = 4,
        log_std_init: float = 0.0,
        numeric_loss_weight: float = 1.0,
        value_loss_weight: float = 0.5,
    ):
        super().__init__()
        self.backbone = backbone
        self.numeric_head = NumericHead(hidden_size, numeric_dim)
        self.value_head = ValueHead(hidden_size)
        self.log_std = nn.Parameter(torch.full((numeric_dim,), log_std_init))
        self.numeric_loss_weight = numeric_loss_weight
        self.value_loss_weight = value_loss_weight

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        numeric_labels: Optional[torch.Tensor] = None,
        value_targets: Optional[torch.Tensor] = None,
        **backbone_kwargs,
    ) -> GameActorCriticOutput:
        out = self.backbone(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            output_hidden_states=True,
            **backbone_kwargs,
        )
        last_hidden = out.hidden_states[-1][:, -1, :]
        head_dtype = next(self.numeric_head.parameters()).dtype
        if last_hidden.dtype != head_dtype:
            last_hidden = last_hidden.to(head_dtype)

        numeric_mean = self.numeric_head(last_hidden)
        value = self.value_head(last_hidden)

        text_loss = out.loss
        numeric_loss = None
        value_loss = None
        total_loss = torch.zeros((), device=numeric_mean.device, dtype=numeric_mean.dtype)
        if text_loss is not None:
            total_loss = total_loss + text_loss.to(total_loss.dtype)
        if numeric_labels is not None:
            numeric_loss = F.mse_loss(numeric_mean, numeric_labels.to(numeric_mean.dtype))
            total_loss = total_loss + self.numeric_loss_weight * numeric_loss
        if value_targets is not None:
            value_loss = F.mse_loss(value, value_targets.to(value.dtype))
            total_loss = total_loss + self.value_loss_weight * value_loss

        return GameActorCriticOutput(
            loss=total_loss,
            text_loss=text_loss,
            numeric_loss=numeric_loss,
            value_loss=value_loss,
            logits=out.logits,
            numeric_mean=numeric_mean,
            value=value,
            log_std=self.log_std,
        )

    @torch.no_grad()
    def sample_numeric(self, numeric_mean: torch.Tensor):
        """高斯采样连续动作并返回 log_prob（PPO 用）。"""
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(numeric_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
