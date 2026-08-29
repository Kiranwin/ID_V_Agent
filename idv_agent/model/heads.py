"""三头 Actor-Critic 共用小模块：连续动作均值回归头 + 价值估计头。

- 都是简单 MLP（GELU + 一层 hidden 放大），不依赖具体 backbone，输入 [B, hidden_size]。
- 数值稳定：log_std 用单独 nn.Parameter 而非头输出，避免与动作均值耦合。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class NumericHead(nn.Module):
    """连续动作均值回归头：hidden -> numeric_dim（输出高斯均值 μ，std 由外部 log_std 给）。"""

    def __init__(self, hidden_size: int, numeric_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, numeric_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class ValueHead(nn.Module):
    """状态价值头 V(s)：hidden -> 1（squeeze 后 [B]）。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden).squeeze(-1)
