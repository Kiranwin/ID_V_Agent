"""可训的空间聚合层：把 k×k 保胞 token 压回单向量（阶段二共享件）。

现役视觉链路把视觉塔 per-patch 输出 ``mean-pool`` 成单向量（probe 实测画面
可分性只剩 ~7.5e-5 残差）。C 阶段一验证：固定 **k×k 胞 average-pool**（保留胞
位置）能把画面可分残差提升到 ~151×（k=8）。本模块是这个 C 阶段二的**可训**配套：
把每帧的 k² 个胞 token（通常 ``[B, k*k, D]``）按可学权重合成单向量 ``[B, D]``，
从而在**不改变下游“每帧单向量”协议**的前提下让网络学会“注意哪些胞 + 注意多少”。

下游接法（M2_VG / M3_ACT）：在 projection 之前用本层把 64×D 保胞压回 D，
再把该单向量当现役 raw 喂给后续投影/grounding/temporal，保持
``visual_projection(1024→2560)``、``GroundingHead``、GRU 等 shape 契约不变。

可训部分：胞级 attention(logits 由可学 query 与胞向量点积获得) + LayerNorm + MLP。
保胞序之外额外加入可训练的 cell position 参数，避免加权聚合再次对网格
置换不敏感；不含卷积假设。
"""

from __future__ import annotations

import torch
from torch import nn


class SpatialAgg(nn.Module):
    """可学地把 ``[B, C, D]`` 保胞 token 聚合成 ``[B, D]`` 单向量。

    Args:
        dim: 特征维（Qwen3-VL 视觉塔 per-token dim = 1024）。
        n_query: attention query 数（默认 1；聚合仍输出单向量，多个 query 会被
            再平均以保持 ``[B,D]`` 契约；可用 1 以获得确定性单向量）。
        eps: LayerNorm epsilon。
    """

    def __init__(self, dim: int = 1024, n_query: int = 1, k: int = 8, eps: float = 1e-5):
        super().__init__()
        self.dim = int(dim)
        self.n_query = max(1, int(n_query))
        self.k = max(1, int(k))
        self.n_cells = self.k * self.k
        self.query = nn.Parameter(torch.randn(1, self.n_query, self.dim) * (self.dim ** -0.5))
        # The grid order is not available to a permutation-invariant attention
        # pool unless it is made part of each cell representation explicitly.
        self.position = nn.Parameter(torch.randn(1, self.n_cells, self.dim) * (self.dim ** -0.5))
        # score: batch*C 胞对 query 点积 -> softmax attn * value(=胞) -> [1, D]
        self.temperature = float(dim ** -0.5)
        self.norm = nn.LayerNorm(self.dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: ``[B, C, D]``（保胞序）。Return ``[B, D]``。"""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim or tokens.shape[1] != self.n_cells:
            raise ValueError(f"tokens 需 [B,{self.n_cells},{self.dim}]，收到 {tuple(tokens.shape)}")
        q = self.query.to(tokens.device)                      # [1, n_query, D]
        x = tokens + self.position.to(tokens.device, dtype=tokens.dtype)
        attn_logits = torch.einsum("bcd,nqd->bnc", x, q.to(tokens.dtype)) * self.temperature
        attn = torch.softmax(attn_logits, dim=-1)             # [B,n_query,C]
        out = torch.einsum("bnc,bcd->bnd", attn, x)          # weighted position-aware mean
        out = out.mean(dim=1)                                # 平均多 query -> [B,D] 保契约
        return self.mlp(self.norm(out))                       # [B,D]

    def cell_weights(self, tokens: torch.Tensor) -> torch.Tensor:
        """给出 attention softmax 权重 ``[B, C]``（聚合 query 平均后），供 debug。"""
        if tokens.ndim != 3 or tokens.shape[-1] != self.dim or tokens.shape[1] != self.n_cells:
            raise ValueError(f"tokens 需 [B,{self.n_cells},{self.dim}]")
        q = self.query.to(tokens.device)
        x = tokens + self.position.to(tokens.device, dtype=tokens.dtype)
        attn_logits = torch.einsum("bcd,nqd->bnc", x, q.to(tokens.dtype)) * self.temperature
        attn = torch.softmax(attn_logits, dim=-1)
        return attn.mean(dim=1)
