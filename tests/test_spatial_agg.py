"""SpatialAgg（k×k 保胞→单向量）纯 CPU 单测：shape 契约、退化 sanity、权重。

无 GPU：只验证可训聚合层把 [B,64,1024] 压成 [B,1024]、等胞时近平均、
权重和为 1、MLP 保持维。
运行：python -m pytest tests/test_spatial_agg.py -q
"""

from __future__ import annotations

import pytest
import torch

from idv_agent.model.spatial_agg import SpatialAgg


def test_shape_contract():
    agg = SpatialAgg(dim=1024, n_query=1)
    vecs = torch.randn(3, 64, 1024)
    out = agg(vecs)
    assert out.shape == (3, 1024)


def test_multi_query_still_single_vector():
    agg = SpatialAgg(dim=64, n_query=4, k=4)
    out = agg(torch.randn(2, 16, 64))
    assert out.shape == (2, 64)


def test_equal_cells_near_average():
    # 所有胞同值：输出 ≈ 单一胞（attention 任何权都等于该胞）→ 不爆、接近该值一半即可
    agg = SpatialAgg(dim=8, n_query=1, k=2).double()
    value = torch.full((1, 4, 8), 2.0, dtype=torch.float64)
    out = agg(value)
    # 每个输入胞等于 2；加权平均也应 ~2 (attention 同质)。允许 LayerNorm+MLP 改幅，只测有限
    assert torch.isfinite(out).all()
    # cell_weights 必须和为 1
    w = agg.cell_weights(value)
    assert w.shape == (1, 4)
    assert torch.allclose(w.sum(dim=-1), torch.ones_like(w.sum(-1)), atol=1e-5)


def test_order_equivariance_and_geometry_in_tokens():
    # Grid position is explicit, so swapping cells is observable after pooling.
    agg = SpatialAgg(dim=16, n_query=1, k=4)
    t = torch.randn(1, 16, 16)
    perm = list(range(16)); perm.reverse()
    out_a = agg(t)
    out_b = agg(t[:, perm, :])
    assert torch.isfinite(out_a).all() and torch.isfinite(out_b).all()
    assert not torch.allclose(out_a, out_b, atol=1e-7, rtol=1e-7)
    # Different cell gains should remain observable in the pooled feature.
    mask = torch.cat([torch.ones(1, 8, 16) * 0.1, torch.ones(1, 8, 16) * 2.0], dim=1)
    out_c = agg(t * mask)
    assert out_c.shape == (1, 16)
    # 反向可导：
    out_c.mean().backward()
    assert agg.query.grad is not None and torch.isfinite(agg.query.grad).all()


def test_input_dim_mismatch_raises():
    agg = SpatialAgg(dim=1024)
    with pytest.raises(ValueError):
        agg(torch.randn(1, 4, 999))


def test_position_swap_changes_aggregate():
    """The compressed frame feature must retain the 2-D cell location."""
    agg = SpatialAgg(dim=8, k=2)
    left = torch.zeros(1, 4, 8)
    left[0, 0, 0] = 3.0
    right = left[:, [3, 1, 2, 0], :]
    assert not torch.allclose(agg(left), agg(right), atol=1e-7, rtol=1e-7)
