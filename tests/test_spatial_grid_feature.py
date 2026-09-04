"""SpatialGridEncoder 纯 CPU 单元测试：验证 k×k 胞折叠几何与数学。

无 GPU/模型依赖，先锁定折叠几何正确性再触碰冻结视觉塔。
运行：``python -m pytest tests/test_spatial_grid_feature.py -q``
"""

from __future__ import annotations

import pytest
import torch

from idv_agent.model.spatial_grid_feature import (
    _spans,
    grid_spatial_pool,
    grid_spatial_pool_batch,
    per_image_rows_and_geo,
)


def test_spans_evenly_splits_non_multiple():
    assert _spans(10, 3) == [(0, 4), (4, 7), (7, 10)]
    assert _spans(8, 2) == [(0, 4), (4, 8)]
    assert _spans(9, 3) == [(0, 3), (3, 6), (6, 9)]


def test_spans_k_larger_than_length_safe():
    spans = _spans(3, 5)
    assert len(spans) == 5
    assert spans[0] == (0, 1)
    assert spans[-1][1] <= 3


def test_grid_spatial_pool_value_math():
    gh, gw, D, k = 4, 4, 2, 2
    grid = torch.arange(gh * gw, dtype=torch.float32).reshape(gh, gw, 1).repeat(1, 1, D)
    flat = grid.reshape(-1, D)
    out = grid_spatial_pool(flat, gh * gw, gh, gw, k)
    assert out.shape == (k * k, D)
    assert out[0, 0].item() == pytest.approx(2.5)   # rows0..2,cols0..2 → {0,1,4,5}
    assert out[1, 0].item() == pytest.approx(4.5)   # cols2..4           → {2,3,6,7}
    assert out[2, 0].item() == pytest.approx(10.5)  # rows2..4,cols0..2  → {8,9,12,13}
    assert out[3, 0].item() == pytest.approx(12.5)  # → {10,11,14,15}


def test_grid_spatial_pool_batch_matches_per_image_pooling_exactly():
    """Same-geometry batch pooling must preserve the existing cell values."""
    batch = torch.arange(3 * 5 * 7 * 2, dtype=torch.float32).reshape(3, 5, 7, 2)
    expected = torch.stack([
        grid_spatial_pool(image.reshape(-1, 2), 35, 5, 7, 3)
        for image in batch
    ])

    actual = grid_spatial_pool_batch(batch, k=3)

    assert actual.shape == (3, 9, 2)
    assert torch.equal(actual, expected)


def test_grid_spatial_pool_rows_mismatch_raises():
    # rows 断言在 gh*gw 维度：给的矛盾会触发 ValueError。
    with pytest.raises(ValueError, match="≠"):
        # rows=12 却声明 gh×gw=4×4=16
        grid_spatial_pool(torch.zeros(16, 3), rows=12, gh=4, gw=4, k=2)


def test_grid_spatial_pool_token_row_mismatch_raises():
    # 实际 token 行数与 gh*gw 不符：reshape 元素数对不上应报错（ValueError）。
    # 实现封装为显式 ValueError 更友好，这里仅证明不会静默出错。
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        # tokens 只有 8 行，但声明 4×4=16 网格
        grid_spatial_pool(torch.zeros(8, 3), rows=16, gh=4, gw=4, k=2)
    assert excinfo.value


def test_per_image_rows_and_geo_unmerged():
    rows, geo = per_image_rows_and_geo([(1, 28, 28)], 2, 784)
    assert rows == [784] and geo == [(28, 28)]
    rows, geo = per_image_rows_and_geo([(1, 28, 28)], 2, 196)
    assert rows == [196] and geo == [(14, 14)]


def test_per_image_rows_and_geo_multitile():
    rows, geo = per_image_rows_and_geo([(2, 28, 28)], 2, 2 * 28 * 28)
    assert rows == [1568] and geo == [(28, 56)]
    rows, geo = per_image_rows_and_geo([(2, 28, 28)], 2, 2 * 14 * 14)
    assert rows == [392] and geo == [(14, 28)]


def test_per_image_invalid_rows_raises():
    with pytest.raises(ValueError):
        per_image_rows_and_geo([(1, 28, 28)], 2, 123)
