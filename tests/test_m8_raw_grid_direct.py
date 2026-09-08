"""m8 ACT contract: deterministic raw-grid visual features are first-class."""

from types import SimpleNamespace

import pytest
import torch


def _adapter(*, hidden_size: int = 5, act_feature_dim: int = 16):
    from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stub = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=hidden_size, vocab_size=8),
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )

    return Qwen3VLBackboneAdapter(Model(), act_feature_dim=act_feature_dim)


def test_m8_raw_grid_pool_is_deterministic_2x2_and_has_4096_width():
    from idv_agent.model.qwen_backbone_adapter import pool_raw_spatial_grid

    raw = torch.arange(64 * 1024, dtype=torch.float32).reshape(1, 64, 1024)
    pooled = pool_raw_spatial_grid(raw)
    assert pooled.shape == (1, 4096)
    expected = raw.reshape(1, 8, 8, 1024).reshape(1, 2, 4, 2, 4, 1024).mean((2, 4))
    assert torch.equal(pooled, expected.reshape(1, 4096))


def test_m8_project_raw_features_preserves_quadrant_swap_signal_without_projector():
    from idv_agent.model.temporal import TaskConditionCache

    adapter = _adapter(act_feature_dim=16)
    raw = torch.zeros(1, 64, 4)
    raw[:, :16, 0] = 1.0
    swapped = raw.reshape(1, 8, 8, 4).roll(shifts=4, dims=1).reshape(1, 64, 4)
    task = TaskConditionCache(torch.zeros(5), torch.zeros(5), task_id="test")
    first = adapter.project_raw_features(raw, task)
    second = adapter.project_raw_features(swapped, task)
    assert first.shape == (1, 16)
    assert not torch.equal(first, second)
    assert adapter.spatial_cell_projector is None


def test_m8_core_uses_act_feature_dim_not_text_hidden_size():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    adapter = _adapter(hidden_size=5, act_feature_dim=16)
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=4, history_action_dim=0)
    assert core.conditioner.feature_dim == 16
    assert core.fast_visual_residual.in_features == 16


def test_m8_checkpoint_rejects_m7_schema():
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    adapter = _adapter(act_feature_dim=16)
    core = SharedFastSlowVLA(16, temporal_dim=4, history_action_dim=0)
    with pytest.raises(ValueError, match="m28"):
        load_visual_grounded_act_checkpoint(
            adapter, core,
            {"checkpoint_schema_version": "m7_act.visual_expert.v1",
             "adapter": {}, "core": {}},
        )


def test_m8_raw_cache_projection_matches_live_direct_pooling():
    from idv_agent.model.temporal import TaskConditionCache

    adapter = _adapter(act_feature_dim=16)
    task = TaskConditionCache(torch.zeros(5), torch.zeros(5), task_id="test")
    raw = torch.arange(64 * 4, dtype=torch.float32).reshape(1, 64, 4)
    live = adapter.project_raw_features(raw, task)
    cached = adapter.project_raw_features(raw.clone(), task)
    assert torch.equal(live, cached)
