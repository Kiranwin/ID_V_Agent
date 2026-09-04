"""ACT visual-grounding checkpoint and history-dropout contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _adapter(hidden_size: int = 5):
    from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stub = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=hidden_size, vocab_size=8))
    return Qwen3VLBackboneAdapter(Model())


def test_act_checkpoint_restores_only_raster_projector_and_condition():
    from idv_agent.model.act_checkpoint import act_adapter_state, load_act_adapter_state

    source = _adapter()
    source._ensure_spatial_cell_projector(4)
    state = act_adapter_state(source)
    assert set(state) == {"spatial_cell_projector", "condition_projection"}

    target = _adapter()
    load_act_adapter_state(target, state)
    assert target.spatial_cell_projector is not None
    assert torch.equal(target.spatial_cell_projector.position,
                       source.spatial_cell_projector.position)
    assert torch.equal(target.condition_projection.weight, source.condition_projection.weight)


def test_act_checkpoint_rejects_legacy_spatial_aggregator():
    from idv_agent.model.act_checkpoint import load_act_adapter_state

    with pytest.raises(ValueError, match="legacy|旧"):
        load_act_adapter_state(_adapter(), {"spatial_agg": {}})


def test_act_checkpoint_schema_rejects_layernorm_cell_projector_contract():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    checkpoint = {"checkpoint_schema_version": "m4_act.visual_grounded.v1",
                  "adapter": {}, "core": {}}

    with pytest.raises(ValueError, match="m7_act"):
        load_visual_grounded_act_checkpoint(adapter, core, checkpoint)


def test_act_checkpoint_rejects_previous_m6_visual_residual_schema():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    checkpoint = {"checkpoint_schema_version": "m6_act.visual_decision_residual.v1",
                  "adapter": {}, "core": {}}

    with pytest.raises(ValueError, match="m7_act"):
        load_visual_grounded_act_checkpoint(adapter, core, checkpoint)


def test_history_dropout_is_training_opt_in_and_batchwise():
    from idv_agent.scripts.train_vla import _drop_history_actions

    history = torch.ones(3, 72)
    unchanged, mask = _drop_history_actions(history, probability=0.0)
    assert torch.equal(unchanged, history)
    assert not mask.any()
    dropped, mask = _drop_history_actions(history, probability=1.0)
    assert mask.all()
    assert torch.equal(dropped, torch.zeros_like(history))
    with pytest.raises(ValueError, match="history_dropout"):
        _drop_history_actions(history, probability=1.1)


def test_act_trainable_parameters_exclude_uninitialized_legacy_projection():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.scripts.train_vla import _named_act_trainable

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    names = [name for name, _parameter in _named_act_trainable(adapter, core)]
    assert not any(name.startswith("adapter.visual_projection") for name in names)
    assert "adapter.condition_projection.weight" in names
    assert "core.fast_temporal.input_residual.weight" in names
