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


def test_act_checkpoint_restores_only_condition_projection():
    from idv_agent.model.act_checkpoint import act_adapter_state, load_act_adapter_state

    source = _adapter()
    state = act_adapter_state(source)
    assert set(state) == {"condition_projection"}

    target = _adapter()
    load_act_adapter_state(target, state)
    assert torch.equal(target.condition_projection.weight, source.condition_projection.weight)


def test_act_checkpoint_rejects_legacy_spatial_aggregator():
    from idv_agent.model.act_checkpoint import load_act_adapter_state

    with pytest.raises(ValueError, match="m7|legacy|旧"):
        load_act_adapter_state(_adapter(), {"spatial_agg": {}})


def test_act_checkpoint_schema_rejects_layernorm_cell_projector_contract():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    checkpoint = {"checkpoint_schema_version": "m4_act.visual_grounded.v1",
                  "adapter": {}, "core": {}}

    with pytest.raises(ValueError, match="m27"):
        load_visual_grounded_act_checkpoint(adapter, core, checkpoint)


def test_act_checkpoint_rejects_previous_m6_visual_residual_schema():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    checkpoint = {"checkpoint_schema_version": "m6_act.visual_decision_residual.v1",
                  "adapter": {}, "core": {}}

    with pytest.raises(ValueError, match="m27"):
        load_visual_grounded_act_checkpoint(adapter, core, checkpoint)


def test_act_checkpoint_rejects_previous_m17_eventless_schema():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    checkpoint = {"checkpoint_schema_version": "m17_act.camera_visual_summary.v1",
                  "adapter": {}, "core": {}}

    with pytest.raises(ValueError, match="m27"):
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


def test_visual_center_is_checkpointed_in_core_state():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA

    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=0)
    center = torch.arange(45, dtype=torch.float32)
    core.visual_expert.set_input_center(center)
    restored = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=0)
    restored.load_state_dict(core.state_dict())
    assert torch.equal(restored.visual_expert.input_center,
                       center)


def test_m26_checkpoint_rejects_nonzero_camera_prior_manifest():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint

    adapter = _adapter()
    core = SharedFastSlowVLA(frame_feature_dim=5, temporal_dim=4, history_action_dim=72)
    checkpoint = {
        "checkpoint_schema_version": "m27_act.visual_camera_control.v1",
        "adapter": {"condition_projection": adapter.condition_projection.state_dict()},
        "core": core.state_dict(),
        "manifest": {"training": {"camera_prior_scale": 0.5}},
    }
    with pytest.raises(ValueError, match="禁止 camera prior"):
        load_visual_grounded_act_checkpoint(adapter, core, checkpoint)
