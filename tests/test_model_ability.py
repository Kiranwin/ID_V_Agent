"""SGan 模型能力评估入口（evaluate_model_ability）的接口契约：不需要 GPU。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def test_gate_rejects_legacy_act_checkpoint(tmp_path):
    from idv_agent.scripts.evaluate_model_ability import evaluate_checkpoint

    checkpoint = tmp_path / "act.pt"
    torch.save({"schema": "m4_act.visual_grounded.v1"}, checkpoint)
    with pytest.raises(ValueError, match="只接受"):
        evaluate_checkpoint(checkpoint, SimpleNamespace())


def test_gate_requires_checkpoint_argument():
    from idv_agent.scripts.evaluate_model_ability import _checkpoint_paths

    with pytest.raises(ValueError, match="至少提供一个"):
        _checkpoint_paths(SimpleNamespace(checkpoint=[], checkpoint_dir=""))


def test_gate_reports_missing_checkpoint_dir():
    from idv_agent.scripts.evaluate_model_ability import _checkpoint_paths

    with pytest.raises(ValueError, match="不存在"):
        _checkpoint_paths(SimpleNamespace(checkpoint=[], checkpoint_dir="no/such/dir"))
