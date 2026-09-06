"""Fail-closed adapter state for visual-grounded ACT checkpoints."""

from __future__ import annotations

from typing import Any

import torch

ACT_CHECKPOINT_SCHEMA = "m25_act.rolling_causal_step.v1"
_ACT_ADAPTER_KEYS = frozenset({"condition_projection"})


def act_adapter_state(adapter: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Serialize only the task-condition projection.

    The visual feature is deterministic raw-grid pooling, so no learned
    spatial projector is allowed in an m8 checkpoint.
    """
    projection = getattr(adapter, "condition_projection", None)
    if projection is None or not hasattr(projection, "weight"):
        raise RuntimeError("ACT checkpoint 需要已 materialize 的 condition_projection")
    return {"condition_projection": projection.state_dict()}


def load_act_adapter_state(adapter: torch.nn.Module, state: dict[str, Any]) -> None:
    """Restore the m8 adapter contract and reject all learned raster states."""
    supplied = set(state)
    if "spatial_cell_projector" in supplied or "spatial_agg" in supplied or "visual_projection" in supplied:
        raise ValueError("m7/M2/legacy visual projector checkpoint 不兼容 m19")
    if supplied != _ACT_ADAPTER_KEYS:
        raise ValueError(f"m19 ACT adapter state 必须恰好包含 {sorted(_ACT_ADAPTER_KEYS)}，收到 {sorted(supplied)}")
    projection_state = state["condition_projection"]
    try:
        output_dim, input_dim = projection_state["weight"].shape
    except (KeyError, ValueError) as exc:
        raise ValueError("condition_projection state 不完整") from exc
    if int(input_dim) != int(adapter.hidden_size) * 2:
        raise ValueError("condition_projection 输入维度与 text hidden_size 不匹配")
    if int(output_dim) != int(getattr(adapter, "act_feature_dim", -1)):
        raise ValueError("condition_projection 输出维度与 act_feature_dim 不匹配")
    adapter.condition_projection.load_state_dict(projection_state)


def load_visual_grounded_act_checkpoint(adapter: torch.nn.Module, core: torch.nn.Module,
                                        checkpoint: dict[str, Any]) -> None:
    """Restore a complete m8 ACT checkpoint or fail before partial loading."""
    if checkpoint.get("checkpoint_schema_version") != ACT_CHECKPOINT_SCHEMA:
        raise ValueError("旧 ACT checkpoint 不兼容；m19_act/m24 及更早四步协议不能加载 m25 rolling causal step")
    if not isinstance(checkpoint.get("adapter"), dict) or not isinstance(checkpoint.get("core"), dict):
        raise ValueError("ACT checkpoint 缺少 adapter/core")
    load_act_adapter_state(adapter, checkpoint["adapter"])
    core.load_state_dict(checkpoint["core"])
