"""Fail-closed adapter state for visual-grounded ACT checkpoints."""

from __future__ import annotations

import math
from typing import Any

import torch

from idv_agent.model.spatial_cell_projector import SpatialCellProjector


ACT_CHECKPOINT_SCHEMA = "m7_act.visual_expert.v1"
_ACT_ADAPTER_KEYS = frozenset({"spatial_cell_projector", "condition_projection"})


def act_adapter_state(adapter: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    """Serialize exactly the ACT visual-grounding modules, never legacy poolers."""
    projector = getattr(adapter, "spatial_cell_projector", None)
    if projector is None:
        raise RuntimeError("ACT checkpoint 需要已 materialize 的 SpatialCellProjector")
    return {
        "spatial_cell_projector": projector.state_dict(),
        "condition_projection": adapter.condition_projection.state_dict(),
    }


def load_act_adapter_state(adapter: torch.nn.Module, state: dict[str, Any]) -> None:
    """Restore only m4 raster modules and reject every legacy ACT contract."""
    supplied = set(state)
    if "spatial_agg" in supplied or "visual_projection" in supplied:
        raise ValueError("旧 SpatialAgg/visual_projection ACT checkpoint 不兼容")
    if supplied != _ACT_ADAPTER_KEYS:
        raise ValueError(f"ACT adapter state 必须恰好包含 {sorted(_ACT_ADAPTER_KEYS)}，收到 {sorted(supplied)}")
    projector_state = state["spatial_cell_projector"]
    try:
        cell_dim, raw_dim = projector_state["cell_projection.weight"].shape
        n_cells = int(projector_state["position"].shape[1])
        output_dim, fusion_width = projector_state["fusion_projection.weight"].shape
    except (KeyError, ValueError) as exc:
        raise ValueError("SpatialCellProjector state 不完整") from exc
    k = math.isqrt(n_cells)
    if k * k != n_cells or fusion_width != n_cells * cell_dim:
        raise ValueError("SpatialCellProjector state 的 raster shape 非法")
    if int(output_dim) != int(adapter.hidden_size):
        raise ValueError("SpatialCellProjector output_dim 与 backbone hidden_size 不匹配")
    projector = SpatialCellProjector(dim=int(raw_dim), k=k, cell_dim=int(cell_dim),
                                     output_dim=int(output_dim)).to(adapter.device)
    projector.load_state_dict(projector_state)
    adapter.spatial_k = k
    adapter.spatial_cell_projector = projector
    adapter.condition_projection.load_state_dict(state["condition_projection"])


def load_visual_grounded_act_checkpoint(adapter: torch.nn.Module, core: torch.nn.Module,
                                        checkpoint: dict[str, Any]) -> None:
    """Restore a complete m7 ACT checkpoint or fail before partial loading."""
    if checkpoint.get("checkpoint_schema_version") != ACT_CHECKPOINT_SCHEMA:
        raise ValueError("旧 ACT checkpoint 不兼容；需要 m7_act.visual_expert.v1")
    if not isinstance(checkpoint.get("adapter"), dict) or not isinstance(checkpoint.get("core"), dict):
        raise ValueError("ACT checkpoint 缺少 adapter/core")
    load_act_adapter_state(adapter, checkpoint["adapter"])
    core.load_state_dict(checkpoint["core"])
