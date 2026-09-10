"""Strict M29 checkpoint contract; reject legacy and underspecified heads."""
from pathlib import Path

import torch

from idv_agent.model.state_guided_action import StateGuidedActionNetwork, M29_SCHEMA, CAMERA_COMMANDS
from idv_agent.vla.prompt_q import Q_CONTRACT


def load_m29_checkpoint(path, *, device="cpu"):
    saved = torch.load(Path(path), map_location=device, weights_only=False)
    if saved.get("schema") != M29_SCHEMA:
        raise ValueError("checkpoint is not M29; m28/v5 weights are incompatible")
    manifest = saved.get("manifest", {})
    if (manifest.get("camera_commands") != list(CAMERA_COMMANDS) or manifest.get("action_ms") != 200
            or manifest.get("q_contract") != Q_CONTRACT or manifest.get("hierarchy") != "predicted_beliefs_only"):
        raise ValueError("M29 checkpoint control contract mismatch")
    if manifest.get("state_bottleneck") not in {"discrete_straight_through_v1", "soft_continuous_geometry_v2"}:
        raise ValueError("M29 checkpoint state bottleneck contract mismatch")
    core = StateGuidedActionNetwork(**manifest["model_config"]).to(device)
    core.load_state_dict(saved["model"], strict=True)
    core.eval()
    return core, manifest
