"""Offline held-out evaluation for human camera-control semantic labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint
from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.scripts.train_vla import (
    _camera_control_metrics,
    _dataset_paths,
    _load_act_base_backbone,
    _model_inputs,
    encode_batch,
)
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    dataset = VLASequenceDataset(
        _dataset_paths(args.data), verify_images=True,
        camera_control_annotations=args.camera_control_annotations,
    )
    loader = DataLoader(dataset, batch_size=max(1, int(args.batch_size)), shuffle=False,
                        collate_fn=VLASequenceCollator(max_frames=8))
    adapter, _ = _load_act_base_backbone(
        args.model_path, dtype=torch.float16 if amp_enabled else torch.float32, device=device)
    adapter.condition_projection.to(device=device, dtype=torch.float32)
    first = next(iter(loader))
    with torch.no_grad():
        encode_batch(adapter, first, device=device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=int(args.temporal_dim),
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    load_visual_grounded_act_checkpoint(adapter, core, saved)
    adapter.eval(); core.eval()
    rows: dict[str, list[torch.Tensor]] = {}
    with torch.no_grad():
        for batch in loader:
            model_batch = _model_inputs(batch, device)
            features, visual_features = encode_batch(adapter, batch, device=device, return_visual=True)
            condition = core.initial_condition(features.shape[0], device=device, mode_id=0)
            condition.mode_id = model_batch["mode_id"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = core(features, condition, valid_mask=model_batch["frame_valid_mask"],
                              visual_frame_features=visual_features,
                              history_actions=model_batch["history_actions"], run_slow=False)
            control = output.visual.camera_control if output.visual is not None else None
            if control is None:
                raise RuntimeError("checkpoint lacks camera-control visual heads")
            values = {
                "phase_logits": control.phase_logits,
                "target_id_logits": control.target_id_logits,
                "steering_logits": control.steering_logits,
                "path_logits": control.path_logits,
                "desired_turn_dx_logits": control.desired_turn_dx_logits,
                "desired_turn_dy_logits": control.desired_turn_dy_logits,
                "mask": model_batch["camera_control_mask"],
                "phase_target": model_batch["camera_control_phase_target"],
                "target_id_target": model_batch["camera_control_target_id_target"],
                "steering_target": model_batch["camera_control_steering_target"],
                "path_target": model_batch["camera_control_path_target"],
                "desired_turn_dx_target": model_batch["desired_turn_dx_target"],
                "desired_turn_dy_target": model_batch["desired_turn_dy_target"],
            }
            for name, value in values.items():
                rows.setdefault(name, []).append(value.detach().cpu())
    report = _camera_control_metrics(**{name: torch.cat(values) for name, values in rows.items()})
    return {"schema": "vla.camera_control_eval.v1", "checkpoint": str(Path(args.checkpoint).resolve()),
            "data": _dataset_paths(args.data), "metrics": report}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--camera-control-annotations", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
