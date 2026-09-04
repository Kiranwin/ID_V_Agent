"""Fail-closed k=8 deep-feature and left/right occlusion acceptance for ACT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint
from idv_agent.scripts.train_vla import (
    _dataset_paths, _evaluation_stratified_subset, _load_act_base_backbone, _model_inputs,
    _scheduled_condition, encode_batch,
)
from idv_agent.training.feature_activity import feature_activity_gate, summarize_feature_activity
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset


def _occlusion(side: str) -> Callable[[Image.Image], Image.Image]:
    if side not in {"left", "right"}:
        raise ValueError("occlusion side 必须是 left 或 right")

    def transform(image: Image.Image) -> Image.Image:
        result = image.copy()
        width, height = result.size
        draw = ImageDraw.Draw(result)
        if side == "left":
            draw.rectangle((0, 0, width // 2, height), fill=(0, 0, 0))
        else:
            draw.rectangle((width // 2, 0, width, height), fill=(0, 0, 0))
        return result
    return transform


def _decision_pass(adapter, core: SharedFastSlowVLA, batch: dict[str, Any], *,
                   device: torch.device, amp_enabled: bool,
                   image_transform: Callable[[Image.Image], Image.Image] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the deployed slow→fast decision path and expose final GRU state."""
    model_batch = _model_inputs(batch, device)
    features, visual_features = encode_batch(
        adapter, batch, device=device, image_transform=image_transform, return_visual=True)
    condition = core.initial_condition(features.shape[0], device=device, mode_id=0)
    condition.mode_id = model_batch["mode_id"]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        slow_pass = core(features, condition, valid_mask=model_batch["frame_valid_mask"],
                         visual_frame_features=visual_features,
                         history_actions=model_batch["history_actions"], run_slow=True)
        if slow_pass.slow is None:
            raise RuntimeError("feature activity 需要 slow output")
        next_condition = _scheduled_condition(core, slow_pass.slow, model_batch,
                                              teacher_forcing_ratio=0.0)
        fast_pass = core(features, next_condition, valid_mask=model_batch["frame_valid_mask"],
                         visual_frame_features=visual_features,
                         history_actions=model_batch["history_actions"], run_slow=False,
                         detach_slow_condition=False)
    fast = fast_pass.fast
    logits = torch.cat((fast.move_logits.flatten(1), fast.camera_dx_logits.flatten(1),
                        fast.camera_dy_logits.flatten(1), fast.button_logits.flatten(1),
                        slow_pass.slow.intent_logits), dim=-1)
    return fast_pass.temporal_feature.detach().float().cpu(), logits.detach().float().cpu()


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    dataset = VLASequenceDataset(_dataset_paths(args.data), verify_images=True)
    dataset = _evaluation_stratified_subset(dataset, int(args.max_samples))
    if len(dataset) < 2:
        raise ValueError("feature activity 至少需要两个验证场景")
    loader = DataLoader(dataset, batch_size=max(1, min(int(args.batch_size), 2)), shuffle=False,
                        collate_fn=VLASequenceCollator(max_frames=8))
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "adapter" not in saved or "core" not in saved:
        raise ValueError(f"ACT checkpoint 缺少 adapter/core: {args.checkpoint}")
    adapter, _ = _load_act_base_backbone(
        args.model_path, dtype=torch.float16 if amp_enabled else torch.float32, device=device)
    temporal_dim = int(saved.get("manifest", {}).get("training", {}).get("temporal_dim", args.temporal_dim))
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    first = next(iter(loader))
    with torch.no_grad():
        encode_batch(adapter, first, device=device)
    load_visual_grounded_act_checkpoint(adapter, core, saved)
    adapter.eval(); core.eval()
    values: dict[str, list[torch.Tensor]] = {key: [] for key in
                                              ("normal", "repeat", "left", "right",
                                               "normal_output", "left_output", "right_output")}
    with torch.no_grad():
        for batch in loader:
            normal, normal_output = _decision_pass(adapter, core, batch, device=device, amp_enabled=amp_enabled)
            repeated, _ = _decision_pass(adapter, core, batch, device=device, amp_enabled=amp_enabled)
            left, left_output = _decision_pass(adapter, core, batch, device=device, amp_enabled=amp_enabled,
                                                image_transform=_occlusion("left"))
            right, right_output = _decision_pass(adapter, core, batch, device=device, amp_enabled=amp_enabled,
                                                  image_transform=_occlusion("right"))
            for key, value in (("normal", normal), ("repeat", repeated), ("left", left), ("right", right),
                               ("normal_output", normal_output), ("left_output", left_output),
                               ("right_output", right_output)):
                values[key].append(value)
    summary = summarize_feature_activity(*(torch.cat(values[key]) for key in
                                           ("normal", "repeat", "left", "right", "normal_output",
                                            "left_output", "right_output")))
    gate = feature_activity_gate(summary)
    return {"schema": "vla.feature_activity.v1", "checkpoint": str(Path(args.checkpoint).resolve()),
            "data": _dataset_paths(args.data), "summary": summary, "gate": gate}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="验证 vla_chunks_v5.jsonl")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--output", default="reports/feature_activity.json")
    args = parser.parse_args(argv)
    report = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
