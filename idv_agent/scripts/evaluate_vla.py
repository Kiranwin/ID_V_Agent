"""Offline evaluation for a trained ACT action-chunk checkpoint.

This command only reads recorded frames and writes a JSON metrics report.  It
never opens the game window and never sends keyboard/mouse input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.scripts.train_vla import (_bounded_subset, _dataset_paths, _load_act_backbone,
                                          encode_batch, _model_inputs)
from idv_agent.training.checkpoint_manifest import load_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset
from idv_agent.training.vla_loss import compute_vla_loss
from idv_agent.vla.action_chunk import INTENTS


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    dataset = VLASequenceDataset(_dataset_paths(args.data), verify_images=True)
    if args.max_samples > 0:
        dataset = _bounded_subset(dataset, args.max_samples)
    loader = DataLoader(dataset, batch_size=max(1, min(args.batch_size, 2)), shuffle=False,
                        collate_fn=VLASequenceCollator(max_frames=8))

    adapter, _, parent_manifest = _load_act_backbone(
        args.model_path, args.init_checkpoint,
        dtype=torch.float16 if amp_enabled else torch.float32, device=device)
    act_manifest = load_manifest(Path(args.checkpoint).parent / "manifest.json",
                                 require_artifacts=True)
    if act_manifest["stage"] != "M3_ACT":
        raise ValueError(f"评估 checkpoint 必须是 M3_ACT，收到 {act_manifest['stage']}")
    for name in ("visual_projection", "condition_projection"):
        getattr(adapter, name).to(device=device, dtype=torch.float32)
    core = SharedFastSlowVLA(adapter.hidden_size, temporal_dim=args.temporal_dim).to(
        device=device, dtype=torch.float32)

    first = next(iter(loader))
    with torch.no_grad():
        encode_batch(adapter, first, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "adapter" not in checkpoint or "core" not in checkpoint:
        raise ValueError(f"ACT checkpoint 缺少 adapter/core: {args.checkpoint}")
    adapter.visual_projection.load_state_dict(checkpoint["adapter"]["visual_projection"])
    adapter.condition_projection.load_state_dict(checkpoint["adapter"]["condition_projection"])
    core.load_state_dict(checkpoint["core"])
    adapter.eval()
    core.eval()
    frame_cache: dict[str, torch.Tensor] = {}

    sums = {"loss": 0.0, "loss_count": 0, "move_correct": 0, "move_total": 0,
            "move_nonstop_correct": 0, "move_nonstop_total": 0,
            "camera_dx_correct": 0, "camera_dy_correct": 0, "camera_total": 0,
            "button_correct": 0, "button_total": 0, "button_exact": 0,
            "button_pressed_tp": 0, "button_pressed_pred": 0, "button_pressed_target": 0,
            "duration_abs": 0.0, "duration_total": 0,
            "intent_correct": 0, "intent_total": 0, "intent_buckets": {name: [0, 0] for name in INTENTS}}
    with torch.no_grad():
        for batch in loader:
            model_batch = _model_inputs(batch, device)
            features = encode_batch(adapter, batch, device=device, frame_cache=frame_cache)
            condition = core.initial_condition(features.shape[0], device=device, mode_id=0)
            condition.mode_id = model_batch["mode_id"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = core(features, condition,
                              valid_mask=model_batch["frame_valid_mask"],
                              time_deltas=model_batch["time_deltas"], run_slow=True)
                losses = compute_vla_loss(output.fast, output.slow, model_batch)
            sample_mask = model_batch["fast_loss_mask"] > 0
            move_pred = output.fast.move_logits.argmax(-1)
            dx_pred = output.fast.camera_dx_logits.argmax(-1)
            dy_pred = output.fast.camera_dy_logits.argmax(-1)
            buttons_pred = (output.fast.button_logits.sigmoid() >= 0.5).to(torch.float32)
            move_target = model_batch["move_target"]
            dx_target = model_batch["camera_dx_target"]
            dy_target = model_batch["camera_dy_target"]
            button_target = model_batch["button_target"]
            mask = sample_mask.unsqueeze(1).expand_as(move_target)
            sums["loss"] += float(losses["total"].detach().cpu()) * int(sample_mask.sum())
            sums["loss_count"] += int(sample_mask.sum())
            sums["move_correct"] += int(((move_pred == move_target) & mask).sum())
            sums["move_total"] += int(mask.sum())
            non_stop = mask & (move_target != 0)
            sums["move_nonstop_correct"] += int(((move_pred == move_target) & non_stop).sum())
            sums["move_nonstop_total"] += int(non_stop.sum())
            sums["camera_dx_correct"] += int(((dx_pred == dx_target) & mask).sum())
            sums["camera_dy_correct"] += int(((dy_pred == dy_target) & mask).sum())
            sums["camera_total"] += int(mask.sum())
            button_mask = sample_mask[:, None, None].expand_as(button_target)
            sums["button_correct"] += int(((buttons_pred == button_target) & button_mask).sum())
            sums["button_total"] += int(button_mask.sum())
            sums["button_pressed_tp"] += int(((buttons_pred == 1) & (button_target == 1) & button_mask).sum())
            sums["button_pressed_pred"] += int(((buttons_pred == 1) & button_mask).sum())
            sums["button_pressed_target"] += int(((button_target == 1) & button_mask).sum())
            sums["button_exact"] += int((((buttons_pred == button_target).all(-1)) & mask).sum())
            duration_mask = sample_mask[:, None].expand_as(model_batch["duration_target"])
            sums["duration_abs"] += float((output.fast.duration - model_batch["duration_target"]).abs()[duration_mask].sum().cpu())
            sums["duration_total"] += int(duration_mask.sum())
            valid_intent = model_batch["intent_target"] != -100
            intent_mask = valid_intent & sample_mask
            sums["intent_correct"] += int(((output.slow.intent_logits.argmax(-1) == model_batch["intent_target"]) & intent_mask).sum())
            sums["intent_total"] += int(intent_mask.sum())
            for target, predicted, valid in zip(model_batch["intent_target"].tolist(),
                                                output.slow.intent_logits.argmax(-1).tolist(),
                                                intent_mask.tolist()):
                if valid:
                    bucket = sums["intent_buckets"][INTENTS[target]]
                    bucket[1] += 1
                    bucket[0] += int(target == predicted)

    def ratio(n: int, d: int) -> float | None:
        return float(n / d) if d else None

    records = len(dataset)
    return {
        "data": [str(args.data)], "samples": records,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "parent_stage": parent_manifest["stage"],
        "unique_frames_encoded": len(frame_cache),
        "loss": (sums["loss"] / sums["loss_count"] if sums["loss_count"] else None),
        "move_accuracy": ratio(sums["move_correct"], sums["move_total"]),
        "move_nonstop_accuracy": ratio(sums["move_nonstop_correct"], sums["move_nonstop_total"]),
        "camera_dx_accuracy": ratio(sums["camera_dx_correct"], sums["camera_total"]),
        "camera_dy_accuracy": ratio(sums["camera_dy_correct"], sums["camera_total"]),
        "button_element_accuracy": ratio(sums["button_correct"], sums["button_total"]),
        "button_exact_accuracy": ratio(sums["button_exact"], sums["move_total"]),
        "button_pressed_precision": ratio(sums["button_pressed_tp"], sums["button_pressed_pred"]),
        "button_pressed_recall": ratio(sums["button_pressed_tp"], sums["button_pressed_target"]),
        "duration_mae_frames": ratio(int(sums["duration_abs"] * 1_000_000), sums["duration_total"] * 1_000_000),
        "intent_accuracy": ratio(sums["intent_correct"], sums["intent_total"]),
        "intent_buckets": {
            name: {"correct": values[0], "total": values[1],
                   "accuracy": ratio(values[0], values[1])}
            for name, values in sums["intent_buckets"].items() if values[1]
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="一个或多个 vla_chunks_v4.jsonl（逗号/分号分隔）")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-checkpoint", required=True, help="M2_VG 目录")
    parser.add_argument("--checkpoint", required=True, help="M3_ACT act_*.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--output", default="reports/vla_eval.json")
    args = parser.parse_args(argv)
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
