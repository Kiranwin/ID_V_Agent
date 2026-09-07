"""Read-only feasibility scan for camera zero-bucket logit calibration.

The scan fits a bias for the existing visual camera logits using the training
split only, then reports the resulting zero-bucket false-turn rate and
non-zero-bucket recall on train/validation.  It does not modify a checkpoint,
labels, or runtime configuration.
"""

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
    _dataset_paths, _load_act_base_backbone, _model_inputs, _scheduled_condition,
    encode_batch,
)
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset


ZERO_CLASS = 2
NONZERO_CLASSES = (0, 1, 3, 4)


def _fit_zero_bias(logits: torch.Tensor, targets: torch.Tensor,
                   *, max_zero_false_turn_rate: float = 0.15) -> tuple[float, dict[str, Any]]:
    """Choose the least damaging bias that meets the train zero-turn limit."""
    if logits.ndim != 2 or logits.shape[-1] != 5:
        raise ValueError("camera logits 必须是 [N,5]")
    targets = targets.reshape(-1).to(torch.long)
    logits = logits.reshape(-1, 5).float()
    if logits.shape[0] != targets.shape[0] or logits.shape[0] < 1:
        raise ValueError("camera logits/targets 必须等长非空")
    if not 0.0 <= float(max_zero_false_turn_rate) <= 1.0:
        raise ValueError("max_zero_false_turn_rate 必须在 [0,1]")
    margins = logits[:, [0, 1, 3, 4]].max(dim=-1).values - logits[:, ZERO_CLASS]
    candidates = torch.unique(torch.cat((margins, margins.new_tensor([-10.0, 10.0])))).sort().values
    zero_mask = targets == ZERO_CLASS
    nonzero_mask = ~zero_mask
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for bias in candidates.tolist():
        adjusted = logits.clone()
        adjusted[:, ZERO_CLASS] += float(bias)
        predicted = adjusted.argmax(-1)
        zero_false = (predicted[zero_mask] != ZERO_CLASS).float().mean().item() if bool(zero_mask.any()) else 0.0
        recalls = []
        for cls in NONZERO_CLASSES:
            cls_mask = targets == cls
            recalls.append(float((predicted[cls_mask] == cls).float().mean()) if bool(cls_mask.any()) else 0.0)
        nonzero_macro = sum(recalls) / len(recalls)
        overall = float((predicted == targets).float().mean())
        if zero_false <= float(max_zero_false_turn_rate):
            # Prefer non-zero recall, then overall accuracy, then the smallest
            # bias.  This preserves useful rare camera buckets when possible.
            key = (nonzero_macro, overall, -abs(float(bias)))
            report = {"bias": float(bias), "zero_false_turn_rate": float(zero_false),
                      "nonzero_macro_recall": float(nonzero_macro),
                      "nonzero_recalls": recalls, "accuracy": overall,
                      "samples": int(targets.numel())}
            if best is None or key > best[0]:
                best = (key, float(bias), report)
    if best is None:
        raise ValueError("训练集不存在满足 zero_false_turn_rate 限制的 bias")
    return best[1], {"method": "train_argmax_zero_bias", "max_zero_false_turn_rate": float(max_zero_false_turn_rate),
                     "selected": best[2]}


def _metrics(logits: torch.Tensor, targets: torch.Tensor, *, bias: float) -> dict[str, Any]:
    adjusted = logits.float().clone()
    adjusted[:, ZERO_CLASS] += float(bias)
    predicted = adjusted.argmax(-1)
    confusion = [[0] * 5 for _ in range(5)]
    for target, value in zip(targets.tolist(), predicted.tolist()):
        confusion[int(target)][int(value)] += 1
    recall = []
    for row_index, row in enumerate(confusion):
        total = sum(row)
        recall.append(row[row_index] / total if total else None)
    zero_total = sum(confusion[ZERO_CLASS])
    return {"samples": int(targets.numel()), "confusion": confusion, "recall": recall,
            "zero_false_turn_rate": ((zero_total - confusion[ZERO_CLASS][ZERO_CLASS]) / zero_total
                                       if zero_total else None),
            "accuracy": float((predicted == targets).float().mean())}


def _collect(data: str, *, adapter: torch.nn.Module, core: SharedFastSlowVLA,
             device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = VLASequenceDataset(_dataset_paths(data), verify_images=True)
    loader = DataLoader(dataset, batch_size=max(1, min(int(batch_size), 2)), shuffle=False,
                        collate_fn=VLASequenceCollator(max_frames=8))
    logits_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            model_batch = _model_inputs(batch, device)
            features, visual_features = encode_batch(adapter, batch, device=device, return_visual=True)
            condition = core.initial_condition(features.shape[0], device=device, mode_id=0)
            condition.mode_id = model_batch["mode_id"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                slow_pass = core(features, condition, valid_mask=model_batch["frame_valid_mask"],
                                 visual_frame_features=visual_features,
                                 history_actions=model_batch["history_actions"], run_slow=True)
                if slow_pass.slow is None:
                    raise RuntimeError("slow pass 未产生输出")
                next_condition = _scheduled_condition(core, slow_pass.slow, model_batch,
                                                      teacher_forcing_ratio=0.0)
                fast_pass = core(features, next_condition, valid_mask=model_batch["frame_valid_mask"],
                                 visual_frame_features=visual_features,
                                 history_actions=model_batch["history_actions"], run_slow=False,
                                 detach_slow_condition=False)
            mask = model_batch["fast_loss_mask"].bool()
            logits_rows.append(fast_pass.fast.camera_dx_logits[:, 0].float().cpu()[mask.cpu()])
            target_rows.append(model_batch["camera_dx_target"][:, 0].long().cpu()[mask.cpu()])
    return torch.cat(logits_rows), torch.cat(target_rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    adapter, _ = _load_act_base_backbone(args.model_path,
                                          dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                          device=device)
    train_dataset = VLASequenceDataset(_dataset_paths(args.train_data), verify_images=True)
    first = next(iter(DataLoader(train_dataset, batch_size=1,
                                 collate_fn=VLASequenceCollator(max_frames=8))))
    with torch.no_grad():
        encode_batch(adapter, first, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    core = SharedFastSlowVLA(adapter.act_feature_dim,
                             temporal_dim=int(checkpoint.get("manifest", {}).get("training", {}).get("temporal_dim", 256)),
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    load_visual_grounded_act_checkpoint(adapter, core, checkpoint)
    adapter.eval(); core.eval()
    train_logits, train_targets = _collect(args.train_data, adapter=adapter, core=core,
                                            device=device, batch_size=args.batch_size)
    val_logits, val_targets = _collect(args.val_data, adapter=adapter, core=core,
                                       device=device, batch_size=args.batch_size)
    bias, fit = _fit_zero_bias(train_logits, train_targets,
                                max_zero_false_turn_rate=args.max_zero_false_turn_rate)
    return {"schema": "vla.camera_zero_calibration_diagnostic.v1",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "bias": bias, "fit": fit,
            "train": _metrics(train_logits, train_targets, bias=bias),
            "val": _metrics(val_logits, val_targets, bias=bias)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-zero-false-turn-rate", type=float, default=0.15)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
