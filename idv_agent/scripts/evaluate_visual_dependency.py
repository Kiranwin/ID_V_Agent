"""Fail-closed visual-dependency acceptance for ACT checkpoints.

The evaluator runs the same samples through four inference conditions:
normal, image_zero, image_shuffle, and history_zero.  It is offline-only and
never sends game input.  A checkpoint is accepted only when both image
degradations reduce move, camera, and intent accuracy by the configured
minimum amount.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.scripts.train_vla import (
    _contiguous_subset,
    _dataset_paths,
    _load_act_backbone,
    _model_inputs,
    _scheduled_condition,
    encode_batch,
)
from idv_agent.training.checkpoint_manifest import load_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset


REQUIRED_METRICS = ("move_accuracy", "camera_accuracy", "intent_accuracy")
CONDITIONS = ("normal", "image_zero", "image_shuffle", "history_zero")


def _image_transform(condition: str) -> Callable[[Image.Image], Image.Image] | None:
    """Return the deterministic image transform for an evaluation condition."""
    if condition == "normal":
        return None
    if condition == "image_zero":
        return lambda image: Image.new("RGB", image.size, (0, 0, 0))
    if condition == "image_blur":
        return lambda image: image.filter(ImageFilter.GaussianBlur(radius=12))
    if condition in {"image_shuffle", "history_zero"}:
        return None
    raise ValueError(f"unknown condition: {condition}")


def _shuffle_frame_paths(frame_paths_by_sample: list[list[Path | None]]) -> list[list[Path | None]]:
    """Cyclically replace every sample's image window with another sample's."""
    if len(frame_paths_by_sample) < 2:
        raise ValueError("image_shuffle 至少需要两个样本")
    return [list(frame_paths_by_sample[(index + 1) % len(frame_paths_by_sample)])
            for index in range(len(frame_paths_by_sample))]


def _zero_history(batch: dict[str, Any]) -> dict[str, Any]:
    """Copy a collated batch and clear only its action-history input."""
    result = dict(batch)
    result["history_actions"] = batch["history_actions"].clone().zero_()
    return result


def _finite_metric(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是有限数字或 null") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数字或 null")
    return value


def _dependency_gate(normal: dict[str, Any], degraded: dict[str, Any],
                     required_metrics: Iterable[str] = REQUIRED_METRICS) -> dict[str, Any]:
    """Check that every degraded metric drops by max(0.05, 15% of normal)."""
    metrics: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for name in required_metrics:
        baseline = _finite_metric(normal.get(name), f"normal.{name}")
        observed = _finite_metric(degraded.get(name), f"degraded.{name}")
        if baseline is None or observed is None:
            failures.append(name)
            metrics[name] = {"normal": baseline, "degraded": observed,
                             "drop": None, "threshold": None, "pass": False}
            continue
        drop = baseline - observed
        threshold = max(0.05, baseline * 0.15)
        passed = drop >= threshold - 1e-12
        if not passed:
            failures.append(name)
        metrics[name] = {"normal": baseline, "degraded": observed,
                         "drop": drop, "threshold": threshold, "pass": passed}
    return {"pass": not failures, "metrics": metrics, "failures": failures,
            "rule": "drop >= max(0.05, normal * 0.15)"}


def _copy_samples(samples: list[dict[str, Any]], *, frame_paths: list[list[Path | None]] | None = None,
                  zero_history: bool = False) -> list[dict[str, Any]]:
    copied = []
    for index, sample in enumerate(samples):
        item = dict(sample)
        item["frame_paths"] = list(frame_paths[index] if frame_paths is not None
                                    else sample["frame_paths"])
        if zero_history:
            item["history_actions"] = sample["history_actions"].clone().zero_()
        copied.append(item)
    return copied


def _condition_metrics(adapter: torch.nn.Module, core: SharedFastSlowVLA,
                       samples: list[dict[str, Any]], *, condition: str,
                       device: torch.device, amp_enabled: bool, batch_size: int,
                       image_transform: Callable[[Image.Image], Image.Image] | None = None) -> dict[str, Any]:
    """Run one condition over a fixed list of already-selected samples."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition: {condition}")
    loader = DataLoader(samples, batch_size=max(1, min(batch_size, 2)), shuffle=False,
                        collate_fn=VLASequenceCollator(max_frames=8))
    counts = {"move_correct": 0, "move_total": 0, "dx_correct": 0, "dy_correct": 0,
              "camera_total": 0, "intent_correct": 0, "intent_total": 0}
    with torch.no_grad():
        for batch in loader:
            model_batch = _model_inputs(batch, device)
            if condition == "history_zero":
                model_batch["history_actions"] = torch.zeros_like(model_batch["history_actions"])
            features = encode_batch(adapter, batch, device=device,
                                    image_transform=image_transform,
                                    frame_stats={"hits": 0, "encoded": 0})
            condition_state = core.initial_condition(features.shape[0], device=device, mode_id=0)
            condition_state.mode_id = model_batch["mode_id"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                slow_pass = core(features, condition_state,
                                 valid_mask=model_batch["frame_valid_mask"],
                                 history_actions=model_batch["history_actions"], run_slow=True)
                if slow_pass.slow is None:
                    raise RuntimeError("slow pass 未产生 SlowVLAOutput")
                next_condition = _scheduled_condition(core, slow_pass.slow, model_batch,
                                                      teacher_forcing_ratio=0.0)
                fast_pass = core(features, next_condition,
                                 valid_mask=model_batch["frame_valid_mask"],
                                 history_actions=model_batch["history_actions"], run_slow=False,
                                 detach_slow_condition=False)
            sample_mask = model_batch["fast_loss_mask"] > 0
            mask = sample_mask[:, None].expand_as(model_batch["move_target"])
            move_pred = fast_pass.fast.move_logits.argmax(-1)
            dx_pred = fast_pass.fast.camera_dx_logits.argmax(-1)
            dy_pred = fast_pass.fast.camera_dy_logits.argmax(-1)
            counts["move_correct"] += int(((move_pred == model_batch["move_target"]) & mask).sum())
            counts["move_total"] += int(mask.sum())
            counts["dx_correct"] += int(((dx_pred == model_batch["camera_dx_target"]) & mask).sum())
            counts["dy_correct"] += int(((dy_pred == model_batch["camera_dy_target"]) & mask).sum())
            counts["camera_total"] += int(mask.sum())
            intent_mask = sample_mask & (model_batch["intent_target"] >= 0)
            intent_pred = slow_pass.slow.intent_logits.argmax(-1)
            counts["intent_correct"] += int(((intent_pred == model_batch["intent_target"]) & intent_mask).sum())
            counts["intent_total"] += int(intent_mask.sum())

    def ratio(n: int, d: int) -> float | None:
        return n / d if d else None

    dx_accuracy = ratio(counts["dx_correct"], counts["camera_total"])
    dy_accuracy = ratio(counts["dy_correct"], counts["camera_total"])
    return {
        "condition": condition,
        "samples": len(samples),
        "move_accuracy": ratio(counts["move_correct"], counts["move_total"]),
        "camera_dx_accuracy": dx_accuracy,
        "camera_dy_accuracy": dy_accuracy,
        "camera_accuracy": ((dx_accuracy + dy_accuracy) / 2
                             if dx_accuracy is not None and dy_accuracy is not None else None),
        "intent_accuracy": ratio(counts["intent_correct"], counts["intent_total"]),
        "counts": counts,
    }


def evaluate_checkpoint(checkpoint: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one M3_ACT checkpoint under all four dependency conditions."""
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    dataset = VLASequenceDataset(_dataset_paths(args.data), verify_images=True)
    if args.max_samples > 0:
        dataset = _contiguous_subset(dataset, args.max_samples)
    samples = [dataset[index] for index in range(len(dataset))]
    if len(samples) < 2:
        raise ValueError("visual dependency gate 至少需要两个样本")
    batch_size = max(1, int(args.batch_size))
    adapter, _, parent_manifest = _load_act_backbone(
        args.model_path, args.init_checkpoint,
        dtype=torch.float16 if amp_enabled else torch.float32, device=device)
    act_manifest = load_manifest(Path(checkpoint).parent / "manifest.json", require_artifacts=True)
    if act_manifest["stage"] != "M3_ACT":
        raise ValueError(f"评估 checkpoint 必须是 M3_ACT，收到 {act_manifest['stage']}")
    for name in ("visual_projection", "condition_projection"):
        getattr(adapter, name).to(device=device, dtype=torch.float32)
    core = SharedFastSlowVLA(adapter.hidden_size, temporal_dim=args.temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    first_loader = DataLoader(samples[:1], batch_size=1, collate_fn=VLASequenceCollator(max_frames=8))
    with torch.no_grad():
        encode_batch(adapter, next(iter(first_loader)), device=device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    if "adapter" not in saved or "core" not in saved:
        raise ValueError(f"ACT checkpoint 缺少 adapter/core: {checkpoint}")
    adapter.visual_projection.load_state_dict(saved["adapter"]["visual_projection"])
    adapter.condition_projection.load_state_dict(saved["adapter"]["condition_projection"])
    core.load_state_dict(saved["core"])
    adapter.eval()
    core.eval()

    paths = [list(sample["frame_paths"]) for sample in samples]
    shuffled = _shuffle_frame_paths(paths)
    condition_samples = {
        "normal": _copy_samples(samples),
        "image_zero": _copy_samples(samples),
        "image_shuffle": _copy_samples(samples, frame_paths=shuffled),
        "history_zero": _copy_samples(samples, zero_history=True),
    }
    metrics = {}
    for condition in CONDITIONS:
        metrics[condition] = _condition_metrics(
            adapter, core, condition_samples[condition], condition=condition,
            device=device, amp_enabled=amp_enabled, batch_size=batch_size,
            image_transform=_image_transform(condition),
        )
    normal = metrics["normal"]
    image_gates = {
        condition: _dependency_gate(normal, metrics[condition])
        for condition in ("image_zero", "image_shuffle")
    }
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_step": int(saved.get("step", -1)),
        "parent_stage": parent_manifest["stage"],
        "data": [str(args.data)],
        "metrics": metrics,
        "image_dependency_gates": image_gates,
        "visual_dependency_gate_pass": all(gate["pass"] for gate in image_gates.values()),
    }


def _checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.checkpoint]
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        if not directory.is_dir():
            raise ValueError(f"checkpoint 目录不存在: {directory}")
        paths.extend(sorted(directory.glob("*.pt")))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("至少提供一个 --checkpoint 或 --checkpoint-dir")
    return unique


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="v5 JSONL 或逗号/分号分隔的 JSONL")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-checkpoint", required=True, help="M2_VG checkpoint 目录")
    parser.add_argument("--checkpoint", action="append", default=[], help="M3_ACT checkpoint，可重复")
    parser.add_argument("--checkpoint-dir", default="", help="批量评估目录下的 *.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--output", default="reports/visual_dependency_gate.json")
    args = parser.parse_args(argv)
    results = [evaluate_checkpoint(path, args) for path in _checkpoint_paths(args)]
    report = {
        "schema": "vla.visual_dependency_gate.v1",
        "conditions": list(CONDITIONS),
        "rule": "image_zero and image_shuffle must each drop move/camera/intent by max(0.05, normal * 0.15)",
        "checkpoints": results,
        "visual_dependency_gate_pass": all(result["visual_dependency_gate_pass"] for result in results),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["visual_dependency_gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
