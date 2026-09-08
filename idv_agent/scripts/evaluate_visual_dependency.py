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
from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint
from idv_agent.scripts.train_vla import (
    _dataset_paths,
    _evaluation_stratified_subset,
    _load_act_base_backbone,
    _model_inputs,
    encode_batch,
)
from idv_agent.training.checkpoint_manifest import load_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset


REQUIRED_METRICS = ("move_accuracy", "camera_accuracy", "intent_accuracy")
CONDITIONS = ("normal", "image_zero", "image_shuffle", "history_zero")
GROUNDING_SIDES = ("none", "left", "center", "right")


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


def _shuffle_frame_paths(frame_paths_by_sample: list[list[Path | None]], *,
                         episode_ids: list[str] | None = None) -> list[list[Path | None]]:
    """Replace every visual window with one from another session deterministically."""
    if len(frame_paths_by_sample) < 2:
        raise ValueError("image_shuffle 至少需要两个样本")
    if episode_ids is None:
        return [list(frame_paths_by_sample[(index + 1) % len(frame_paths_by_sample)])
                for index in range(len(frame_paths_by_sample))]
    if len(episode_ids) != len(frame_paths_by_sample):
        raise ValueError("episode_ids 与 image_shuffle 样本数不一致")
    groups: dict[str, list[int]] = {}
    for index, episode_id in enumerate(episode_ids):
        groups.setdefault(str(episode_id), []).append(index)
    episodes = list(groups)
    if len(episodes) < 2:
        raise ValueError("image_shuffle 至少需要两个不同 session")
    shuffled: list[list[Path | None] | None] = [None] * len(frame_paths_by_sample)
    for episode_index, episode_id in enumerate(episodes):
        source_indices = groups[episode_id]
        target_indices = groups[episodes[(episode_index + 1) % len(episodes)]]
        for offset, source_index in enumerate(source_indices):
            shuffled[source_index] = list(frame_paths_by_sample[target_indices[offset % len(target_indices)]])
    return [paths for paths in shuffled if paths is not None]


def _zero_history(batch: dict[str, Any]) -> dict[str, Any]:
    """Copy a collated batch and clear only its action-history input."""
    result = dict(batch)
    result["history_actions"] = batch["history_actions"].clone().zero_()
    return result


def _override_camera_prior_scale(core: SharedFastSlowVLA, value: float | None) -> None:
    """Apply the m26 camera-only protocol override after restore."""
    if value is None:
        return
    scale = float(value)
    if scale != 0.0:
        raise ValueError("m26 ACT 已废除 camera prior；camera-prior-scale 只能为 0")
    core.camera_prior_scale = scale


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


def _macro_recall(confusion: list[list[int]]) -> float | None:
    """Macro recall over observed classes; ignore classes absent from targets."""
    recalls = []
    for index, row in enumerate(confusion):
        total = sum(int(value) for value in row)
        if total:
            recalls.append(float(row[index]) / total)
    return sum(recalls) / len(recalls) if recalls else None


def _dependency_gate(normal: dict[str, Any], degraded: dict[str, Any],
                     required_metrics: Iterable[str] = REQUIRED_METRICS) -> dict[str, Any]:
    """Check that every degraded metric drops by max(0.05, 15% of normal)."""
    metrics: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    metric_alias = {
        "move_accuracy": "balanced_move_accuracy",
        "camera_accuracy": "balanced_camera_accuracy",
        "intent_accuracy": "balanced_intent_accuracy",
    }
    for name in required_metrics:
        selected = metric_alias.get(name, name)
        baseline = _finite_metric(normal.get(selected, normal.get(name)), f"normal.{selected}")
        observed = _finite_metric(degraded.get(selected, degraded.get(name)), f"degraded.{selected}")
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


def _output_change_summary(normal: dict[str, torch.Tensor],
                           changed: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    """Measure actual decision changes before accuracy hides them behind ties."""
    if set(normal) != set(changed):
        raise ValueError("normal/changed output heads 必须一致")
    summary = {}
    for name, baseline in normal.items():
        degraded = changed[name]
        if baseline.shape != degraded.shape or baseline.ndim < 2:
            raise ValueError(f"{name} logits shape 不一致或缺少类别维")
        summary[name] = {
            "mean_abs_logit_delta": float((baseline - degraded).abs().mean().cpu()),
            "argmax_flip_rate": float((baseline.argmax(dim=-1) != degraded.argmax(dim=-1)).float().mean().cpu()),
        }
    return summary


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


def _empty_grounding_camera_group() -> dict[str, Any]:
    """Allocate a compact per-observation-group camera accounting record."""
    return {
        "samples": 0, "dx_correct": 0, "dy_correct": 0,
        "dx_target_counts": [0] * 5, "dy_target_counts": [0] * 5,
        "dx_pred_counts": [0] * 5, "dy_pred_counts": [0] * 5,
        "dx_confusion": [[0] * 5 for _ in range(5)],
        "dy_confusion": [[0] * 5 for _ in range(5)],
    }


def _record_grounding_camera_groups(groups: dict[str, dict[str, Any]], *,
                                    batch: dict[str, Any], sample_mask: torch.Tensor,
                                    dx_prediction: torch.Tensor, dy_prediction: torch.Tensor) -> None:
    """Score h=0 camera predictions by human same-frame visual annotations only."""
    for index in torch.nonzero(sample_mask.bool(), as_tuple=False).flatten().tolist():
        if not bool(batch["grounding_mask"][index].item()):
            continue
        side = GROUNDING_SIDES[int(batch["grounding_side_target"][index].item())]
        prompt = int(float(batch["grounding_prompt_target"][index].item()) > 0)
        reachable = int(float(batch["grounding_reachable_target"][index].item()) > 0)
        names = ("annotated", f"side={side}", f"prompt={prompt}", f"reachable={reachable}")
        dx_target = int(batch["camera_dx_target"][index, 0].item())
        dy_target = int(batch["camera_dy_target"][index, 0].item())
        dx_pred = int(dx_prediction[index, 0].item())
        dy_pred = int(dy_prediction[index, 0].item())
        for name in names:
            group = groups.setdefault(name, _empty_grounding_camera_group())
            group["samples"] += 1
            group["dx_correct"] += int(dx_pred == dx_target)
            group["dy_correct"] += int(dy_pred == dy_target)
            group["dx_target_counts"][dx_target] += 1
            group["dy_target_counts"][dy_target] += 1
            group["dx_pred_counts"][dx_pred] += 1
            group["dy_pred_counts"][dy_pred] += 1
            group["dx_confusion"][dx_target][dx_pred] += 1
            group["dy_confusion"][dy_target][dy_pred] += 1


def _grounding_camera_group_metrics(groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert raw group counts into compact accuracy and macro-recall summaries."""
    result = {}
    for name, group in sorted(groups.items()):
        samples = int(group["samples"])
        dx_accuracy = group["dx_correct"] / samples if samples else None
        dy_accuracy = group["dy_correct"] / samples if samples else None
        result[name] = {
            "samples": samples,
            "camera_dx_accuracy": dx_accuracy,
            "camera_dy_accuracy": dy_accuracy,
            "camera_accuracy": ((dx_accuracy + dy_accuracy) / 2
                                if dx_accuracy is not None and dy_accuracy is not None else None),
            "balanced_camera_accuracy": ((
                (_macro_recall(group["dx_confusion"]) or 0.0) +
                (_macro_recall(group["dy_confusion"]) or 0.0)
            ) / 2),
            "dx_target_counts": group["dx_target_counts"],
            "dy_target_counts": group["dy_target_counts"],
            "dx_pred_counts": group["dx_pred_counts"],
            "dy_pred_counts": group["dy_pred_counts"],
        }
    return result


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
              "camera_total": 0, "intent_correct": 0, "intent_total": 0,
              "visual_move_correct": 0, "visual_dx_correct": 0, "visual_dy_correct": 0,
              "visual_intent_correct": 0, "prior_move_correct": 0, "prior_dx_correct": 0,
              "prior_dy_correct": 0, "prior_intent_correct": 0}
    confusions = {"move": [[0] * 9 for _ in range(9)],
                  "camera_dx": [[0] * 5 for _ in range(5)],
                  "camera_dy": [[0] * 5 for _ in range(5)],
                  "intent": [[0] * 8 for _ in range(8)]}
    visual_abs_sum = 0.0
    visual_logit_sum = 0.0
    visual_count = 0
    visual_input_norm_sum = 0.0
    visual_input_abs_sum = 0.0
    visual_input_count = 0
    output_vectors: dict[str, list[torch.Tensor]] = {name: [] for name in
                                                       ("move", "camera_dx", "camera_dy", "intent")}
    grounding_camera_groups: dict[str, dict[str, Any]] = {}
    with torch.no_grad():
        for batch in loader:
            model_batch = _model_inputs(batch, device)
            if condition == "history_zero":
                model_batch["history_actions"] = torch.zeros_like(model_batch["history_actions"])
            features, visual_features = encode_batch(
                adapter, batch, device=device, image_transform=image_transform,
                frame_stats={"hits": 0, "encoded": 0}, return_visual=True)
            valid_indices = model_batch["frame_valid_mask"].long().sum(dim=1).clamp_min(1) - 1
            rows = torch.arange(visual_features.shape[0], device=device)
            last_visual_input = visual_features[rows, valid_indices].float()
            visual_input_norm_sum += float(last_visual_input.norm(dim=-1).sum().cpu())
            visual_input_abs_sum += float(last_visual_input.abs().sum().cpu())
            visual_input_count += int(last_visual_input.shape[0])
            condition_state = core.initial_condition(features.shape[0], device=device, mode_id=0)
            condition_state.mode_id = model_batch["mode_id"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                fast_pass = core(features, condition_state,
                                 valid_mask=model_batch["frame_valid_mask"],
                                 visual_frame_features=visual_features,
                                 history_actions=model_batch["history_actions"], run_slow=False)
            sample_mask = model_batch["fast_loss_mask"] > 0
            # m25 executes a single six-frame macro action before observing a
            # new image.  Gate accuracy must assess exactly that deployed
            # decision, never later predictions made from stale pixels.
            horizon = int(getattr(core, "execution_horizon", 1))
            if not 1 <= horizon <= model_batch["move_target"].shape[1]:
                raise ValueError("checkpoint execution_horizon 超出 action chunk")
            mask = sample_mask[:, None].expand_as(model_batch["move_target"]).clone()
            mask[:, horizon:] = False
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
            output_vectors["move"].append(fast_pass.fast.move_logits.detach().float().cpu())
            output_vectors["camera_dx"].append(fast_pass.fast.camera_dx_logits.detach().float().cpu())
            output_vectors["camera_dy"].append(fast_pass.fast.camera_dy_logits.detach().float().cpu())
            output_vectors["intent"].append(slow_pass.slow.intent_logits.detach().float().cpu())
            counts["intent_correct"] += int(((intent_pred == model_batch["intent_target"]) & intent_mask).sum())
            counts["intent_total"] += int(intent_mask.sum())
            for target, prediction in zip(model_batch["move_target"][mask].detach().cpu().reshape(-1).tolist(),
                                          move_pred[mask].detach().cpu().reshape(-1).tolist()):
                confusions["move"][int(target)][int(prediction)] += 1
            for key, target_key, prediction in (("camera_dx", "camera_dx_target", dx_pred),
                                                 ("camera_dy", "camera_dy_target", dy_pred)):
                for target, prediction_value in zip(model_batch[target_key][mask].detach().cpu().reshape(-1).tolist(),
                                                    prediction[mask].detach().cpu().reshape(-1).tolist()):
                    confusions[key][int(target)][int(prediction_value)] += 1
            _record_grounding_camera_groups(
                grounding_camera_groups, batch=model_batch, sample_mask=sample_mask,
                dx_prediction=dx_pred, dy_prediction=dy_pred)
            for target, prediction_value in zip(model_batch["intent_target"][intent_mask].detach().cpu().reshape(-1).tolist(),
                                                intent_pred[intent_mask].detach().cpu().reshape(-1).tolist()):
                confusions["intent"][int(target)][int(prediction_value)] += 1
            visual = fast_pass.visual
            prior = fast_pass.prior_fast
            if visual is None or prior is None:
                raise RuntimeError("visual dependency 需要 visual/prior branch output")
            visual_abs_sum += float(visual.feature.abs().sum().cpu())
            visual_logit_sum += float(visual.fast.move_logits.abs().sum().cpu())
            visual_count += int(visual.feature.numel())
            visual_move = visual.fast.move_logits.argmax(-1)
            visual_dx = visual.fast.camera_dx_logits.argmax(-1)
            visual_dy = visual.fast.camera_dy_logits.argmax(-1)
            prior_move = prior.move_logits.argmax(-1)
            prior_dx = prior.camera_dx_logits.argmax(-1)
            prior_dy = prior.camera_dy_logits.argmax(-1)
            counts["visual_move_correct"] += int(((visual_move == model_batch["move_target"]) & mask).sum())
            counts["visual_dx_correct"] += int(((visual_dx == model_batch["camera_dx_target"]) & mask).sum())
            counts["visual_dy_correct"] += int(((visual_dy == model_batch["camera_dy_target"]) & mask).sum())
            counts["prior_move_correct"] += int(((prior_move == model_batch["move_target"]) & mask).sum())
            counts["prior_dx_correct"] += int(((prior_dx == model_batch["camera_dx_target"]) & mask).sum())
            counts["prior_dy_correct"] += int(((prior_dy == model_batch["camera_dy_target"]) & mask).sum())
            counts["visual_intent_correct"] += int(((visual.slow.intent_logits.argmax(-1) ==
                                                        model_batch["intent_target"]) & intent_mask).sum())
            prior_slow = slow_pass.prior_slow
            if prior_slow is None:
                raise RuntimeError("visual dependency 需要 prior slow branch output")
            counts["prior_intent_correct"] += int(((prior_slow.intent_logits.argmax(-1) ==
                                                      model_batch["intent_target"]) & intent_mask).sum())

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
        "balanced_move_accuracy": _macro_recall(confusions["move"]),
        "balanced_camera_accuracy": ((
            (_macro_recall(confusions["camera_dx"]) or 0.0) +
            (_macro_recall(confusions["camera_dy"]) or 0.0)
        ) / 2),
        "balanced_intent_accuracy": _macro_recall(confusions["intent"]),
        "confusion": confusions,
        "counts": counts,
        "branches": {
            "visual": {"move_accuracy": ratio(counts["visual_move_correct"], counts["move_total"]),
                        "camera_accuracy": ratio(counts["visual_dx_correct"] + counts["visual_dy_correct"],
                                                  counts["camera_total"] * 2),
                        "intent_accuracy": ratio(counts["visual_intent_correct"], counts["intent_total"])},
            "prior": {"move_accuracy": ratio(counts["prior_move_correct"], counts["move_total"]),
                      "camera_accuracy": ratio(counts["prior_dx_correct"] + counts["prior_dy_correct"],
                                                counts["camera_total"] * 2),
                      "intent_accuracy": ratio(counts["prior_intent_correct"], counts["intent_total"])},
        },
        "visual_signatures": {
            "feature_abs_mean": visual_abs_sum / max(1, visual_count),
            "move_logit_abs_mean": visual_logit_sum / max(1, counts["move_total"] * 9),
            "last_visual_input_norm_mean": visual_input_norm_sum / max(1, visual_input_count),
            "last_visual_input_abs_mean": visual_input_abs_sum / max(1, visual_input_count * visual_features.shape[-1]),
        },
        "grounding_camera_groups": _grounding_camera_group_metrics(grounding_camera_groups),
        "_output_vectors": {name: torch.cat(values, dim=0) for name, values in output_vectors.items()},
    }


def evaluate_checkpoint(checkpoint: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """Evaluate one M3_ACT checkpoint under all four dependency conditions."""
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda"
    dataset = VLASequenceDataset(_dataset_paths(args.data), verify_images=True,
                                 grounding_annotations=(getattr(args, "grounding_annotations", "") or None))
    if args.max_samples > 0:
        dataset = _evaluation_stratified_subset(dataset, args.max_samples)
    samples = [dataset[index] for index in range(len(dataset))]
    if len(samples) < 2:
        raise ValueError("visual dependency gate 至少需要两个样本")
    batch_size = max(1, int(args.batch_size))
    adapter, _ = _load_act_base_backbone(
        args.model_path, dtype=torch.float16 if amp_enabled else torch.float32, device=device)
    act_manifest = load_manifest(Path(checkpoint).parent / "manifest.json", require_artifacts=True)
    if act_manifest["stage"] != "M3_ACT":
        raise ValueError(f"评估 checkpoint 必须是 M3_ACT，收到 {act_manifest['stage']}")
    adapter.condition_projection.to(device=device, dtype=torch.float32)
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=args.temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    first_loader = DataLoader(samples[:1], batch_size=1, collate_fn=VLASequenceCollator(max_frames=8))
    with torch.no_grad():
        encode_batch(adapter, next(iter(first_loader)), device=device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    load_visual_grounded_act_checkpoint(adapter, core, saved)
    _override_camera_prior_scale(core, getattr(args, "camera_prior_scale", None))
    adapter.eval()
    core.eval()

    paths = [list(sample["frame_paths"]) for sample in samples]
    shuffled = _shuffle_frame_paths(paths, episode_ids=[str(sample["episode_id"]) for sample in samples])
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
    output_changes = {
        condition: _output_change_summary(metrics["normal"]["_output_vectors"],
                                          metrics[condition]["_output_vectors"])
        for condition in ("image_zero", "image_shuffle", "history_zero")
    }
    for values in metrics.values():
        values.pop("_output_vectors")
    normal = metrics["normal"]
    image_gates = {
        condition: _dependency_gate(normal, metrics[condition])
        for condition in ("image_zero", "image_shuffle")
    }
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_step": int(saved.get("step", -1)),
        "camera_prior_scale": float(core.camera_prior_scale),
        "initialization": "base_without_m2",
        "data": [str(args.data)],
        "metrics": metrics,
        "output_changes": output_changes,
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
    parser.add_argument("--init-checkpoint", default="", help="已废弃：仅为旧命令兼容，不会加载 M2")
    parser.add_argument("--checkpoint", action="append", default=[], help="M3_ACT checkpoint，可重复")
    parser.add_argument("--checkpoint-dir", default="", help="批量评估目录下的 *.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=256)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--camera-prior-scale", type=float, default=None,
                        help="诊断时覆盖 camera prior 权重；默认使用模型配置")
    parser.add_argument("--grounding-annotations", default="",
                        help="可选同帧人工标注；只生成分层诊断，不参与模型输入或门禁")
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
