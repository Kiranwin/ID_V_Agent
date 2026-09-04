"""Diagnostics for ACT train/validation prediction and slow-layer signals.

This is read-only with respect to model/data.  It reports representative
prediction distributions, slow intent/subgoal losses, and the condition fed
to the fast head (teacher forcing disabled, matching deployment).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.act_checkpoint import load_visual_grounded_act_checkpoint
from idv_agent.scripts.train_vla import (_contiguous_subset, _bounded_subset, _stratified_subset, _dataset_paths,
                                          _load_act_base_backbone, _model_inputs,
                                          encode_batch, _scheduled_condition)
from idv_agent.training.checkpoint_manifest import load_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset
from idv_agent.training.vla_loss import compute_vla_loss
from idv_agent.vla.action_chunk import INTENTS
from idv_agent.configs.subgoal import SUBGOAL_NAMES


def _load(args):
    device = torch.device(args.device)
    amp = device.type == "cuda"
    adapter, _ = _load_act_base_backbone(
        args.model_path, dtype=torch.float16 if amp else torch.float32, device=device)
    manifest = load_manifest(Path(args.checkpoint).parent / "manifest.json", require_artifacts=True)
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=args.temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    adapter.condition_projection.to(device=device, dtype=torch.float32)
    # Materialize lazy projection exactly as evaluation does.
    probe = VLASequenceDataset(_dataset_paths(args.data), verify_images=True)
    probe = _bounded_subset(probe, min(1, len(probe)))
    loader = DataLoader(probe, batch_size=1, collate_fn=VLASequenceCollator(max_frames=8))
    with torch.no_grad():
        encode_batch(adapter, next(iter(loader)), device=device)
    saved = torch.load(args.checkpoint, map_location=device, weights_only=False)
    load_visual_grounded_act_checkpoint(adapter, core, saved)
    adapter.eval(); core.eval()
    return device, amp, adapter, core, {"stage": "base_without_m2"}, manifest


def _run(name: str, paths: str, *, device, amp, adapter, core, max_samples: int, cache: dict[tuple[str, int, str], torch.Tensor], sampling: str = "uniform", seed: int = 0, force_slow_mask: bool = False):
    dataset = VLASequenceDataset(_dataset_paths(paths), verify_images=True)
    if max_samples > 0:
        dataset = _stratified_subset(dataset, max_samples, seed=seed) if sampling == "stratified" else _contiguous_subset(dataset, max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=VLASequenceCollator(max_frames=8))
    stats = {"hits": 0, "encoded": 0}
    out: dict[str, Any] = {
        "name": name, "samples": len(dataset), "move_pred_counts": [0] * 9,
        "move_target_counts": [0] * 9, "slow_intent_pred_counts": [0] * len(INTENTS),
        "slow_intent_target_counts": [0] * len(INTENTS), "fast_condition_intent_counts": [0] * len(INTENTS),
        "slow_intent_loss_sum": 0.0, "slow_subgoal_loss_sum": 0.0, "slow_count": 0,
        "move_total": 0, "move_nonstop_pred": 0, "move_nonstop_total": 0,
        "button_pred_positive_counts": [0] * 6, "button_target_positive_counts": [0] * 6,
    }
    with torch.no_grad():
        for batch in loader:
            if force_slow_mask:
                batch["slow_loss_mask"] = torch.ones_like(batch["slow_loss_mask"])
            mb = _model_inputs(batch, device)
            features = encode_batch(adapter, batch, device=device, frame_cache=cache, frame_stats=stats)
            condition = core.initial_condition(1, device=device, mode_id=0)
            condition.mode_id = mb["mode_id"]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                slow_pass = core(features, condition, valid_mask=mb["frame_valid_mask"],
                                 history_actions=mb["history_actions"], run_slow=True)
                next_condition = _scheduled_condition(core, slow_pass.slow, mb, teacher_forcing_ratio=0.0)
                fast_pass = core(features, next_condition, valid_mask=mb["frame_valid_mask"],
                                 history_actions=mb["history_actions"], run_slow=False,
                                 detach_slow_condition=False)
                losses = compute_vla_loss(fast_pass.fast, slow_pass.slow, mb)
            intent_target = int(mb["intent_target"][0])
            intent_pred = int(slow_pass.slow.intent_logits.argmax(-1)[0])
            # Only rows with slow_loss_mask=1 contribute to the slow loss.
            # Counting every valid intent here would dilute CE by the ~90%
            # of chunks deliberately skipped by the low-frequency slow tick.
            slow_supervised = bool(mb["slow_loss_mask"][0].item() > 0 and intent_target >= 0)
            if intent_target >= 0:
                out["slow_intent_target_counts"][intent_target] += 1
                out["slow_intent_pred_counts"][intent_pred] += 1
            if slow_supervised:
                out["slow_intent_loss_sum"] += float(losses["slow_intent"].cpu())
                out["slow_subgoal_loss_sum"] += float(losses["slow_subgoal"].cpu())
                out["slow_count"] += 1
            cond_pred = int(next_condition.intent_id[0])
            out["fast_condition_intent_counts"][cond_pred] += 1
            pred = fast_pass.fast.move_logits.argmax(-1)[0].tolist()
            target = mb["move_target"][0].tolist()
            for p, t in zip(pred, target):
                out["move_pred_counts"][p] += 1; out["move_target_counts"][t] += 1
                out["move_total"] += 1
                if t != 0:
                    out["move_nonstop_total"] += 1
                    out["move_nonstop_pred"] += int(p != 0)
            button_pred = (fast_pass.fast.button_logits[0] > 0).to(torch.int64)
            button_target = mb["button_target"][0].to(torch.int64)
            for i in range(button_pred.shape[-1]):
                out["button_pred_positive_counts"][i] += int(button_pred[:, i].sum())
                out["button_target_positive_counts"][i] += int(button_target[:, i].sum())
    if out["slow_count"]:
        out["slow_intent_loss_mean"] = out["slow_intent_loss_sum"] / out["slow_count"]
        out["slow_subgoal_loss_mean"] = out["slow_subgoal_loss_sum"] / out["slow_count"]
    out["move_nonstop_prediction_rate_on_nonstop_targets"] = (
        out["move_nonstop_pred"] / out["move_nonstop_total"] if out["move_nonstop_total"] else None)
    out["interact_pred_positive"] = out["button_pred_positive_counts"][0]
    out["interact_target_positive"] = out["button_target_positive_counts"][0]
    total = stats["hits"] + stats["encoded"]
    out["frame_cache"] = {
        "hits": stats["hits"], "misses": stats["encoded"],
        "encoded": stats["encoded"],
        "hit_rate": (stats["hits"] / total if total else None),
        "unique_frames_encoded": len(cache),
    }
    out["slow_intent_pred_names"] = {INTENTS[i]: n for i, n in enumerate(out["slow_intent_pred_counts"]) if n}
    out["fast_condition_intent_names"] = {INTENTS[i]: n for i, n in enumerate(out["fast_condition_intent_counts"]) if n}
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--val-data", required=True)
    p.add_argument("--model-path", required=True); p.add_argument("--init-checkpoint", default="", help="已废弃：不会加载 M2")
    p.add_argument("--checkpoint", required=True); p.add_argument("--device", default="cuda")
    p.add_argument("--temporal-dim", type=int, default=256)
    p.add_argument("--max-samples", type=int, default=256,
                    help="每个 split 的连续样本数；连续抽样可复用相邻帧缓存")
    p.add_argument("--output", default="reports/act_diagnostics.json")
    p.add_argument("--sampling", choices=("uniform", "stratified"), default="uniform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-slow-mask", action="store_true")
    args = p.parse_args(argv)
    device, amp, adapter, core, parent, manifest = _load(args)
    cache: dict[tuple[str, int, str], torch.Tensor] = {}
    result = {"checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_step": int(torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("step", -1)),
              "manifest": manifest, "train": _run("train", args.data, device=device, amp=amp, adapter=adapter, core=core, max_samples=args.max_samples, cache=cache, sampling=args.sampling, seed=args.seed, force_slow_mask=args.force_slow_mask),
              "val": _run("val", args.val_data, device=device, amp=amp, adapter=adapter, core=core, max_samples=args.max_samples, cache=cache, sampling=args.sampling, seed=args.seed, force_slow_mask=args.force_slow_mask),
              "note": "fast_condition_intent uses teacher_forcing_ratio=0, matching deployment; historical step-250 distributions require per-step logging and are not recoverable from a final checkpoint."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
