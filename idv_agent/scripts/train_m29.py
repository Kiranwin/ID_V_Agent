"""Train/evaluate M29 on v6 reviewed labels or independent prompt-Q exports."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from idv_agent.model.state_guided_action import StateGuidedActionNetwork, M29_SCHEMA, CAMERA_COMMANDS
from idv_agent.model.m29_checkpoint import load_m29_checkpoint
from idv_agent.training.m29_dataset import M29Dataset, collate_m29
from idv_agent.training.m29_features import load_m29_encoder
from idv_agent.training.m29_loss import compute_m29_loss, M29LossWeights
from idv_agent.vla.prompt_q import Q_CONTRACT


CLASS_COUNTS = {
    "visibility": 4,
    "phase": 5,
    "steering": 4,
    "path": 4,
    "move": 9,
    "camera_dx": 5,
    "camera_dy": 5,
}


def _mask_name(key):
    return "navigation" if key in {"move", "camera_dx", "camera_dy"} else key


def label_distributions(dataset):
    distributions = {}
    for key in ("q", "decoding", *CLASS_COUNTS):
        counter = Counter(
            int(sample["targets"][key]) for sample in dataset.samples
            if sample["masks"][_mask_name(key)]
        )
        distributions[key] = dict(sorted(counter.items()))
    return distributions


def validate_readiness(dataset, task, *, require_class_diversity=True):
    coverage = dataset.coverage()
    distributions = label_distributions(dataset)
    if set(distributions["q"]) != {0, 1}:
        raise ValueError(f"M29 {task} requires Q positive and negative labels: {distributions['q']}")
    if task == "q":
        return coverage, distributions
    required = ("navigation", "phase", "decoding", "visibility", "bbox", "prompt_bbox", "steering", "path", "q")
    missing = [key for key in required if coverage[key] == 0]
    if missing:
        raise ValueError(f"full M29 requires complete state/navigation supervision; missing={missing}, coverage={coverage}")
    if require_class_diversity and set(distributions["phase"]) != set(range(5)):
        raise ValueError(f"full M29 requires all five deployed phases: {distributions['phase']}")
    if require_class_diversity and set(distributions["decoding"]) != {0, 1}:
        raise ValueError(f"full M29 requires decoding positive and negative labels: {distributions['decoding']}")
    for key in (("visibility", "steering", "path", "move", "camera_dx")
                if require_class_diversity else ()):
        if len(distributions[key]) < 2:
            raise ValueError(f"full M29 requires class diversity for {key}: {distributions[key]}")
    return coverage, distributions


def make_class_weights(distributions, device):
    result = {}
    for key in ("q", "decoding"):
        negative, positive = distributions[key].get(0, 0), distributions[key].get(1, 0)
        if positive and negative:
            result[key] = torch.tensor(
                min(3.0, max(0.35, math.sqrt(negative / positive))),
                device=device, dtype=torch.float32,
            )
    for key, size in CLASS_COUNTS.items():
        counts = distributions[key]
        if not counts:
            continue
        weights = torch.zeros(size, device=device)
        total = sum(counts.values())
        for index, count in counts.items():
            weights[index] = min(3.0, max(0.35, math.sqrt(total / (len(counts) * count))))
        result[key] = weights
    return result


def make_group_sampler(dataset, task="full"):
    groups = []
    for sample in dataset.samples:
        if task == "q_interact":
            group = ("q_interact", int(sample["targets"]["q"]),
                     int(sample["targets"]["phase"]) if sample["masks"]["phase"] else -1)
        elif sample["q_only"]:
            group = "q_all_frames"
        elif sample["masks"]["navigation"]:
            # Preserve action diversity during overfit/full training. Grouping
            # all navigation rows together made the zero-camera/forward-move
            # majority dominate every minibatch despite class-weighted loss.
            group = ("full_navigation", int(sample["targets"]["move"]),
                     int(sample["targets"]["camera_dx"]),
                     int(sample["targets"]["camera_dy"]))
        else:
            group = "full_state_only"
        groups.append(group)
    counts = Counter(groups)
    weights = [1.0 / counts[group] for group in groups]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler, {str(key): value for key, value in sorted(counts.items(), key=lambda item: str(item[0]))}


def gradient_attribution(core, losses, params):
    groups = {
        "visual_temporal": ("visual_projection.", "time_projection.", "temporal.", "state_norm."),
        "facts": ("visibility.", "bbox.", "decoding.", "prompt_trunk.", "prompt_q.", "prompt_bbox."),
        "decision": ("task", "decision_trunk.", "phase.", "steering.", "path."),
        "navigation": ("navigation_trunk.", "move.", "camera_dx.", "camera_dy."),
    }
    names = [name for name, parameter in core.named_parameters() if parameter.requires_grad]
    result = {}
    components = [key for key in losses if key != "total"]
    for position, key in enumerate(components):
        gradients = torch.autograd.grad(
            losses[key], params, retain_graph=position < len(components) - 1,
            allow_unused=True,
        )
        row = {"all": math.sqrt(sum(float(grad.detach().float().square().sum())
                                    for grad in gradients if grad is not None))}
        for group, prefixes in groups.items():
            row[group] = math.sqrt(sum(
                float(grad.detach().float().square().sum())
                for name, grad in zip(names, gradients)
                if grad is not None and any(name == prefix or name.startswith(prefix) for prefix in prefixes)
            ))
        result[key] = row
    return result


def _metric(classes):
    return {"confusion": [[0] * classes for _ in range(classes)]}


def _update_metric(metric, targets, predictions):
    """Accumulate confusion[ground_truth][prediction]."""
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        metric["confusion"][int(target)][int(prediction)] += 1


def _finish_metric(metric):
    confusion = metric["confusion"]
    count = sum(map(sum, confusion))
    correct = sum(confusion[index][index] for index in range(len(confusion)))
    recalls = []
    for index, row in enumerate(confusion):
        support = sum(row)
        recalls.append(row[index] / support if support else None)
    supported = [value for value in recalls if value is not None]
    metric.update(count=count, correct=correct,
                  accuracy=correct / count if count else None,
                  recall=recalls,
                  balanced_accuracy=sum(supported) / len(supported) if supported else None)
    return metric


def evaluate(core, dataset, encoder, device, batch_size=8):
    core.eval()
    classes = {"q": 2, "decoding": 2, **CLASS_COUNTS}
    metrics = {key: _metric(size) for key, size in classes.items()}
    bbox = {"absolute_error_sum": 0.0, "coordinates": 0, "samples": 0}
    prompt_bbox = {"absolute_error_sum": 0.0, "coordinates": 0, "samples": 0}
    joint = {"correct": 0, "count": 0}
    per_phase = {str(index): {"correct": 0, "count": 0} for index in range(5)}
    interventions = {key: {"abs_change": 0., "argmax_flips": 0, "count": 0,
                           "base_joint_correct": 0, "altered_joint_correct": 0, "labeled": 0}
                     for key in ("visibility", "decoding", "phase", "steering", "path")}
    with torch.no_grad():
        for samples in DataLoader(dataset, batch_size=batch_size, collate_fn=list):
            inputs, targets, masks = collate_m29(samples, encoder, device=device)
            output = core(**inputs)
            logits = {"q": output.q_logits, "decoding": output.facts.decoding_logits,
                      "visibility": output.facts.visibility_logits,
                      "phase": output.decision.phase_logits,
                      "steering": output.decision.steering_logits,
                      "path": output.decision.path_logits,
                      "move": output.navigation.move_logits,
                      "camera_dx": output.navigation.camera_dx_logits, "camera_dy": output.navigation.camera_dy_logits}
            for key, value in logits.items():
                mask = masks[_mask_name(key)]
                pred = value.ge(0) if value.ndim == 1 else value.argmax(-1)
                _update_metric(metrics[key], targets[key][mask].long(), pred[mask].long())
            if bool(masks["bbox"].any()):
                difference = (output.facts.bbox[masks["bbox"]] - targets["bbox"][masks["bbox"]]).abs()
                bbox["absolute_error_sum"] += float(difference.sum())
                bbox["coordinates"] += difference.numel()
                bbox["samples"] += difference.shape[0]
            if bool(masks["prompt_bbox"].any()):
                difference = (output.facts.prompt_bbox[masks["prompt_bbox"]] -
                              targets["prompt_bbox"][masks["prompt_bbox"]]).abs()
                prompt_bbox["absolute_error_sum"] += float(difference.sum())
                prompt_bbox["coordinates"] += difference.numel()
                prompt_bbox["samples"] += difference.shape[0]
            nav_mask = masks["navigation"]
            base_predictions = torch.stack((output.navigation.move_logits.argmax(-1),
                                            output.navigation.camera_dx_logits.argmax(-1),
                                            output.navigation.camera_dy_logits.argmax(-1)), -1)
            nav_targets = torch.stack((targets["move"], targets["camera_dx"], targets["camera_dy"]), -1)
            base_joint = (base_predictions == nav_targets).all(-1)
            joint["correct"] += int(base_joint[nav_mask].sum())
            joint["count"] += int(nav_mask.sum())
            phase_mask = nav_mask & masks["phase"]
            for phase in range(5):
                selected = phase_mask & (targets["phase"] == phase)
                per_phase[str(phase)]["correct"] += int(base_joint[selected].sum())
                per_phase[str(phase)]["count"] += int(selected.sum())
            for key in interventions:
                facts, decision = output.facts, output.decision
                if key == "visibility":
                    facts = replace(facts, visibility_logits=facts.visibility_logits.roll(1, -1))
                    decision = core.decide_from_facts(facts, inputs["execution_feedback"], output.top_level_context)
                elif key == "decoding":
                    facts = replace(facts, decoding_logits=-facts.decoding_logits)
                    decision = core.decide_from_facts(facts, inputs["execution_feedback"], output.top_level_context)
                elif key == "phase":
                    decision = replace(decision, phase_logits=decision.phase_logits.roll(1, -1))
                elif key == "steering":
                    decision = replace(decision, steering_logits=decision.steering_logits.roll(1, -1))
                else:
                    decision = replace(decision, path_logits=decision.path_logits.roll(1, -1))
                nav = core.navigate_from_beliefs(torch.cat((facts.probabilities(), decision.probabilities()), -1), output.top_level_context)
                stats = interventions[key]
                altered_predictions = torch.stack((nav.move_logits.argmax(-1), nav.camera_dx_logits.argmax(-1),
                                                   nav.camera_dy_logits.argmax(-1)), -1)
                altered_joint = (altered_predictions == nav_targets).all(-1)
                for name in ("move", "camera_dx", "camera_dy"):
                    before, after = getattr(output.navigation, name+"_logits"), getattr(nav, name+"_logits")
                    stats["abs_change"] += float((before.softmax(-1)-after.softmax(-1)).abs().sum())
                    stats["argmax_flips"] += int((before.argmax(-1) != after.argmax(-1)).sum())
                    stats["count"] += before.shape[0]
                stats["base_joint_correct"] += int(base_joint[nav_mask].sum())
                stats["altered_joint_correct"] += int(altered_joint[nav_mask].sum())
                stats["labeled"] += int(nav_mask.sum())
    metrics = {key: _finish_metric(value) for key, value in metrics.items()}
    q_confusion = metrics["q"]["confusion"]
    q_tn, q_fp = q_confusion[0]
    q_fn, q_tp = q_confusion[1]
    bbox["mean_l1"] = bbox["absolute_error_sum"] / bbox["coordinates"] if bbox["coordinates"] else None
    prompt_bbox["mean_l1"] = (prompt_bbox["absolute_error_sum"] / prompt_bbox["coordinates"]
                               if prompt_bbox["coordinates"] else None)
    joint["accuracy"] = joint["correct"] / joint["count"] if joint["count"] else None
    for stats in per_phase.values():
        stats["accuracy"] = stats["correct"] / stats["count"] if stats["count"] else None
    return {"metrics": metrics, "bbox": bbox, "prompt_bbox": prompt_bbox, "joint_navigation": joint,
            "joint_navigation_by_target_phase": per_phase,
            "q": {"tp": q_tp, "fp": q_fp, "tn": q_tn, "fn": q_fn,
                  "recall": q_tp/(q_tp+q_fn) if q_tp+q_fn else None,
                  "false_positive_rate": q_fp/(q_fp+q_tn) if q_fp+q_tn else None},
            "state_interventions": interventions,
            "note": "Intervention sensitivity alone is not learned control success; use labeled accuracy degradation."}


def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    dataset = M29Dataset(args.data, expected_split="train")
    validation = M29Dataset(args.val_data, expected_split="val") if args.val_data else None
    if validation and (dataset.sessions & validation.sessions or dataset.groups & validation.groups) and not getattr(args, "allow_split_overlap", False):
        raise ValueError("train/val session or scenario-group overlap")
    coverage, distributions = validate_readiness(dataset, args.task)
    if args.task == "full" and validation is None:
        raise ValueError("full M29 requires an independent --val-data export")
    val_coverage = val_distributions = None
    if validation:
        val_coverage, val_distributions = validate_readiness(
            validation, args.task, require_class_diversity=False)
    output = Path(args.output)
    if output.exists():
        raise ValueError("output exists; use a new run directory")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable in selected Python environment")
    encoder = load_m29_encoder(args.model_path, device)
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if init_checkpoint:
        core, initial_manifest = load_m29_checkpoint(init_checkpoint, device=device)
        valid_transition = ((initial_manifest.get("task") == "q" and args.task == "full") or
                            (initial_manifest.get("task") == "full" and args.task == "q_interact"))
        if not valid_transition:
            raise ValueError("--init-checkpoint accepts Q->full or full->q_interact transitions")
        if core.feature_dim != encoder.adapter.act_feature_dim or core.hidden_dim != args.hidden_dim:
            raise ValueError("initial Q checkpoint model dimensions do not match full training")
        if initial_manifest.get("base_model") != str(Path(args.model_path).resolve()):
            raise ValueError("initial Q checkpoint and full training must use the same base model")
    else:
        core = StateGuidedActionNetwork(encoder.adapter.act_feature_dim, args.hidden_dim).to(device)
    if args.task == "q":
        for name, parameter in core.named_parameters():
            parameter.requires_grad_(name.startswith("prompt_trunk.") or
                                     name.startswith("prompt_q.") or
                                     name.startswith("prompt_bbox."))
    elif args.task == "q_interact":
        for name, parameter in core.named_parameters():
            parameter.requires_grad_(name.startswith("prompt_trunk.") or
                                     name.startswith("prompt_q.") or
                                     name.startswith("prompt_bbox.") or
                                     name.startswith("decision_trunk.") or
                                     name.startswith("phase."))
    # Train-only frozen-feature centering; exact buffer is checkpointed.
    calibration_paths = list(dict.fromkeys(p for sample in dataset.samples for p in sample["paths"]))[:64]
    if not init_checkpoint:
        core.feature_center.copy_(encoder(calibration_paths).float().mean(0).to(device))
    weights = (M29LossWeights() if args.task == "full" else
               M29LossWeights(navigation=0, phase=1.0, facts=0, route=0, q=2.0)
               if args.task == "q_interact" else
               M29LossWeights(navigation=0, phase=0, facts=0.25, route=0))
    weighted_distributions = distributions if args.task in {"full", "q_interact"} else {
        **distributions, "visibility": {}, "decoding": {}, "phase": {}, "steering": {},
        "path": {}, "move": {}, "camera_dx": {}, "camera_dy": {},
    }
    class_weights = make_class_weights(weighted_distributions, device)
    params = [p for p in core.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.)
    sampler, sampling_groups = make_group_sampler(dataset, args.task)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=list)
    iterator = iter(loader)
    output.mkdir(parents=True)
    history = []
    for step in range(args.steps):
        try:
            samples = next(iterator)
        except StopIteration:
            iterator = iter(loader); samples = next(iterator)
        inputs, targets, masks = collate_m29(samples, encoder, device=device)
        optimizer.zero_grad(set_to_none=True)
        core.train()
        losses = compute_m29_loss(core(**inputs), targets, masks, weights, class_weights)
        if not bool(torch.isfinite(losses["total"])):
            raise FloatingPointError("nonfinite M29 loss")
        scaler.scale(losses["total"]).backward(); scaler.unscale_(optimizer)
        grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if not math.isfinite(float(grad)):
            raise FloatingPointError("nonfinite M29 gradient norm")
        max_preclip_grad = getattr(args, "max_preclip_grad", 100.0)
        if max_preclip_grad is not None and float(grad) > max_preclip_grad:
            optimizer.zero_grad(set_to_none=True)
            diagnostic_losses = compute_m29_loss(core(**inputs), targets, masks, weights, class_weights)
            diagnostic = {
                "schema": "m29.gradient_failure.v1", "step": step + 1,
                "preclip_grad_norm": float(grad), "threshold": max_preclip_grad,
                "losses": {key: float(value.detach()) for key, value in diagnostic_losses.items()},
                "attribution": gradient_attribution(core, diagnostic_losses, params),
            }
            (output / "gradient_failure.json").write_text(
                json.dumps(diagnostic, indent=2), encoding="utf-8"
            )
            raise FloatingPointError(
                f"M29 pre-clip gradient peak {float(grad):.6f} exceeds {max_preclip_grad:.6f}"
            )
        scaler.step(optimizer); scaler.update()
        history.append({"step": step+1, "grad_norm": float(grad), **{k: float(v.detach()) for k,v in losses.items()}})
        if step == 0 or (step+1) % 10 == 0:
            print(f"[m29 {step+1}/{args.steps}] loss={history[-1]['total']:.5f} grad={float(grad):.3f}", flush=True)
    peak_grad = max(row["grad_norm"] for row in history)
    max_preclip_grad = getattr(args, "max_preclip_grad", 100.0)
    manifest = {"architecture": core.architecture_name, "model_config": {"feature_dim": core.feature_dim, "hidden_dim": core.hidden_dim},
                "hierarchy": "predicted_beliefs_only", "action_ms": 200, "camera_commands": list(CAMERA_COMMANDS),
                "state_bottleneck": "soft_continuous_geometry_v2",
                "q_contract": Q_CONTRACT, "base_model": str(Path(args.model_path).resolve()), "task": args.task,
                "coverage": coverage, "train_files": dataset.files, "val_files": validation.files if validation else [],
                "val_coverage": val_coverage, "val_label_distributions": val_distributions,
                "train_sessions": sorted(dataset.sessions), "train_scenario_groups": sorted(dataset.groups),
                "val_sessions": sorted(validation.sessions) if validation else [],
                "val_scenario_groups": sorted(validation.groups) if validation else [],
                "precision": "frozen_Qwen_FP16+FP32_heads+GradScaler" if device.type=="cuda" else "FP32",
                "weights": asdict(weights), "steps": args.steps, "lr": args.lr, "seed": args.seed,
                "label_distributions": distributions, "sampling_groups": sampling_groups,
                "class_weights": {key: value.detach().cpu().tolist() for key, value in class_weights.items()},
                "initial_checkpoint": str(Path(init_checkpoint).resolve()) if init_checkpoint else None,
                "max_preclip_grad": max_preclip_grad, "peak_preclip_grad": peak_grad,
                "deployable": False, "diagnostic_random_split": bool(getattr(args, "allow_split_overlap", False)),
                "reason": "training checkpoint requires independent validation and sandbox gates"}
    torch.save({"schema": M29_SCHEMA, "model": core.state_dict(), "manifest": manifest}, output / "m29.pt")
    restored, _ = load_m29_checkpoint(output / "m29.pt", device=device)
    core.eval()
    with torch.no_grad():
        original, reloaded = core(**inputs), restored(**inputs)
        reload_delta = max(float((a-b).abs().max()) for a,b in
                           ((original.q_logits,reloaded.q_logits),(original.navigation.move_logits,reloaded.navigation.move_logits),
                            (original.navigation.camera_dx_logits,reloaded.navigation.camera_dx_logits),
                            (original.navigation.camera_dy_logits,reloaded.navigation.camera_dy_logits)))
    metrics = {"history": history, "checkpoint_reload_max_delta": reload_delta,
               "train": evaluate(core, dataset, encoder, device, args.batch_size),
               "val": evaluate(core, validation, encoder, device, args.batch_size) if validation else None}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"output": str(output), "coverage": coverage, "reload_delta": reload_delta, "deployable": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task", choices=("q", "full", "q_interact"), default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-preclip-grad", type=float, default=100.0)
    parser.add_argument("--allow-split-overlap", action="store_true",
                        help="仅诊断：允许从同一 session/group 随机抽取 validation；结果不代表泛化")
    args = parser.parse_args(argv)
    if (min(args.steps, args.batch_size, args.hidden_dim) < 1 or args.lr <= 0 or
            args.max_preclip_grad is not None and args.max_preclip_grad <= 0):
        parser.error("steps, batch size, hidden dimension, lr must be positive")
    args.allow_split_overlap = args.allow_split_overlap
    print(json.dumps(train(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
