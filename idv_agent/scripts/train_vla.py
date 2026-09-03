"""Minimal end-to-end VLA v4 training loop.

The Qwen visual-language backbone is frozen.  Only the adapter projections,
FiLM conditioner, temporal GRU and fast/slow heads are optimized.  The CLI is
intentionally small so it can be used for a 16--32 sample over-fit check or a
short 20--100 step smoke run on the real session data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from idv_agent.model.fast_slow_vla import FastSlowVLAOutput, SharedFastSlowVLA
from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter, load_qwen3vl_backbone
from idv_agent.training.checkpoint_manifest import build_manifest, load_manifest, sha256_file, write_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset
from idv_agent.training.vla_loss import compute_vla_loss
from idv_agent.training.utils import set_seed
from idv_agent.configs.subgoal import SUBGOAL_NAMES
from idv_agent.vla.action_chunk import INTENTS


def _freeze_qwen(adapter: torch.nn.Module) -> None:
    """Freeze the language/vision backbone while keeping adapter modules trainable."""
    model = getattr(adapter, "model", None)
    if model is not None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def _device_dtype(device: torch.device) -> tuple[torch.dtype, bool]:
    use_amp = device.type == "cuda"
    return (torch.float16 if use_amp else torch.float32), use_amp


def _load_act_backbone(model_path: str | Path, init_checkpoint: str | Path,
                       *, dtype: torch.dtype, device: torch.device):
    """Load M2_VG's M1_WK parent and reuse its trained visual projection."""
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    init_dir = Path(init_checkpoint)
    parent_manifest = load_manifest(init_dir / "manifest.json", require_artifacts=True)
    if parent_manifest["stage"] != "M2_VG":
        raise ValueError(f"ACT 初始化必须使用 M2_VG，收到 {parent_manifest['stage']}")
    base = str(model_path or parent_manifest["base_model"])
    processor = AutoProcessor.from_pretrained(base)
    base_model = AutoModelForImageTextToText.from_pretrained(base, dtype=dtype, device_map=None).to(device)
    # M2_VG itself keeps lora_wk in its parent M1_WK checkpoint.
    parent_path = Path(parent_manifest["parent"])
    if not parent_path.is_absolute():
        parent_path = init_dir / parent_path
    wk_manifest = load_manifest(parent_path / "manifest.json", require_artifacts=True)
    model = PeftModel.from_pretrained(
        base_model, str(parent_path / wk_manifest["artifacts"]["adapter"]), is_trainable=False,
    ).to(device)
    adapter = Qwen3VLBackboneAdapter(model, processor=processor).to(device)
    projection_path = init_dir / parent_manifest["artifacts"]["grounding_head"]
    saved = torch.load(projection_path, map_location="cpu", weights_only=False)
    if "visual_projection" not in saved:
        raise ValueError(f"M2_VG grounding checkpoint 缺少 visual_projection: {projection_path}")
    adapter.visual_projection.load_state_dict(saved["visual_projection"])
    return adapter, processor, parent_manifest


def _load_images(paths: list[Path | None]) -> list[Image.Image | None]:
    images: list[Image.Image | None] = []
    for path in paths:
        if path is None:
            images.append(None)
        else:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
    return images


def _dataset_paths(value: str | list[str] | None) -> list[str]:
    """Parse one or more JSONL paths (comma/semicolon separated)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


def _bounded_subset(dataset, limit: int):
    """Select evenly spaced records so a cap does not bias toward episode start."""
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    if limit == 1:
        indices = [0]
    else:
        indices = [round(i * (len(dataset) - 1) / (limit - 1)) for i in range(limit)]
    return Subset(dataset, indices)


def _contiguous_subset(dataset, limit: int):
    """Return up to ``limit`` records in contiguous episode runs.

    If one episode is shorter than the requested cap, continue with the next
    run; each run remains contiguous so only the run boundary incurs misses.
    """
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    runs: list[tuple[int, int]] = []
    start = 0
    previous = dataset[0].get("episode_id")
    for index in range(1, len(dataset)):
        current = dataset[index].get("episode_id")
        if current != previous:
            runs.append((start, index))
            start, previous = index, current
    runs.append((start, len(dataset)))
    # Prefer the longest run, then continue in source order if needed.
    runs.sort(key=lambda run: (run[1] - run[0]), reverse=True)
    selected: list[int] = []
    for begin, end in runs:
        if len(selected) >= limit:
            break
        selected.extend(range(begin, min(end, begin + limit - len(selected))))
    return Subset(dataset, selected[:limit])


ACTION_STRATA = ("interact", "move", "other_key", "stop")
DEFAULT_STRATIFIED_RATIOS = {
    "interact": 0.35, "move": 0.35, "other_key": 0.15, "stop": 0.15,
}


def _action_stratum(sample: dict[str, Any]) -> str:
    """Classify one complete action chunk, with interaction taking priority."""
    buttons = sample["button_target"]
    if bool((buttons[:, 0] > 0).any()):  # BUTTON_NAMES[0] == interact/Q
        return "interact"
    if bool((buttons > 0).any()):
        return "other_key"
    if bool((sample["move_target"] != 0).any()):
        return "move"
    return "stop"


def _stratified_subset(dataset, limit: int, *, seed: int = 0,
                       ratios: dict[str, float] | None = None):
    """Sample whole chunks with action-stratum quotas and episode round-robin.

    This deliberately operates on dataset indices, so observation/action
    alignment remains untouched.  If a requested stratum is undersupplied,
    unused quota is redistributed from the remaining pool.
    """
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    ratios = dict(ratios or DEFAULT_STRATIFIED_RATIOS)
    if set(ratios) != set(ACTION_STRATA) or any(value < 0 for value in ratios.values()):
        raise ValueError("stratified ratios 必须覆盖四类且非负")
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("stratified ratios 总和必须为正")
    ratios = {key: value / total_ratio for key, value in ratios.items()}
    rng = random.Random(seed)
    grouped: dict[str, dict[str, list[int]]] = {
        key: defaultdict(list) for key in ACTION_STRATA
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        grouped[_action_stratum(sample)][sample["episode_id"]].append(index)
    for strata in grouped.values():
        for indices in strata.values():
            rng.shuffle(indices)

    raw_quota = {key: ratios[key] * limit for key in ACTION_STRATA}
    quota = {key: int(raw_quota[key]) for key in ACTION_STRATA}
    for key in sorted(ACTION_STRATA, key=lambda item: raw_quota[item] - quota[item], reverse=True)[:limit - sum(quota.values())]:
        quota[key] += 1
    selected: list[int] = []
    leftovers = limit
    for key in ACTION_STRATA:
        episodes = list(grouped[key])
        rng.shuffle(episodes)
        take = min(quota[key], sum(len(values) for values in grouped[key].values()))
        while take and episodes:
            progressed = False
            for episode in list(episodes):
                values = grouped[key][episode]
                if values:
                    selected.append(values.pop())
                    take -= 1
                    progressed = True
                    if not take:
                        break
                else:
                    episodes.remove(episode)
            if not progressed:
                break
        leftovers -= quota[key] - take

    if len(selected) < limit:
        used = set(selected)
        remaining = [index for index in range(len(dataset)) if index not in used]
        rng.shuffle(remaining)
        selected.extend(remaining[:limit - len(selected)])
    rng.shuffle(selected)
    return Subset(dataset, selected[:limit])


def encode_batch(adapter: torch.nn.Module, batch: dict[str, Any], *, device: torch.device,
                 frame_cache: dict[tuple[str, int, str], torch.Tensor] | None = None,
                 frame_stats: dict[str, int] | None = None) -> torch.Tensor:
    """Encode real images one frame at a time and return ``[B, L, D]`` features.

    ``frame_cache`` deduplicates vision-tower forwards across calls by task and
    frame path.  Cache entries are ``detach().cpu()`` snapshots, so a cache may only
    be passed on inference paths that run under ``torch.no_grad()`` with the
    adapter in eval mode; gradient training must keep it ``None`` (reusing a
    detached feature would silently cut gradients for the trainable
    projections).  Cached features fold instruction/mode conditioning
    (``encode_frame`` adds the condition projection), so the key is
    ``(instruction, mode_id, path)`` — path alone would wrongly reuse a
    feature across different tasks.  When ``frame_stats`` is given it must
    contain ``hits``/``encoded`` counters (pad slots are not counted); both
    are accumulated in place.
    """
    features = []
    cache_by_task: dict[tuple[str, int], Any] = {}
    # Per-batch cache deliberately keeps graph-connected tensors.  This is
    # safe for training and preserves gradient accumulation when duplicate
    # frame/task pairs occur within one batch; persistent ``frame_cache``
    # remains detached and is therefore restricted to inference.
    batch_frame_cache: dict[tuple[str, int, str], torch.Tensor] = {}
    for paths, instruction, mode_id in zip(
        batch["frame_paths"], batch["task_instruction"], batch["mode_id"].tolist()
    ):
        task_key = (instruction, int(mode_id))
        # Dataset stores the canonical mode id; model cache needs its token.
        mode = ("standard", "joint_hunt", "blackjack")[int(mode_id)]
        if task_key not in cache_by_task:
            cache_by_task[task_key] = adapter.encode_task_once(instruction, mode, task_id=f"{instruction}:{mode}")
        cache = cache_by_task[task_key]
        row = []
        for path, image in zip(paths, _load_images(paths)):
            cache_key = (instruction, int(mode_id),
                         str(path) if path is not None else "<pad>")
            if path is not None and cache_key in batch_frame_cache:
                row.append(batch_frame_cache[cache_key])
                if frame_stats is not None:
                    frame_stats["hits"] += 1
                continue
            if path is not None and frame_cache is not None and cache_key in frame_cache:
                row.append(frame_cache[cache_key].to(device=device))
                if frame_stats is not None:
                    frame_stats["hits"] += 1
                continue
            if image is None:
                row.append(torch.zeros(adapter.hidden_size, device=device, dtype=next(adapter.parameters()).dtype))
            else:
                feature = adapter.encode_frame(image, cache)
                if path is not None:
                    batch_frame_cache[cache_key] = feature
                if path is not None and frame_cache is not None:
                    frame_cache[cache_key] = feature.detach().cpu()
                if frame_stats is not None:
                    frame_stats["encoded"] += 1
                row.append(feature)
        features.append(torch.stack(row))
    return torch.stack(features).to(device)


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("mode_id", "intent_target", "subgoal_target", "subgoal_weight", "move_target",
            "camera_dx_target", "camera_dy_target", "button_target", "duration_target",
            "slow_loss_mask", "fast_loss_mask", "frame_valid_mask", "history_actions")
    return {key: batch[key].to(device) for key in keys}


def _scheduled_condition(core, slow, model_batch, *, teacher_forcing_ratio: float):
    """Mix ground-truth and predicted slow discrete conditions.

    The target labels are only used when valid and when the per-sample
    scheduled-sampling coin flip succeeds.  The continuous context is always
    predicted by the slow head, which keeps the fast loss differentiable with
    respect to the slow representation.
    """
    ratio = float(teacher_forcing_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("teacher_forcing_ratio 必须在 [0, 1]")
    pred_intent = slow.intent_id
    pred_subgoal = slow.subgoal_id
    intent_target = model_batch["intent_target"].to(pred_intent.device)
    subgoal_target = model_batch["subgoal_target"].to(pred_subgoal.device)
    valid = (intent_target >= 0) & (subgoal_target >= 0)
    if ratio <= 0.0:
        use_teacher = torch.zeros_like(valid)
    elif ratio >= 1.0:
        use_teacher = valid
    else:
        use_teacher = valid & (torch.rand(valid.shape, device=valid.device) < ratio)
    intent_id = torch.where(use_teacher, intent_target, pred_intent)
    subgoal_id = torch.where(use_teacher, subgoal_target, pred_subgoal)
    return core.condition_from_slow_output(
        slow, model_batch["mode_id"], intent_id=intent_id, subgoal_id=subgoal_id,
    )


def _interact_stage_counts(button_logits: torch.Tensor, button_target: torch.Tensor,
                           intent_target: torch.Tensor, subgoal_target: torch.Tensor) -> dict[str, dict[str, int]]:
    """Summarize interact predictions by the sample's intent/subgoal stage."""
    predicted = (button_logits[..., 0] > 0).sum(dim=-1).detach().cpu()
    targets = (button_target[..., 0] > 0).sum(dim=-1).detach().cpu()
    intents = intent_target.detach().cpu().reshape(-1)
    subgoals = subgoal_target.detach().cpu().reshape(-1)
    out: dict[str, dict[str, int]] = {}
    for pred, target, intent, subgoal in zip(predicted.tolist(), targets.tolist(), intents.tolist(), subgoals.tolist()):
        if intent < 0 or intent >= len(INTENTS):
            continue
        subgoal_name = SUBGOAL_NAMES[subgoal] if 0 <= subgoal < len(SUBGOAL_NAMES) else "unknown"
        out[f"{INTENTS[intent]}/{subgoal_name}"] = {
            "pred": out.get(f"{INTENTS[intent]}/{subgoal_name}", {}).get("pred", 0) + int(pred),
            "target": out.get(f"{INTENTS[intent]}/{subgoal_name}", {}).get("target", 0) + int(target),
        }
    return out


def forward_loss(adapter, core, batch, *, device, amp_enabled: bool, loss_weights=None,
                 teacher_forcing_ratio: float = 0.0,
                 frame_cache: dict | None = None, frame_stats: dict[str, int] | None = None):
    """Forward + loss for one batch.

    ``frame_cache``/``frame_stats`` are passed through to ``encode_batch``.
    The gradient training loop must leave them ``None``: cache entries are
    detached CPU snapshots, and reusing them across steps would silently
    disconnect ``visual_projection``/``condition_projection`` from the loss.
    Only ``eval()``+``torch.no_grad()`` evaluation paths (``_evaluate``,
    ``_slow_accuracy``) may pass a cache.
    """
    model_batch = _model_inputs(batch, device)
    frame_features = encode_batch(adapter, batch, device=device,
                                  frame_cache=frame_cache, frame_stats=frame_stats)
    condition = core.initial_condition(frame_features.shape[0], device=device,
                                       mode_id=0)
    # Keep the per-record mode conditioning from v4 (normally all rows in the
    # current session are ``standard``).  ``initial_condition`` accepts one
    # mode for convenience, so replace it with the batched ids here.
    condition.mode_id = model_batch["mode_id"]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        # First pass updates the slow state from the current visual window.
        slow_pass = core(frame_features, condition,
                         valid_mask=model_batch["frame_valid_mask"],
                         history_actions=model_batch["history_actions"], run_slow=True)
        if slow_pass.slow is None:  # defensive; run_slow=True above is required
            raise RuntimeError("slow pass 未产生 SlowVLAOutput")
        # Second pass is the actual fast decision.  During training the
        # discrete condition follows scheduled sampling; validation/inference
        # uses ratio=0 and therefore consumes the slow prediction.
        next_condition = _scheduled_condition(
            core, slow_pass.slow, model_batch,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        fast_pass = core(frame_features, next_condition,
                         valid_mask=model_batch["frame_valid_mask"],
                         history_actions=model_batch["history_actions"], run_slow=False,
                         detach_slow_condition=False)
        output = FastSlowVLAOutput(
            temporal_feature=fast_pass.temporal_feature,
            fast=fast_pass.fast,
            slow=slow_pass.slow,
        )
        losses = compute_vla_loss(output.fast, output.slow, model_batch,
                                  weights=loss_weights) if loss_weights is not None else compute_vla_loss(output.fast, output.slow, model_batch)
    return losses, output


def _mask_comparison(losses, adapter, core, batch, device, amp_enabled):
    zero_batch = dict(batch)
    zero_batch["slow_loss_mask"] = torch.zeros_like(batch["slow_loss_mask"])
    one_batch = dict(batch)
    one_batch["slow_loss_mask"] = torch.ones_like(batch["slow_loss_mask"])
    z, _ = forward_loss(adapter, core, zero_batch, device=device, amp_enabled=amp_enabled)
    o, _ = forward_loss(adapter, core, one_batch, device=device, amp_enabled=amp_enabled)
    return {"mask0_slow": float((z["slow_intent"] + z["slow_subgoal"]).detach().cpu()),
            "mask1_slow": float((o["slow_intent"] + o["slow_subgoal"]).detach().cpu())}


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(getattr(args, "seed", 0)))
    device = torch.device(args.device)
    dtype, amp_enabled = _device_dtype(device)
    if not args.init_checkpoint and not args.allow_base_init:
        raise ValueError("ACT 训练必须提供 --init-checkpoint M2_VG；仅兼容测试可加 --allow-base-init")
    dataset = VLASequenceDataset(_dataset_paths(args.data), verify_images=True)
    sampling = getattr(args, "sampling", "uniform")
    if sampling not in ("uniform", "stratified"):
        raise ValueError("sampling 必须是 uniform 或 stratified")
    if args.max_samples:
        dataset = (_stratified_subset(dataset, args.max_samples, seed=int(getattr(args, "seed", 0)))
                   if sampling == "stratified" else _bounded_subset(dataset, args.max_samples))
    # Do not silently cap the requested batch.  The caller is responsible for
    # selecting a size that fits the available GPU memory.
    batch_size = max(1, int(args.batch_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=VLASequenceCollator(max_frames=8))
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=VLASequenceCollator(max_frames=8))
    val_paths = _dataset_paths(getattr(args, "val_data", None))
    val_dataset = VLASequenceDataset(val_paths, verify_images=True) if val_paths else None
    max_val_samples = int(getattr(args, "max_val_samples", 64) or 0)
    if val_dataset is not None and max_val_samples > 0:
        val_dataset = _contiguous_subset(val_dataset, max_val_samples)
    val_loader = (DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=VLASequenceCollator(max_frames=8))
                  if val_dataset is not None else None)

    if args.init_checkpoint:
        adapter, _, parent_manifest = _load_act_backbone(
            args.model_path, args.init_checkpoint, dtype=dtype, device=device)
    elif args.allow_base_init:
        adapter, _ = load_qwen3vl_backbone(args.model_path, dtype=dtype, device_map=args.device,
                                           apply_lora=False)
        adapter.to(device=device)
        parent_manifest = None
    else:
        raise ValueError("ACT 训练必须提供 --init-checkpoint M2_VG；仅兼容测试可加 --allow-base-init")
    _freeze_qwen(adapter)
    # Keep newly-trained layers in FP32 (GradScaler cannot unscale FP16
    # gradients); autocast still executes their matmuls in FP16 on CUDA.
    for name in ("visual_projection", "condition_projection"):
        getattr(adapter, name).to(device=device, dtype=torch.float32)
    core = SharedFastSlowVLA(adapter.hidden_size, temporal_dim=args.temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    from idv_agent.training.vla_loss import VLALossWeights
    global_counts = None
    button_global_counts = None
    if args.move_direction_balance:
        global_counts = torch.zeros(9, dtype=torch.float32)
        for index in range(len(dataset)):
            global_counts += torch.bincount(dataset[index]["move_target"], minlength=9).to(torch.float32)
    if getattr(args, "button_global_balance", False):
        button_global_counts = torch.zeros((6, 2), dtype=torch.float32)
        for index in range(len(dataset)):
            target = dataset[index]["button_target"]
            button_global_counts[:, 1] += target.sum(dim=0)
            button_global_counts[:, 0] += (target.shape[0] - target.sum(dim=0))
    loss_weights = VLALossWeights(move_stop_weight=args.move_stop_weight,
                                   button_positive_weight=args.button_positive_weight,
                                   move_direction_balance=args.move_direction_balance,
                                   move_global_counts=global_counts,
                                   button_global_counts=button_global_counts)

    # Materialize LazyLinear before constructing the optimizer/checkpoint.
    first_batch = next(iter(loader))
    if getattr(args, "overfit_slow_mask", False):
        # Small-set memorization check: supervise the slow head on every
        # sampled chunk instead of simulating its low-frequency tick.
        first_batch["slow_loss_mask"] = torch.ones_like(first_batch["slow_loss_mask"])
    with torch.no_grad():
        encode_batch(adapter, first_batch, device=device)
    trainable = [p for p in list(adapter.parameters()) + list(core.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    baseline_losses, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=amp_enabled,
                                      loss_weights=loss_weights,
                                      teacher_forcing_ratio=_teacher_forcing_ratio(args, 0))
    baseline = float(baseline_losses["total"].detach().cpu())
    history: list[float] = []
    component_history: list[dict[str, float]] = []
    peak_memory = 0
    started = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if getattr(args, "overfit_slow_mask", False):
            batch["slow_loss_mask"] = torch.ones_like(batch["slow_loss_mask"])
        optimizer.zero_grad(set_to_none=True)
        losses, output = forward_loss(adapter, core, batch, device=device, amp_enabled=amp_enabled,
                                      loss_weights=loss_weights,
                                      teacher_forcing_ratio=_teacher_forcing_ratio(args, step - 1))
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"step {step}: loss 非有限值")
        if amp_enabled:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        history.append(float(total.detach().cpu()))
        component_history.append({key: float(value.detach().cpu()) for key, value in losses.items()})
        log_interval = max(1, int(getattr(args, "log_interval", 25)))
        if step == 1 or step % log_interval == 0 or step == args.steps:
            with torch.no_grad():
                move_pred = output.fast.move_logits.argmax(-1)
                move_target = batch["move_target"].to(output.fast.move_logits.device)
                nonstop = move_target != 0
                move_hit = ((move_pred != 0) & nonstop).sum().item() / max(1, nonstop.sum().item())
                intent_pred = int(output.slow.intent_logits.argmax(-1)[0].item()) if output.slow is not None else -1
                interact_pred = int((output.fast.button_logits[..., 0] > 0).sum().item())
                interact_target = int((batch["button_target"][..., 0].to(output.fast.button_logits.device) > 0).sum().item())
                interact_stages = _interact_stage_counts(
                    output.fast.button_logits,
                    batch["button_target"].to(output.fast.button_logits.device),
                    batch["intent_target"].to(output.fast.button_logits.device),
                    batch["subgoal_target"].to(output.fast.button_logits.device),
                )
            print(
                f"[step {step}/{args.steps}] total={float(total.detach().cpu()):.4f} "
                f"move={float(losses['fast_move'].detach().cpu()):.4f} "
                f"slow_intent={float(losses['slow_intent'].detach().cpu()):.4f} "
                f"slow_subgoal={float(losses['slow_subgoal'].detach().cpu()):.4f} "
                f"interact={interact_pred}/{interact_target} "
                f"move_nonstop_hit={move_hit:.3f} intent_pred={intent_pred} "
                f"tf={_teacher_forcing_ratio(args, step - 1):.3f}", flush=True,
            )
            if step == 1 or step % (log_interval * 4) == 0 or step == args.steps:
                print(f"[interact_stage step={step}] {json.dumps(interact_stages, ensure_ascii=False)}", flush=True)
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

    mask = _mask_comparison(history, adapter, core, first_batch, device, amp_enabled)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint_schema_version": "m3_act.internal.v1",
                "adapter": {"visual_projection": adapter.visual_projection.state_dict(),
                             "condition_projection": adapter.condition_projection.state_dict()},
                "core": core.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": args.steps, "baseline_loss": baseline}, checkpoint)

    # Checkpoint reproducibility: restore into the same modules and compare outputs.
    before, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    adapter.visual_projection.load_state_dict(saved["adapter"]["visual_projection"])
    adapter.condition_projection.load_state_dict(saved["adapter"]["condition_projection"])
    core.load_state_dict(saved["core"])
    after, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=False)
    elapsed = time.perf_counter() - started
    manifest = build_manifest(
        stage="M3_ACT", parent=str(Path(args.init_checkpoint).resolve()) if args.init_checkpoint else None,
        base_model=args.model_path, adapters=["lora_wk", "visual_projection", "act_heads"],
        frozen=["qwen_vision_tower", "lora_wk"],
        trainable=["visual_projection", "condition_projection", "shared_temporal_encoder", "slow_head", "fast_head"],
        data={"act": str(Path(args.data)).replace("\\", "/"), "act_glob": "vla.action_chunk.v4"},
        counts={"train": len(dataset), "val": len(val_dataset) if val_dataset is not None else 0},
        precision="fp16+GradScaler" if amp_enabled else "fp32",
        training={"method": "action_chunk_sft", "steps": args.steps, "learning_rate": args.lr,
                  "batch_size": batch_size, "temporal_dim": args.temporal_dim,
                  "max_samples": args.max_samples, "sampling": sampling,
                  "seed": int(getattr(args, "seed", 0)), "move_stop_weight": args.move_stop_weight,
                  "button_positive_weight": args.button_positive_weight,
                  "move_direction_balance": args.move_direction_balance,
                  "global_move_counts": global_counts.tolist() if global_counts is not None else None,
                  "global_button_counts": button_global_counts.tolist() if button_global_counts is not None else None,
                  "teacher_forcing_start": args.teacher_forcing_start,
                  "teacher_forcing_end": args.teacher_forcing_end,
                  "teacher_forcing_decay_steps": args.teacher_forcing_decay_steps},
        schema_versions=["vla.action_chunk.v4"],
        artifacts={"act_checkpoint": str(checkpoint.name)},
        extra={"parent_stage": parent_manifest["stage"] if parent_manifest else None,
               "val_data": [str(Path(path)).replace("\\", "/") for path in val_paths]},
    )
    manifest_path = checkpoint.parent / "manifest.json"
    write_manifest(manifest_path, manifest)
    # Embed the exact manifest snapshot in the binary as well as writing the
    # human-readable sidecar.  This prevents later experiments in the same
    # directory from making a checkpoint's provenance ambiguous.
    torch.save({**saved, "manifest": manifest}, checkpoint)
    result = {"samples": len(dataset), "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "steps": args.steps, "baseline_loss": baseline,
            "final_loss": history[-1] if history else baseline,
            "loss_history": history, "component_loss_history": component_history, "mask": mask,
            "checkpoint": str(checkpoint), "checkpoint_loss_delta": abs(float(before["total"] - after["total"])),
            "peak_memory_mb": peak_memory / (1024 * 1024), "elapsed_sec": elapsed,
            "manifest": str(manifest_path.resolve()),
            "checkpoint_manifest_embedded": True,
            "slow_accuracy": _slow_accuracy(adapter, core, eval_loader, device, amp_enabled),
            "val": (_evaluate(adapter, core, val_loader, device, amp_enabled)
                    if val_loader is not None else None)}
    (checkpoint.parent / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _slow_accuracy(adapter, core, loader, device, amp_enabled, *, max_samples: int | None = None,
                   frame_cache: dict | None = None, frame_stats: dict[str, int] | None = None):
    """Compute slow intent accuracy over every sample in the training subset.

    Runs under ``eval()`` + ``torch.no_grad()``, so a frame cache is safe;
    pass one in to share hits across multiple passes over the same loader.
    """
    correct = 0
    total = 0
    core.eval()
    adapter.eval()
    cache = {} if frame_cache is None else frame_cache
    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and total >= max_samples:
                break
            _, out = forward_loss(adapter, core, batch, device=device, amp_enabled=amp_enabled,
                                  frame_cache=cache, frame_stats=frame_stats)
            target = batch["intent_target"]
            valid = target != -100
            if bool(valid.any()):
                pred = out.slow.intent_logits.argmax(-1).cpu()
                correct += int((pred[valid] == target[valid]).sum())
                total += int(valid.sum())
    return (float(correct / total) if total else None)


def _evaluate(adapter, core, loader, device, amp_enabled):
    """Evaluate a bounded validation loader without updating model parameters."""
    if loader is None:
        return None
    core.eval()
    adapter.eval()
    losses: list[float] = []
    move_pred = [0] * 9
    move_target = [0] * 9
    intent_pred = [0] * 8
    intent_target = [0] * 8
    button_pred = [0] * 6
    button_target = [0] * 6
    non_stop_pred = 0
    non_stop_total = 0
    # Eval runs adapter.eval() + torch.no_grad(): safe to deduplicate vision
    # forwards.  The cache is shared with the slow-accuracy pass below so the
    # second traversal hits whatever the first one already encoded.
    frame_cache: dict = {}
    frame_stats = {"hits": 0, "encoded": 0}
    with torch.no_grad():
        for batch in loader:
            values, output = forward_loss(adapter, core, batch, device=device, amp_enabled=amp_enabled,
                                          frame_cache=frame_cache, frame_stats=frame_stats)
            losses.append(float(values["total"].detach().cpu()))
            mask = batch["fast_loss_mask"].to(output.fast.move_logits.device).bool()
            for value in output.fast.move_logits.argmax(-1)[mask].detach().cpu().reshape(-1).tolist():
                move_pred[int(value)] += 1
            targets = batch["move_target"][mask].detach().cpu().reshape(-1).tolist()
            for value in targets:
                move_target[int(value)] += 1
            target_tensor = batch["move_target"][mask]
            pred_tensor = output.fast.move_logits.argmax(-1)[mask]
            non_stop = target_tensor != 0
            non_stop_total += int(non_stop.sum())
            non_stop_pred += int(((pred_tensor != 0) & non_stop).sum())
            button_logits = output.fast.button_logits[mask]
            button_targets = batch["button_target"][mask]
            button_pred_tensor = (button_logits > 0).to(torch.int64)
            for i in range(button_pred_tensor.shape[-1]):
                button_pred[i] += int(button_pred_tensor[..., i].sum())
                button_target[i] += int((button_targets[..., i] > 0).sum())
            if output.slow is not None:
                valid = batch["intent_target"] >= 0
                for value in output.slow.intent_logits.argmax(-1)[valid].detach().cpu().tolist():
                    intent_pred[int(value)] += 1
                for value in batch["intent_target"][valid].detach().cpu().tolist():
                    intent_target[int(value)] += 1
    # Slow-accuracy pass reuses the same cache (and stats) so its forward
    # re-encodes nothing the loss pass above already encoded; the reported
    # hits therefore cover BOTH traversals of this bounded loader (loss pass +
    # slow-accuracy pass).  Without the cache those two passes would encode
    # roughly twice the unique frames, so hit_rate near 0.5 or higher means
    # the second traversal was almost fully served from cache.
    slow_acc = _slow_accuracy(adapter, core, loader, device, amp_enabled,
                              frame_cache=frame_cache, frame_stats=frame_stats)
    hits = frame_stats["hits"]
    total = frame_stats["encoded"] + hits
    return {"loss": (sum(losses) / len(losses) if losses else None),
            "slow_accuracy": slow_acc,
            "frame_cache": {
                "hits": hits, "misses": frame_stats["encoded"],
                "encoded": frame_stats["encoded"],
                "unique_frames_encoded": len(frame_cache),
                "hit_rate": (hits / total if total else None),
            },
            "move_pred_counts": move_pred, "move_target_counts": move_target,
            "intent_pred_counts": intent_pred, "intent_target_counts": intent_target,
            "button_pred_positive_counts": button_pred, "button_target_positive_counts": button_target,
            "interact_pred_positive": button_pred[0], "interact_target_positive": button_target[0],
            "move_nonstop_prediction_rate": (non_stop_pred / non_stop_total if non_stop_total else None)}


def _teacher_forcing_ratio(args: argparse.Namespace, step: int) -> float:
    """Linear scheduled-sampling ratio for one-based training progress."""
    start = float(getattr(args, "teacher_forcing_start", 1.0))
    end = float(getattr(args, "teacher_forcing_end", 0.0))
    decay = int(getattr(args, "teacher_forcing_decay_steps", 0) or args.steps)
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0 or decay < 1:
        raise ValueError("teacher forcing 参数必须在 [0,1] 且 decay_steps>=1")
    progress = min(max(int(step), 0), decay) / decay
    return start + (end - start) * progress


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="vla_chunks_v4.jsonl")
    parser.add_argument("--val-data", default="", help="可选：逗号/分号分隔的 session v4 JSONL 留出集")
    parser.add_argument("--max-val-samples", type=int, default=64,
                        help="验证集最多编码的样本数，0 表示整集")
    parser.add_argument("--model-path", required=True, help="本地 Qwen3-VL checkpoint")
    parser.add_argument("--init-checkpoint", default="", help="M2_VG checkpoint 目录（ACT 必填）")
    parser.add_argument("--allow-base-init", action="store_true", help="仅兼容测试：允许从原始 Qwen 初始化")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--move-stop-weight", type=float, default=0.5)
    parser.add_argument("--move-direction-balance", action="store_true",
                        help="batch>=4 时按 batch 频率；batch<4 时按采样训练集全局频率做 inverse-sqrt balance")
    parser.add_argument("--button-positive-weight", type=float, default=4.0)
    parser.add_argument("--button-global-balance", action="store_true",
                        help="按训练子集每个 button 的全局正/负计数计算 BCE 权重")
    parser.add_argument("--teacher-forcing-start", type=float, default=1.0,
                        help="训练开始时使用真实 intent/subgoal 的比例")
    parser.add_argument("--teacher-forcing-end", type=float, default=0.0,
                        help="scheduled sampling 结束时使用真实标签的比例")
    parser.add_argument("--teacher-forcing-decay-steps", type=int, default=0,
                        help="teacher forcing 线性退火步数；0 表示使用总训练步数")
    parser.add_argument("--sampling", choices=("uniform", "stratified"), default="uniform",
                        help="训练 chunk 采样策略；stratified 按交互/移动/按键/停止分层")
    parser.add_argument("--overfit-slow-mask", action="store_true",
                        help="小数据过拟合验证时将每条样本 slow_loss_mask 置 1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=25,
                        help="训练过程分量日志间隔（步）")
    parser.add_argument("--checkpoint", default="checkpoints/vla_minimal.pt")
    args = parser.parse_args(argv)
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
