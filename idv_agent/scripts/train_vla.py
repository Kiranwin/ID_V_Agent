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

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter, load_qwen3vl_backbone
from idv_agent.training.checkpoint_manifest import build_manifest, load_manifest, sha256_file, write_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset
from idv_agent.training.vla_loss import compute_vla_loss


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
                 frame_cache: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
    """Encode real images one frame at a time and return ``[B, L, D]`` features."""
    features = []
    cache_by_task: dict[tuple[str, int], Any] = {}
    for paths, instruction, mode_id in zip(
        batch["frame_paths"], batch["task_instruction"], batch["mode_id"].tolist()
    ):
        key = (instruction, int(mode_id))
        # Dataset stores the canonical mode id; model cache needs its token.
        mode = ("standard", "joint_hunt", "blackjack")[int(mode_id)]
        if key not in cache_by_task:
            cache_by_task[key] = adapter.encode_task_once(instruction, mode, task_id=f"{instruction}:{mode}")
        cache = cache_by_task[key]
        row = []
        for path, image in zip(paths, _load_images(paths)):
            cache_key = str(path) if path is not None else "<pad>"
            if frame_cache is not None and cache_key in frame_cache:
                row.append(frame_cache[cache_key].to(device=device))
                continue
            if image is None:
                row.append(torch.zeros(adapter.hidden_size, device=device, dtype=next(adapter.parameters()).dtype))
            else:
                feature = adapter.encode_frame(image, cache)
                if frame_cache is not None:
                    frame_cache[cache_key] = feature.detach().cpu()
                row.append(feature)
        features.append(torch.stack(row))
    return torch.stack(features).to(device)


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("mode_id", "intent_target", "subgoal_target", "subgoal_weight", "move_target",
            "camera_dx_target", "camera_dy_target", "button_target", "duration_target",
            "slow_loss_mask", "fast_loss_mask", "frame_valid_mask", "time_deltas")
    return {key: batch[key].to(device) for key in keys}


def forward_loss(adapter, core, batch, *, device, amp_enabled: bool, loss_weights=None):
    model_batch = _model_inputs(batch, device)
    frame_features = encode_batch(adapter, batch, device=device)
    condition = core.initial_condition(frame_features.shape[0], device=device,
                                       mode_id=0)
    # Keep the per-record mode conditioning from v4 (normally all rows in the
    # current session are ``standard``).  ``initial_condition`` accepts one
    # mode for convenience, so replace it with the batched ids here.
    condition.mode_id = model_batch["mode_id"]
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        output = core(frame_features, condition,
                      valid_mask=model_batch["frame_valid_mask"],
                      time_deltas=model_batch["time_deltas"], run_slow=True)
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
    batch_size = max(1, min(args.batch_size, 2))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=VLASequenceCollator(max_frames=8))
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=VLASequenceCollator(max_frames=8))
    val_paths = _dataset_paths(getattr(args, "val_data", None))
    val_dataset = VLASequenceDataset(val_paths, verify_images=True) if val_paths else None
    max_val_samples = int(getattr(args, "max_val_samples", 64) or 0)
    if val_dataset is not None and max_val_samples > 0:
        val_dataset = _bounded_subset(val_dataset, max_val_samples)
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
    core = SharedFastSlowVLA(adapter.hidden_size, temporal_dim=args.temporal_dim).to(device=device, dtype=torch.float32)
    from idv_agent.training.vla_loss import VLALossWeights
    loss_weights = VLALossWeights(move_stop_weight=args.move_stop_weight,
                                   button_positive_weight=args.button_positive_weight,
                                   move_direction_balance=args.move_direction_balance)

    # Materialize LazyLinear before constructing the optimizer/checkpoint.
    first_batch = next(iter(loader))
    with torch.no_grad():
        encode_batch(adapter, first_batch, device=device)
    trainable = [p for p in list(adapter.parameters()) + list(core.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    baseline_losses, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=amp_enabled,
                                      loss_weights=loss_weights)
    baseline = float(baseline_losses["total"].detach().cpu())
    history: list[float] = []
    peak_memory = 0
    started = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        losses, output = forward_loss(adapter, core, batch, device=device, amp_enabled=amp_enabled,
                                      loss_weights=loss_weights)
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
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

    mask = _mask_comparison(history, adapter, core, first_batch, device, amp_enabled)
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": {"visual_projection": adapter.visual_projection.state_dict(),
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
                  "batch_size": args.batch_size, "temporal_dim": args.temporal_dim,
                  "max_samples": args.max_samples, "sampling": sampling,
                  "seed": int(getattr(args, "seed", 0)), "move_stop_weight": args.move_stop_weight,
                  "button_positive_weight": args.button_positive_weight,
                  "move_direction_balance": args.move_direction_balance},
        schema_versions=["vla.action_chunk.v4"],
        artifacts={"act_checkpoint": str(checkpoint.name)},
        extra={"parent_stage": parent_manifest["stage"] if parent_manifest else None,
               "val_data": [str(Path(path)).replace("\\", "/") for path in val_paths]},
    )
    manifest_path = checkpoint.parent / "manifest.json"
    write_manifest(manifest_path, manifest)
    result = {"samples": len(dataset), "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "steps": args.steps, "baseline_loss": baseline,
            "final_loss": history[-1] if history else baseline,
            "loss_history": history, "mask": mask,
            "checkpoint": str(checkpoint), "checkpoint_loss_delta": abs(float(before["total"] - after["total"])),
            "peak_memory_mb": peak_memory / (1024 * 1024), "elapsed_sec": elapsed,
            "manifest": str(manifest_path.resolve()),
            "slow_accuracy": _slow_accuracy(adapter, core, eval_loader, device, amp_enabled),
            "val": (_evaluate(adapter, core, val_loader, device, amp_enabled)
                    if val_loader is not None else None)}
    (checkpoint.parent / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def _slow_accuracy(adapter, core, loader, device, amp_enabled, *, max_samples: int | None = None):
    """Compute slow intent accuracy over every sample in the training subset."""
    correct = 0
    total = 0
    core.eval()
    adapter.eval()
    with torch.no_grad():
        for batch in loader:
            if max_samples is not None and total >= max_samples:
                break
            _, out = forward_loss(adapter, core, batch, device=device, amp_enabled=amp_enabled)
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
    with torch.no_grad():
        for batch in loader:
            values, _ = forward_loss(adapter, core, batch, device=device, amp_enabled=amp_enabled)
            losses.append(float(values["total"].detach().cpu()))
    return {"loss": (sum(losses) / len(losses) if losses else None),
            "slow_accuracy": _slow_accuracy(adapter, core, loader, device, amp_enabled)}


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
    parser.add_argument("--move-stop-weight", type=float, default=0.25)
    parser.add_argument("--move-direction-balance", action="store_true",
                        help="按当前 batch 的九方向频率做 inverse-sqrt class balance")
    parser.add_argument("--button-positive-weight", type=float, default=4.0)
    parser.add_argument("--sampling", choices=("uniform", "stratified"), default="uniform",
                        help="训练 chunk 采样策略；stratified 按交互/移动/按键/停止分层")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default="checkpoints/vla_minimal.pt")
    args = parser.parse_args(argv)
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
