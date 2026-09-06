"""Minimal end-to-end VLA v4 training loop.

The Qwen visual-language backbone is frozen.  Only the adapter projections,
FiLM conditioner, temporal GRU and fast/slow heads are optimized.  The CLI is
intentionally small so it can be used for a 16--32 sample over-fit check or a
short 20--100 step smoke run on the real session data.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset

from idv_agent.model.fast_slow_vla import FastSlowVLAOutput, SharedFastSlowVLA
from idv_agent.model.act_checkpoint import (
    ACT_CHECKPOINT_SCHEMA, act_adapter_state, load_visual_grounded_act_checkpoint,
)
from idv_agent.model.qwen_backbone_adapter import load_qwen3vl_backbone
from idv_agent.training.checkpoint_manifest import build_manifest, write_manifest
from idv_agent.training.vla_dataset import VLASequenceCollator, VLASequenceDataset
from idv_agent.training.image_augmentation import (
    apply_window_augmentation as _augment_image,
    sample_window_augmentation as _sample_image_augmentation_params,
)
from idv_agent.training.feature_activity import training_behavior_gate
from idv_agent.training.vla_loss import VLALossWeights, balanced_class_weights, compute_vla_loss
from idv_agent.training.utils import set_seed
from idv_agent.configs.subgoal import SUBGOAL_NAMES
from idv_agent.vla.action_chunk import ACTION_CHUNK_HORIZON, INTENTS, VLA_SCHEMA_VERSION_V5


def _freeze_qwen(adapter: torch.nn.Module) -> None:
    """Freeze the language/vision backbone while keeping adapter modules trainable."""
    model = getattr(adapter, "model", None)
    if model is not None:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def _device_dtype(device: torch.device) -> tuple[torch.dtype, bool]:
    use_amp = device.type == "cuda"
    return (torch.float16 if use_amp else torch.float32), use_amp


def _make_grad_scaler(*, amp_enabled: bool, initial_scale: float = 1024.0) -> torch.amp.GradScaler:
    """Construct FP16 loss scaling conservatively for the frozen VLA stack.

    The PyTorch default (65536) overflowed the first scaled backward pass on
    an RTX 2080 Ti even though the unscaled loss was finite.  1024 has been
    verified on the same batch to keep every trainable gradient finite; the
    scaler may still grow it automatically when later steps permit.
    """
    if not math.isfinite(initial_scale) or initial_scale < 1.0:
        raise ValueError("grad initial scale 必须是 >= 1 的有限数")
    return torch.amp.GradScaler("cuda", enabled=amp_enabled, init_scale=float(initial_scale))


def _load_act_base_backbone(model_path: str | Path, *, dtype: torch.dtype,
                            device: torch.device | str):
    """Load the ACT base backbone without the known-bad M2 projection/LoRA."""
    adapter, processor = load_qwen3vl_backbone(
        model_path, dtype=dtype, device_map=None, apply_lora=False)
    return adapter.to(device), processor


def _load_images(paths: list[Path | None], *, augmentation_params=None,
                 image_transform: Callable[[Image.Image], Image.Image] | None = None) -> list[Image.Image | None]:
    images: list[Image.Image | None] = []
    for path in paths:
        if path is None:
            images.append(None)
        else:
            with Image.open(path) as image:
                loaded = image.convert("RGB")
                if image_transform is not None:
                    loaded = image_transform(loaded)
                images.append(_augment_image(loaded, augmentation_params)
                              if augmentation_params is not None else loaded)
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


def _episode_diverse_subset(dataset, limit: int):
    """Select one middle record per episode before filling the requested cap.

    Small evaluation caps must not accidentally become a contiguous slice of
    the first session.  The first pass therefore covers episodes in their
    source order and uses each episode's middle record; the second pass adds
    remaining records in source order.  This preserves deterministic input
    while making ``max_samples`` meaningful for cross-session diagnostics.
    """
    if limit <= 0 or len(dataset) <= limit:
        return dataset
    episode_indices: dict[Any, list[int]] = {}
    for index in range(len(dataset)):
        episode = dataset[index].get("episode_id")
        episode_indices.setdefault(episode, []).append(index)
    representatives = [indices[len(indices) // 2] for indices in episode_indices.values()]
    selected = representatives[:limit]
    if len(selected) < limit:
        used = set(selected)
        selected.extend(index for index in range(len(dataset))
                       if index not in used)
        selected = selected[:limit]
    return Subset(dataset, selected)


def _evaluation_stratified_subset(dataset, limit: int):
    """Pick deterministic cross-episode examples that cover rare output labels.

    Image-dependency acceptance is invalid when its cap accidentally includes
    only the dominant tactical intent or neutral camera bucket.  This selector
    scores each not-yet-selected record by the inverse frequency of *uncovered*
    intent/move/camera labels, with a small bonus for a new episode.  It only
    reads supervision to choose a fixed evaluation cohort; labels never enter
    the model inputs.
    """
    if limit <= 0 or len(dataset) <= limit:
        return dataset

    def tags(sample: dict[str, Any]) -> set[tuple[str, int]]:
        result: set[tuple[str, int]] = set()
        intent = sample.get("intent_target")
        if intent is not None:
            intent_value = int(intent.reshape(-1)[0]) if isinstance(intent, torch.Tensor) else int(intent)
            if intent_value >= 0:
                result.add(("intent", intent_value))
        for name in ("move_target", "camera_dx_target", "camera_dy_target"):
            values = sample.get(name)
            if values is None:
                continue
            values = values.reshape(-1).tolist() if isinstance(values, torch.Tensor) else values
            result.update((name, int(value)) for value in values)
        return result

    rows = [dataset[index] for index in range(len(dataset))]
    tag_counts = Counter(tag for sample in rows for tag in tags(sample))
    selected: list[int] = []
    uncovered = set(tag_counts)
    seen_episodes: set[Any] = set()
    remaining = set(range(len(rows)))

    # Reserve a small but usable estimate for every intent that exists in the
    # held-out data.  A single rare-intent example can prove presence, but it
    # cannot support a meaningful normal-vs-degraded accuracy comparison.
    intent_indices: dict[int, list[int]] = defaultdict(list)
    for index, sample in enumerate(rows):
        intent_tags = [tag for tag in tags(sample) if tag[0] == "intent"]
        if intent_tags:
            intent_indices[intent_tags[0][1]].append(index)
    quota = 4
    if len(intent_indices) * quota <= limit:
        for intent in sorted(intent_indices):
            candidates = set(intent_indices[intent])
            while candidates and len([i for i in selected if i in intent_indices[intent]]) < quota:
                index = max(candidates, key=lambda candidate: (
                    sum(1.0 / tag_counts[tag] for tag in tags(rows[candidate]) & uncovered),
                    int(rows[candidate].get("episode_id") not in seen_episodes),
                    -candidate,
                ))
                candidates.remove(index)
                remaining.remove(index)
                selected.append(index)
                uncovered.difference_update(tags(rows[index]))
                seen_episodes.add(rows[index].get("episode_id"))

    while remaining and len(selected) < limit:
        def score(index: int) -> tuple[float, int, int]:
            sample_tags = tags(rows[index])
            rarity = sum(1.0 / tag_counts[tag] for tag in sample_tags & uncovered)
            new_episode = int(rows[index].get("episode_id") not in seen_episodes)
            # Lower source index makes ties deterministic.
            return rarity, new_episode, -index

        index = max(remaining, key=score)
        selected.append(index)
        remaining.remove(index)
        uncovered.difference_update(tags(rows[index]))
        seen_episodes.add(rows[index].get("episode_id"))
    return Subset(dataset, selected)


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
                 frame_stats: dict[str, int] | None = None,
                 vision_micro_batch_size: int = 8,
                 raw_feature_cache: Any | None = None,
                 augment_images: bool = False,
                 image_transform: Callable[[Image.Image], Image.Image] | None = None,
                 return_visual: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Encode real images in vision micro-batches and return ``[B, L, D]`` features.

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
    if vision_micro_batch_size < 1:
        raise ValueError("vision_micro_batch_size 必须为正数")
    if augment_images and frame_cache is not None:
        raise ValueError("augment_images 与 frame_cache 不兼容；持久特征缓存不能保存随机增强结果")
    if augment_images and raw_feature_cache is not None:
        raise ValueError("augment_images 与 raw_feature_cache 不兼容；增强必须在原始图像上执行")
    features = []
    visual_features = []
    cache_by_task: dict[tuple[str, int], Any] = {}
    # Per-batch cache deliberately keeps graph-connected tensors.  This is
    # safe for training and preserves gradient accumulation when duplicate
    # frame/task pairs occur within one batch; persistent ``frame_cache``
    # remains detached and is therefore restricted to inference.
    batch_frame_cache: dict[tuple[Any, ...], torch.Tensor] = {}
    pending: dict[tuple[str, int], list[tuple[int, int, Any, Any, tuple[Any, ...]]]] = defaultdict(list)
    pending_refs: dict[tuple[Any, ...], list[tuple[int, int]]] = defaultdict(list)
    for row_index, (paths, instruction, mode_id) in enumerate(zip(
        batch["frame_paths"], batch["task_instruction"], batch["mode_id"].tolist()
    )):
        task_key = (instruction, int(mode_id))
        # Dataset stores the canonical mode id; model cache needs its token.
        mode = ("standard", "joint_hunt", "blackjack")[int(mode_id)]
        if task_key not in cache_by_task:
            cache_by_task[task_key] = adapter.encode_task_once(instruction, mode, task_id=f"{instruction}:{mode}")
        cache = cache_by_task[task_key]
        augmentation_params = (_sample_image_augmentation_params() if augment_images else None)
        row = [None] * len(paths)
        visual_row = [None] * len(paths)
        images = []
        load_paths = []
        for path in paths:
            path_key = (instruction, int(mode_id), str(path) if path is not None else "<pad>")
            cache_key = path_key if not augment_images else (*path_key, row_index)
            if (path is None or cache_key in batch_frame_cache or
                    (frame_cache is not None and cache_key in frame_cache) or
                    (raw_feature_cache is not None and path in raw_feature_cache)):
                images.append(None)
            else:
                images.append("__LOAD__")
                load_paths.append(path)
        try:
            loaded_images = _load_images(load_paths, augmentation_params=augmentation_params,
                                         image_transform=image_transform)
        except TypeError as exc:
            # Keep lightweight third-party/test loaders that implement the
            # old one-argument hook working when augmentation is disabled.
            if augmentation_params is not None:
                raise
            if "augmentation_params" not in str(exc):
                raise
            loaded_images = _load_images(load_paths)
        loaded = iter(loaded_images)
        images = [next(loaded) if image == "__LOAD__" else image for image in images]
        for col_index, (path, image) in enumerate(zip(paths, images)):
            path_key = (instruction, int(mode_id),
                        str(path) if path is not None else "<pad>")
            cache_key = path_key if not augment_images else (*path_key, row_index)
            if path is not None and cache_key in batch_frame_cache:
                row[col_index] = batch_frame_cache[cache_key]
                if return_visual:
                    visual_row[col_index] = adapter.uncondition_features(
                        row[col_index].unsqueeze(0), cache)[0]
                if frame_stats is not None:
                    frame_stats["hits"] += 1
                continue
            if path is not None and frame_cache is not None and cache_key in frame_cache:
                row[col_index] = frame_cache[cache_key].to(device=device)
                if return_visual:
                    visual_row[col_index] = adapter.uncondition_features(
                        row[col_index].unsqueeze(0), cache)[0]
                if frame_stats is not None:
                    frame_stats["hits"] += 1
                continue
            if path is not None and raw_feature_cache is not None and path in raw_feature_cache:
                project = getattr(adapter, "project_raw_features", None)
                if project is None:
                    raise TypeError("raw_feature_cache 需要 adapter.project_raw_features")
                raw = raw_feature_cache.get(path).to(device=device)
                spatial_k = getattr(adapter, "spatial_k", None)
                if spatial_k is not None:
                    expected = f"spatial_grid_k{int(spatial_k)}_v1"
                    if getattr(raw_feature_cache, "index", {}).get("pooling") != expected:
                        raise ValueError(f"spatial ACT 需要 {expected} cache")
                    if raw.ndim != 2 or raw.shape[0] != int(spatial_k) ** 2:
                        raise ValueError("spatial ACT 只能加载 [k*k,D] grid cache")
                    raw = raw.unsqueeze(0)
                else:
                    raw = raw.reshape(1, -1)
                feature = project(raw, cache)[0]
                batch_frame_cache[cache_key] = feature
                row[col_index] = feature
                if return_visual:
                    visual_row[col_index] = adapter.project_raw_visual_features(raw)[0]
                if frame_stats is not None:
                    frame_stats["raw_cache_hits"] = frame_stats.get("raw_cache_hits", 0) + 1
                continue
            if image is None:
                row[col_index] = torch.zeros(getattr(adapter, "act_feature_dim",
                                                     getattr(adapter, "hidden_size")), device=device,
                                             dtype=next(adapter.parameters()).dtype)
                if return_visual:
                    visual_row[col_index] = torch.zeros_like(row[col_index])
            else:
                pending_refs[cache_key].append((row_index, col_index))
                if not any(entry[4] == cache_key for entry in pending[task_key]):
                    pending[task_key].append((row_index, col_index, path, image, cache_key))
        features.append(row)
        if return_visual:
            visual_features.append(visual_row)
    for task_key, entries in pending.items():
        cache = cache_by_task[task_key]
        for start in range(0, len(entries), vision_micro_batch_size):
            chunk = entries[start:start + vision_micro_batch_size]
            images = [entry[3] for entry in chunk]
            encode_frames = getattr(adapter, "encode_frames", None)
            encode_pair = getattr(adapter, "encode_frames_with_visual", None)
            if return_visual and encode_pair is not None:
                encoded, encoded_visual = encode_pair(
                    images, cache, micro_batch_size=vision_micro_batch_size)
            elif encode_frames is not None:
                encoded = encode_frames(images, cache, micro_batch_size=vision_micro_batch_size)
                encoded_visual = encoded
            else:
                encoded = torch.stack([adapter.encode_frame(image, cache) for image in images])
                encoded_visual = encoded
            if encoded.ndim != 2 or encoded.shape[0] != len(chunk):
                raise ValueError("encode_frames 必须返回 [N,D] Tensor")
            if encoded_visual.ndim != 2 or encoded_visual.shape != encoded.shape:
                raise ValueError("encode_frames_with_visual 必须返回两个同形状 [N,D] Tensor")
            for entry, feature, visual_feature in zip(chunk, encoded, encoded_visual):
                row_index, col_index, path, _image, cache_key = entry
                for row_index, col_index in pending_refs[cache_key]:
                    features[row_index][col_index] = feature
                    if return_visual:
                        visual_features[row_index][col_index] = visual_feature
                batch_frame_cache[cache_key] = feature
                if frame_cache is not None:
                    frame_cache[cache_key] = feature.detach().cpu()
                if frame_stats is not None:
                    frame_stats["encoded"] += 1
    features = [torch.stack(row) for row in features]
    conditioned = torch.stack(features).to(device)
    if not return_visual:
        return conditioned
    visual_features = [torch.stack(row) for row in visual_features]
    return conditioned, torch.stack(visual_features).to(device)


def _model_inputs(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = ("mode_id", "intent_target", "subgoal_target", "subgoal_weight", "move_target",
            "camera_dx_target", "camera_dy_target", "button_target", "duration_target",
            "slow_loss_mask", "fast_loss_mask", "frame_valid_mask", "history_actions",
            "grounding_mask", "grounding_present_target", "grounding_bbox_mask",
            "grounding_bbox_target", "grounding_side_target", "grounding_prompt_target",
            "grounding_reachable_target")
    return {key: batch[key].to(device) for key in keys}


def _drop_history_actions(history_actions: torch.Tensor, *, probability: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop complete action histories per sample during training only."""
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("history_dropout 必须在 [0,1]")
    if history_actions.ndim != 2:
        raise ValueError("history_actions 必须是 [B,H]")
    drop_mask = torch.rand(history_actions.shape[0], device=history_actions.device) < float(probability)
    return history_actions.masked_fill(drop_mask.unsqueeze(-1), 0), drop_mask


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
                 frame_cache: dict | None = None, frame_stats: dict[str, int] | None = None,
                 vision_micro_batch_size: int = 8,
                 raw_feature_cache: Any | None = None,
                 ablate_frame_avg: bool = False, no_history: bool = False,
                 augment_images: bool = False, history_dropout_p: float = 0.0):
    """Forward + loss for one batch.

    ``frame_cache``/``frame_stats`` are passed through to ``encode_batch``.
    The gradient training loop must leave them ``None``: cache entries are
    detached CPU snapshots, and reusing them across steps would silently
    disconnect the trainable visual/condition projections from the loss.
    Only ``eval()``+``torch.no_grad()`` evaluation paths (``_evaluate``,
    ``_slow_accuracy``) may pass a cache.
    """
    model_batch = _model_inputs(batch, device)
    if no_history:
        # 时序/历史剥离：把 fast 头消费的上一动作条件 history_actions 置零，
        # 使模型无法凭“上一步动作的自相关”去拟合，只留(可选的)画面与时序帧结构。
        model_batch["history_actions"] = torch.zeros_like(model_batch["history_actions"])
    elif history_dropout_p:
        model_batch["history_actions"], _ = _drop_history_actions(
            model_batch["history_actions"], probability=history_dropout_p)
    encoded = encode_batch(adapter, batch, device=device,
                                  frame_cache=frame_cache, frame_stats=frame_stats,
                                  vision_micro_batch_size=vision_micro_batch_size,
                                  raw_feature_cache=raw_feature_cache,
                                  augment_images=augment_images, return_visual=True)
    frame_features, visual_frame_features = encoded
    if ablate_frame_avg:
        # 画面边际贡献 ablation：把 (B,L,D) 里区分不同帧/不同画面(multi-session batch)
        # 的可分辨分量中和——把所有 **valid(真实)帧** 替换为其全局均值特征。
        # 跨画面的均值使模型看不到“这帧是 A 地图/这帧是 B 地图”的身份，只剩共同画面
        # 轮廓与时序结构；均值是可导(自动梯度按平均方向)，保留与原始同 shape 供
        # valid_mask/history 的时序语义不变，从而 loss 能直接反映是否依赖画面差异。
        vm = model_batch.get("frame_valid_mask", None)
        if vm is not None:
            valid_rows = vm.reshape(-1).bool()
            flat = frame_features.reshape(-1, frame_features.shape[-1])
            has = valid_rows.any()
            if has and valid_rows.all():
                mean_feat = flat.mean(dim=0, keepdim=True)
            elif has:
                mean_feat = flat[valid_rows].mean(dim=0, keepdim=True)
            else:  # 无 valid——退化为全均值以免 NaN(不应发生)
                mean_feat = flat.mean(dim=0, keepdim=True)
        else:
            mean_feat = frame_features.mean(dim=(0, 1), keepdim=True)
        frame_features = mean_feat.reshape(1, 1, -1).expand_as(frame_features).clone()
        visual_frame_features = frame_features.clone()
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
                         visual_frame_features=visual_frame_features,
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
                         visual_frame_features=visual_frame_features,
                         history_actions=model_batch["history_actions"], run_slow=False,
                         detach_slow_condition=False)
        output = FastSlowVLAOutput(
            temporal_feature=fast_pass.temporal_feature,
            fast=fast_pass.fast,
            slow=slow_pass.slow,
            visual=fast_pass.visual,
            prior_fast=fast_pass.prior_fast,
            prior_slow=slow_pass.prior_slow,
        )
        loss_kwargs = {
            "visual_fast": output.visual.fast,
            "visual_slow": slow_pass.visual.slow,
        }
        effective_weights = loss_weights
        if effective_weights is not None and getattr(core, "visual_only_deployment", False):
            # ``output.fast/.slow`` are already the visual expert outputs in
            # this ACT schema; applying visual_aux would double its gradient.
            effective_weights = replace(effective_weights, visual_aux=0.0)
        losses = (compute_vla_loss(output.fast, output.slow, model_batch,
                                   weights=effective_weights, grounding=output.visual.grounding, **loss_kwargs)
                  if effective_weights is not None else
                  compute_vla_loss(output.fast, output.slow, model_batch,
                                   grounding=output.visual.grounding, **loss_kwargs))
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


def _class_balance_statistics(dataset) -> dict[str, torch.Tensor]:
    """Count exactly the train-subset labels consumed by each weighted head."""
    counts = {
        "move": torch.zeros(9, dtype=torch.float32),
        "camera_dx": torch.zeros(5, dtype=torch.float32),
        "camera_dy": torch.zeros(5, dtype=torch.float32),
        "intent": torch.zeros(len(INTENTS), dtype=torch.float32),
    }
    for index in range(len(dataset)):
        sample = dataset[index]
        counts["move"] += torch.bincount(sample["move_target"].reshape(-1), minlength=9).to(torch.float32)
        counts["camera_dx"] += torch.bincount(sample["camera_dx_target"].reshape(-1), minlength=5).to(torch.float32)
        counts["camera_dy"] += torch.bincount(sample["camera_dy_target"].reshape(-1), minlength=5).to(torch.float32)
        if int(sample["slow_loss_mask"]) and int(sample["intent_target"]) >= 0:
            counts["intent"][int(sample["intent_target"])] += 1
    return counts


def _interact_event_bias_init(dataset, *, horizon: int = ACTION_CHUNK_HORIZON,
                              eps: float = 1e-4) -> torch.Tensor:
    """Initialize the independent Q-event head from chunk-level event rates."""
    if not 0.0 < eps < 0.5:
        raise ValueError("interact event bias eps 必须在 (0, 0.5) 内")
    positive = torch.zeros(horizon, dtype=torch.float64)
    total_chunks = 0
    for index in range(len(dataset)):
        target = dataset[index]["button_target"]
        if target.ndim != 2 or target.shape != (horizon, 6):
            raise ValueError(f"button_target 必须是 [{horizon},6] Tensor")
        positive += target[:, 0].to(torch.float64)
        total_chunks += 1
    if total_chunks < 1:
        raise ValueError("interact event bias 需要至少一个动作块")
    rate = (positive / total_chunks).clamp(eps, 1.0 - eps)
    return torch.logit(rate).to(torch.float32)


def _ordinary_button_bias_init(dataset, *, horizon: int = ACTION_CHUNK_HORIZON,
                               eps: float = 1e-4) -> torch.Tensor:
    """Initialize five held-button logits from train-subset rates."""
    if not 0.0 < eps < 0.5:
        raise ValueError("ordinary button bias eps 必须在 (0, 0.5) 内")
    positive = torch.zeros((horizon, 5), dtype=torch.float64)
    total_chunks = 0
    for index in range(len(dataset)):
        target = dataset[index]["button_target"]
        if target.ndim != 2 or target.shape != (horizon, 6):
            raise ValueError(f"button_target 必须是 [{horizon},6] Tensor")
        positive += target[:, 1:].to(torch.float64)
        total_chunks += 1
    if total_chunks < 1:
        raise ValueError("ordinary button bias 需要至少一个动作块")
    rate = (positive / total_chunks).clamp(eps, 1.0 - eps)
    return torch.logit(rate).to(torch.float32)


def _class_balance_manifest(statistics: dict[str, torch.Tensor], *, data_paths: list[str],
                            sample_count: int | None = None, sampling: str | None = None,
                            max_samples: int | None = None) -> dict[str, object]:
    """Make the exact balance table and its source reproducible in a checkpoint."""
    return {
        "method": "inverse_sqrt_clamped_mean1",
        "clamp": [0.35, 3.0],
        "normalized_per_head": True,
        "computed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"schema": VLA_SCHEMA_VERSION_V5,
                   "paths": [str(Path(path).resolve()) for path in data_paths],
                   "sample_count": sample_count,
                   "sampling": sampling,
                   "max_samples": max_samples},
        "heads": {
            name: {"counts": values.tolist(), "weights": balanced_class_weights(values).tolist()}
            for name, values in statistics.items()
        },
    }


def _gradient_diagnostics(named_parameters: list[tuple[str, torch.nn.Parameter]]) -> list[dict[str, object]]:
    """Return compact, serializable evidence for non-finite trainable grads.

    This intentionally runs after ``GradScaler.unscale_`` and before clipping:
    clipping an infinity by an infinite global norm turns it into NaN and
    destroys the evidence needed to distinguish a scaler overflow from an
    invalid forward/backward computation.
    """
    invalid: list[dict[str, object]] = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None or bool(torch.isfinite(gradient).all()):
            continue
        max_abs = gradient.detach().abs().max()
        invalid.append({
            "name": name,
            "dtype": str(gradient.dtype),
            "max_abs": (float(max_abs.cpu()) if bool(torch.isfinite(max_abs)) else None),
        })
    return invalid


def _named_act_trainable(adapter: torch.nn.Module, core: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    """Return ACT optimizer parameters while excluding the unused lazy legacy path."""
    adapter_params = [(f"adapter.{name}", parameter) for name, parameter in adapter.named_parameters()
                      if parameter.requires_grad and not name.startswith("visual_projection.")]
    core_params = [(f"core.{name}", parameter) for name, parameter in core.named_parameters()
                   if parameter.requires_grad]
    return adapter_params + core_params


def _calibrate_visual_input_center(adapter: torch.nn.Module, core: SharedFastSlowVLA, dataset,
                                   *, device: torch.device, samples: int,
                                   vision_micro_batch_size: int, seed: int,
                                   raw_feature_cache: Any | None = None) -> dict[str, object]:
    """Set the persistent raw-visual center from deterministic training examples.

    A frozen Qwen grid contains a large cross-image common direction.  The
    visual-only linear heads must see residual image evidence, not use that
    direction as a majority-class bias.  Calibration is separate from random
    augmentation and is checkpointed in ``visual_expert.input_center``.
    """
    samples = int(samples)
    if samples < 1:
        raise ValueError("visual_center_samples 必须为正数")
    subset = _stratified_subset(dataset, min(samples, len(dataset)), seed=int(seed))
    loader = DataLoader(subset, batch_size=1, shuffle=False,
                        collate_fn=VLASequenceCollator(max_frames=8))
    was_adapter_training, was_core_training = adapter.training, core.training
    adapter.eval(); core.eval()
    total: torch.Tensor | None = None
    count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                _, raw_visual = encode_batch(adapter, batch, device=device,
                                             vision_micro_batch_size=vision_micro_batch_size,
                                             raw_feature_cache=raw_feature_cache,
                                             return_visual=True)
                pairs = core.visual_expert.pair_features(
                    raw_visual, valid_mask=batch["frame_valid_mask"].to(device)).detach().double()
                current = pairs.sum(dim=0)
                total = current if total is None else total + current
                count += int(pairs.shape[0])
    finally:
        adapter.train(was_adapter_training); core.train(was_core_training)
    if total is None or count < 1:
        raise RuntimeError("visual center calibration 未产生有效样本")
    center = (total / count).float()
    core.visual_expert.set_input_center(center)
    return {"samples": count, "pair_dim": int(center.numel()),
            "abs_mean": float(center.abs().mean().cpu())}


def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(getattr(args, "seed", 0)))
    device = torch.device(args.device)
    dtype, amp_enabled = _device_dtype(device)
    if args.init_checkpoint:
        raise ValueError("M2_VG 初始化当前已禁用；请不要传 --init-checkpoint")
    grounding_annotations = getattr(args, "grounding_annotations", "") or None
    dataset = VLASequenceDataset(_dataset_paths(args.data), verify_images=True,
                                 grounding_annotations=grounding_annotations)
    sampling = getattr(args, "sampling", "uniform")
    if sampling not in ("uniform", "stratified"):
        raise ValueError("sampling 必须是 uniform 或 stratified")
    if args.max_samples:
        dataset = (_stratified_subset(dataset, args.max_samples, seed=int(getattr(args, "seed", 0)))
                   if sampling == "stratified" else _bounded_subset(dataset, args.max_samples))
    # Do not silently cap the requested batch.  The caller is responsible for
    # selecting a size that fits the available GPU memory.
    batch_size = max(1, int(args.batch_size))
    vision_micro_batch_size = max(1, int(getattr(args, "vision_micro_batch_size", 8)))
    _ablate_frame = bool(getattr(args, "ablate_frame_avg", False))
    _no_history = bool(getattr(args, "no_history", False))
    history_dropout_p = float(getattr(args, "history_dropout_p", 0.5))
    image_augmentation = bool(getattr(args, "image_augmentation", True))
    raw_feature_cache = None
    if getattr(args, "raw_feature_cache", ""):
        from idv_agent.training.raw_feature_cache import RawFeatureCache
        raw_feature_cache = RawFeatureCache(args.raw_feature_cache, spatial_k=8)
    if image_augmentation and raw_feature_cache is not None:
        raise ValueError("--image-augmentation 与 --raw-feature-cache 不兼容；请关闭其中之一")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=VLASequenceCollator(max_frames=8))
    eval_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=VLASequenceCollator(max_frames=8))
    val_paths = _dataset_paths(getattr(args, "val_data", None))
    val_dataset = (VLASequenceDataset(val_paths, verify_images=True,
                                      grounding_annotations=grounding_annotations)
                   if val_paths else None)
    max_val_samples = int(getattr(args, "max_val_samples", 64) or 0)
    if val_dataset is not None and max_val_samples > 0:
        val_dataset = _contiguous_subset(val_dataset, max_val_samples)
    val_loader = (DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=VLASequenceCollator(max_frames=8))
                  if val_dataset is not None else None)

    # M2_VG has known grounding defects; ACT currently starts from the base
    # backbone without loading its projection/adapter weights.
    adapter, _ = load_qwen3vl_backbone(args.model_path, dtype=dtype, device_map=args.device,
                                       apply_lora=False)
    adapter.to(device=device)
    parent_manifest = None
    _freeze_qwen(adapter)
    if raw_feature_cache is not None and getattr(adapter, "spatial_k", None) is not None:
        expected = f"spatial_grid_k{int(adapter.spatial_k)}_v1"
        if raw_feature_cache.index.get("pooling") != expected:
            raise ValueError(f"spatial ACT 需要 {expected} cache，收到 {raw_feature_cache.index.get('pooling')}")
    # Keep newly-trained layers in FP32 (GradScaler cannot unscale FP16
    # gradients); autocast still executes their matmuls in FP16 on CUDA.
    adapter.condition_projection.to(device=device, dtype=torch.float32)
    global_counts = None
    class_balance_statistics = None
    class_balance_manifest = None
    button_global_counts = None
    if args.move_direction_balance:
        global_counts = torch.zeros(9, dtype=torch.float32)
        for index in range(len(dataset)):
            global_counts += torch.bincount(dataset[index]["move_target"], minlength=9).to(torch.float32)
    if getattr(args, "class_balance", True):
        class_balance_statistics = _class_balance_statistics(dataset)
        class_balance_manifest = _class_balance_manifest(
            class_balance_statistics, data_paths=_dataset_paths(args.data),
            sample_count=len(dataset), sampling=sampling,
            max_samples=int(getattr(args, "max_samples", 0) or 0))
    if getattr(args, "button_global_balance", False):
        button_global_counts = torch.zeros((6, 2), dtype=torch.float32)
        for index in range(len(dataset)):
            target = dataset[index]["button_target"]
            button_global_counts[:, 1] += target.sum(dim=0)
            button_global_counts[:, 0] += (target.shape[0] - target.sum(dim=0))
    loss_weights = VLALossWeights(move_stop_weight=args.move_stop_weight,
                                   button_positive_weight=args.button_positive_weight,
                                   visual_aux=float(getattr(args, "visual_aux", 1.0)),
                                   move_direction_balance=args.move_direction_balance,
                                   move_global_counts=(class_balance_statistics["move"]
                                                        if class_balance_statistics else global_counts),
                                   class_balance=bool(getattr(args, "class_balance", True)),
                                   camera_dx_global_counts=(class_balance_statistics["camera_dx"]
                                                            if class_balance_statistics else None),
                                   camera_dy_global_counts=(class_balance_statistics["camera_dy"]
                                                            if class_balance_statistics else None),
                                   intent_global_counts=(class_balance_statistics["intent"]
                                                         if class_balance_statistics else None),
                                   button_global_counts=button_global_counts,
                                   grounding=float(getattr(args, "grounding_loss_weight", 1.0)))

    # Materialize the direct raw-grid feature contract before constructing the
    # core/optimizer.  ``hidden_size`` is Qwen text width (2560), not ACT
    # visual width (2*2*1024 = 4096).
    first_batch = next(iter(loader))
    if getattr(args, "overfit_slow_mask", False):
        # Small-set memorization check: supervise the slow head on every
        # sampled chunk instead of simulating its low-frequency tick.
        first_batch["slow_loss_mask"] = torch.ones_like(first_batch["slow_loss_mask"])
    with torch.no_grad():
        encode_batch(adapter, first_batch, device=device, raw_feature_cache=raw_feature_cache)
    if int(getattr(adapter, "act_feature_dim", 0)) < 1:
        raise RuntimeError("ACT raw-grid feature dim 未 materialize")
    core = SharedFastSlowVLA(adapter.act_feature_dim, temporal_dim=args.temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    interact_event_bias_init = _interact_event_bias_init(dataset, horizon=ACTION_CHUNK_HORIZON)
    core.visual_expert.set_interact_event_bias(interact_event_bias_init)
    ordinary_button_bias_init = _ordinary_button_bias_init(dataset, horizon=ACTION_CHUNK_HORIZON)
    core.visual_expert.set_ordinary_button_bias(ordinary_button_bias_init)
    visual_center = _calibrate_visual_input_center(
        adapter, core, dataset, device=device,
        samples=int(getattr(args, "visual_center_samples", 64)),
        vision_micro_batch_size=vision_micro_batch_size,
        seed=int(getattr(args, "seed", 0)),
        raw_feature_cache=raw_feature_cache,
    )
    named_trainable = _named_act_trainable(adapter, core)
    trainable = [parameter for _name, parameter in named_trainable]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    grad_initial_scale = float(getattr(args, "grad_initial_scale", 1024.0))
    scaler = _make_grad_scaler(amp_enabled=amp_enabled, initial_scale=grad_initial_scale)

    baseline_losses, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=amp_enabled,
                                      vision_micro_batch_size=vision_micro_batch_size,
                                      raw_feature_cache=raw_feature_cache,
                                      loss_weights=loss_weights,
                                      teacher_forcing_ratio=_teacher_forcing_ratio(args, 0),
                                      ablate_frame_avg=_ablate_frame, no_history=_no_history,
                                      augment_images=image_augmentation)
    baseline = float(baseline_losses["total"].detach().cpu())
    history: list[float] = []
    component_history: list[dict[str, float]] = []
    gradient_norm_history: list[float] = []
    gradient_diagnostics: list[dict[str, object]] = []
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
                                      vision_micro_batch_size=vision_micro_batch_size,
                                      raw_feature_cache=raw_feature_cache,
                                      loss_weights=loss_weights,
                                      teacher_forcing_ratio=_teacher_forcing_ratio(args, step - 1),
                                      ablate_frame_avg=_ablate_frame, no_history=_no_history,
                                      augment_images=image_augmentation,
                                      history_dropout_p=history_dropout_p)
        total = losses["total"]
        if not torch.isfinite(total):
            raise FloatingPointError(f"step {step}: loss 非有限值")
        if amp_enabled:
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            invalid_gradients = _gradient_diagnostics(named_trainable)
            gradient_diagnostics.append({"step": step, "scale": float(scaler.get_scale()),
                                         "invalid_parameters": invalid_gradients})
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            invalid_gradients = _gradient_diagnostics(named_trainable)
            gradient_diagnostics.append({"step": step, "scale": None,
                                         "invalid_parameters": invalid_gradients})
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        history.append(float(total.detach().cpu()))
        gradient_norm_history.append(float(gradient_norm.detach().cpu()))
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
                f"grounding={float(losses['grounding_total'].detach().cpu()):.4f} "
                f"move_nonstop_hit={move_hit:.3f} intent_pred={intent_pred} "
                f"tf={_teacher_forcing_ratio(args, step - 1):.3f}", flush=True,
            )
            if step == 1 or step % (log_interval * 4) == 0 or step == args.steps:
                print(f"[interact_stage step={step}] {json.dumps(interact_stages, ensure_ascii=False)}", flush=True)
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))

    mask = _mask_comparison(history, adapter, core, first_batch, device, amp_enabled)
    # Calibrate the independent Q-event decision from training scores only.
    # Validation data is never used to choose this threshold.
    train_event_scores: dict[str, list[torch.Tensor]] = {}
    train_result = _evaluate(
        adapter, core, eval_loader, device, amp_enabled,
        raw_feature_cache=raw_feature_cache, event_score_sink=train_event_scores,
    )
    if not train_event_scores.get("logits"):
        raise RuntimeError("训练集未产生 interact event calibration logits")
    event_threshold, event_threshold_report = _fit_interact_event_threshold(
        torch.cat(train_event_scores["logits"]),
        torch.cat(train_event_scores["targets"]),
        max_false_positive_rate=float(getattr(args, "interact_event_max_fpr", 0.02)),
    )
    train_result = _apply_interact_event_threshold(
        train_result,
        torch.cat(train_event_scores["logits"]),
        torch.cat(train_event_scores["targets"]),
        threshold=event_threshold,
    )
    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    adapter_state = act_adapter_state(adapter)
    torch.save({"checkpoint_schema_version": ACT_CHECKPOINT_SCHEMA,
                "adapter": adapter_state,
                "core": core.state_dict(),
                "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "step": args.steps, "baseline_loss": baseline}, checkpoint)

    # Checkpoint reproducibility: restore into the same modules and compare outputs.
    before, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=False)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    load_visual_grounded_act_checkpoint(adapter, core, saved)
    after, _ = forward_loss(adapter, core, first_batch, device=device, amp_enabled=False)
    elapsed = time.perf_counter() - started
    manifest = build_manifest(
        stage="M3_ACT", parent=None,
        base_model=args.model_path, adapters=["deterministic_raw_grid_2x2", "act_heads"],
        frozen=["qwen_language_vision_backbone"],
        trainable=["condition_projection", "visual_expert",
                   "slow_temporal", "fast_temporal", "slow_head", "fast_head"],
        data={"act": str(Path(args.data)).replace("\\", "/"), "act_glob": VLA_SCHEMA_VERSION_V5},
        counts={"train": len(dataset), "val": len(val_dataset) if val_dataset is not None else 0},
        precision="fp16+GradScaler" if amp_enabled else "fp32",
        training={"method": "action_chunk_sft", "steps": args.steps, "learning_rate": args.lr,
                  "batch_size": batch_size, "vision_micro_batch_size": vision_micro_batch_size,
                  "temporal_dim": args.temporal_dim,
                  "max_samples": args.max_samples, "sampling": sampling,
                  "seed": int(getattr(args, "seed", 0)), "move_stop_weight": args.move_stop_weight,
                  "grad_initial_scale": grad_initial_scale,
                  "history_dropout_p": history_dropout_p,
                  "visual_aux": float(getattr(args, "visual_aux", 1.0)),
                  "grounding_loss_weight": float(getattr(args, "grounding_loss_weight", 1.0)),
                  "grounding_annotations": (str(Path(grounding_annotations).resolve())
                                             if grounding_annotations else None),
                  "prior_scale": float(core.prior_scale), "prior_logits_bounded": True,
                  "camera_prior_scale": float(core.camera_prior_scale),
                  "visual_input_center": visual_center,
                  "visual_feature_scale": float(core.visual_expert.visual_feature_scale),
                  "interact_event_bias_init": interact_event_bias_init.tolist(),
                  "ordinary_button_bias_init": ordinary_button_bias_init.tolist(),
                  "interact_event_threshold": float(event_threshold),
                  "interact_event_threshold_calibration": event_threshold_report,
                  "interact_event_max_false_positive_rate": float(
                      getattr(args, "interact_event_max_fpr", 0.02)),
                  "button_positive_weight": args.button_positive_weight,
                  "move_direction_balance": args.move_direction_balance,
                  "class_balance": class_balance_manifest,
                  "image_augmentation": image_augmentation,
                  "global_move_counts": global_counts.tolist() if global_counts is not None else None,
                  "global_button_counts": button_global_counts.tolist() if button_global_counts is not None else None,
                  "teacher_forcing_start": args.teacher_forcing_start,
                  "teacher_forcing_end": args.teacher_forcing_end,
                  "teacher_forcing_decay_steps": args.teacher_forcing_decay_steps},
        schema_versions=[VLA_SCHEMA_VERSION_V5],
        artifacts={"act_checkpoint": str(checkpoint.name)},
        extra={"initialization": "base_without_m2", "parent_stage": None,
               "val_data": [str(Path(path)).replace("\\", "/") for path in val_paths]},
    )
    manifest_path = checkpoint.parent / "manifest.json"
    write_manifest(manifest_path, manifest)
    # Embed the exact manifest snapshot in the binary as well as writing the
    # human-readable sidecar.  This prevents later experiments in the same
    # directory from making a checkpoint's provenance ambiguous.
    torch.save({**saved, "manifest": manifest}, checkpoint)
    val_result = (_evaluate(adapter, core, val_loader, device, amp_enabled,
                            raw_feature_cache=raw_feature_cache,
                            event_threshold=event_threshold)
                  if val_loader is not None else None)
    baseline_result = None
    baseline_path = str(getattr(args, "acceptance_baseline", "") or "")
    if baseline_path:
        baseline_result = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    acceptance = _balance_acceptance(val_result or {}, baseline_result)
    if getattr(args, "require_acceptance", False) and acceptance["status"] != "pass":
        raise ValueError(f"类别平衡验收未通过: {json.dumps(acceptance, ensure_ascii=False)}")
    training_behavior = training_behavior_gate({
        "steps": args.steps, "loss_history": history,
        "component_loss_history": component_history,
        "gradient_norm_history": gradient_norm_history,
    })
    result = {"samples": len(dataset), "val_samples": len(val_dataset) if val_dataset is not None else 0,
            "steps": args.steps, "baseline_loss": baseline,
            "final_loss": history[-1] if history else baseline,
            "loss_history": history, "component_loss_history": component_history,
            "gradient_norm_history": gradient_norm_history,
            "gradient_diagnostics": gradient_diagnostics,
            "training_behavior_acceptance": training_behavior, "mask": mask,
            "checkpoint": str(checkpoint), "checkpoint_loss_delta": abs(float(before["total"] - after["total"])),
            "peak_memory_mb": peak_memory / (1024 * 1024), "elapsed_sec": elapsed,
            "manifest": str(manifest_path.resolve()),
            "checkpoint_manifest_embedded": True,
            "slow_accuracy": _slow_accuracy(
                adapter, core, eval_loader, device, amp_enabled,
                max_samples=int(getattr(args, "max_eval_samples", 64) or 0) or None,
                raw_feature_cache=raw_feature_cache),
            "train": train_result, "val": val_result,
            "interact_event_threshold": float(event_threshold),
            "interact_event_threshold_calibration": event_threshold_report,
            "class_balance_acceptance": acceptance}
    (checkpoint.parent / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if getattr(args, "require_training_behavior", False) and not training_behavior["pass"]:
        raise ValueError(f"小训练行为验收未通过: {json.dumps(training_behavior, ensure_ascii=False)}")
    return result


def _slow_accuracy(adapter, core, loader, device, amp_enabled, *, max_samples: int | None = None,
                   frame_cache: dict | None = None, frame_stats: dict[str, int] | None = None,
                   raw_feature_cache: Any | None = None):
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
                                  frame_cache=cache, frame_stats=frame_stats,
                                  raw_feature_cache=raw_feature_cache)
            target = batch["intent_target"]
            valid = target != -100
            if bool(valid.any()):
                pred = out.slow.intent_logits.argmax(-1).cpu()
                correct += int((pred[valid] == target[valid]).sum())
                total += int(valid.sum())
    return (float(correct / total) if total else None)


def _classification_metrics(confusion: list[list[int]], *, zero_class: int | None = None) -> dict[str, object]:
    """Return target-oriented recall and explicit no-turn false-action rate."""
    recalls = []
    for target, row in enumerate(confusion):
        total = sum(row)
        recalls.append((row[target] / total) if total else None)
    result: dict[str, object] = {"confusion": confusion, "recall": recalls}
    if zero_class is not None:
        zero_total = sum(confusion[zero_class])
        result["zero_false_turn_rate"] = (
            (zero_total - confusion[zero_class][zero_class]) / zero_total if zero_total else None
        )
    return result


def _fit_interact_event_threshold(logits: torch.Tensor, target: torch.Tensor,
                                  *, max_false_positive_rate: float = 0.15) -> tuple[float, dict[str, float]]:
    """Fit a fixed event threshold using training scores only.

    Among thresholds satisfying the false-positive-rate limit, choose the one
    with highest recall, then the highest threshold for deterministic safety.
    """
    if not 0.0 <= float(max_false_positive_rate) <= 1.0:
        raise ValueError("max_false_positive_rate 必须在 [0,1]")
    scores = logits.detach().reshape(-1).to(torch.float64)
    labels = target.detach().reshape(-1).to(torch.bool)
    if scores.numel() < 1 or scores.numel() != labels.numel():
        raise ValueError("事件 threshold 拟合需要等长非空 logits/target")
    unique = torch.unique(scores).sort().values
    if unique.numel() == 1:
        candidates = unique
    else:
        candidates = torch.cat((
            unique[:1] - 1.0,
            (unique[:-1] + unique[1:]) / 2.0,
            unique[-1:] + 1.0,
        ))
    best = None
    for value in candidates.tolist():
        predicted = scores >= value
        positives = labels.sum().item()
        negatives = (~labels).sum().item()
        tp = (predicted & labels).sum().item()
        fp = (predicted & ~labels).sum().item()
        recall = tp / positives if positives else 0.0
        fpr = fp / negatives if negatives else 0.0
        if fpr <= float(max_false_positive_rate):
            key = (recall, value)
            if best is None or key > best[0]:
                best = (key, float(value), recall, fpr)
    if best is None:
        value = float(scores.max().item() + 1e-6)
        recall = 0.0
        fpr = 0.0
    else:
        _, value, recall, fpr = best
    return value, {"false_positive_rate": float(fpr), "recall": float(recall)}


def _interact_event_metrics(logits: torch.Tensor, target: torch.Tensor,
                            mask: torch.Tensor, *, threshold: float = 0.0) -> dict[str, object]:
    """Report the one-shot Q event independently from held-button metrics."""
    logits = logits.detach()
    target = target.detach()
    mask = mask.bool()
    event_logits = logits[mask]
    event_target = target[mask].to(torch.bool)
    predicted = event_logits >= float(threshold)
    predicted_grid = logits >= float(threshold)
    tp = int((predicted & event_target).sum())
    fp = int((predicted & ~event_target).sum())
    target_positive = int(event_target.sum())
    predicted_positive = int(predicted.sum())
    return {
        "threshold_logit": float(threshold),
        "tp": tp,
        "fp": fp,
        "target_positive": target_positive,
        "predicted_positive": predicted_positive,
        "precision": (tp / predicted_positive if predicted_positive else None),
        "recall": (tp / target_positive if target_positive else None),
        "positive_logit_mean": (float(event_logits[event_target].mean().cpu())
                                 if bool(event_target.any()) else None),
        "negative_logit_mean": (float(event_logits[~event_target].mean().cpu())
                                 if bool((~event_target).any()) else None),
        "per_step_target_positive": (target.detach() * mask.to(target.dtype)).sum(dim=0).cpu().tolist(),
        "per_step_predicted_positive": (predicted_grid * mask).sum(dim=0).cpu().tolist(),
    }


def _apply_interact_event_threshold(result: dict[str, Any], logits: torch.Tensor,
                                    target: torch.Tensor, *, threshold: float) -> dict[str, Any]:
    """Apply a calibrated Q threshold to an existing evaluation result."""
    logits = logits.detach()
    target = target.detach()
    predicted = logits >= float(threshold)
    positive_count = int(predicted.sum())
    target_count = int(target.to(torch.bool).sum())
    updated = dict(result)
    counts = list(result.get("button_pred_positive_counts", [0] * 6))
    counts[0] = positive_count
    updated["button_pred_positive_counts"] = counts
    updated["interact_pred_positive"] = positive_count
    updated["interact_target_positive"] = target_count
    updated["interact_event"] = _interact_event_metrics(
        logits, target, torch.ones_like(logits, dtype=torch.bool), threshold=threshold,
    )
    return updated


def _balance_acceptance(current: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compare held-out class recalls without confusing observability with proof.

    A checkpoint only gets a pass when a supplied baseline proves the rare
    classes improved by at least five points while majority recalls decline by
    at most five points.  The no-turn false-action limit applies regardless.
    """
    rule = {"rare_recall_delta_min": 0.05, "majority_recall_drop_max": 0.05,
            "zero_false_turn_rate_max": 0.15}
    classification = current.get("classification", {})
    zero_rates = [classification.get(head, {}).get("zero_false_turn_rate")
                  for head in ("camera_dx", "camera_dy")]
    zero_failures = [head for head, value in zip(("camera_dx", "camera_dy"), zero_rates)
                     if value is None or float(value) > rule["zero_false_turn_rate_max"]]
    if baseline is None:
        return {"status": "not_comparable", "rule": rule, "zero_bucket_failures": zero_failures,
                "reason": "没有 --acceptance-baseline，不能声明稀有类 recall 显著提升"}
    baseline_classification = baseline.get("classification", {})
    if not baseline_classification and isinstance(baseline.get("val"), dict):
        baseline_classification = baseline["val"].get("classification", {})
    groups = {
        "rare": (("camera_dx", (0, 1, 3, 4)), ("camera_dy", (0, 1, 3, 4)),
                 ("intent", (INTENTS.index("decipher"),))),
        "majority": (("move", (0,)), ("camera_dx", (2,)), ("camera_dy", (2,)), ("intent", (4,))),
    }
    comparisons: dict[str, list[dict[str, Any]]] = {name: [] for name in groups}
    failures = list(zero_failures)
    for group, specs in groups.items():
        for head, indices in specs:
            now_values = classification.get(head, {}).get("recall", [])
            old_values = baseline_classification.get(head, {}).get("recall", [])
            for index in indices:
                now = now_values[index] if index < len(now_values) else None
                old = old_values[index] if index < len(old_values) else None
                delta = (float(now) - float(old)) if now is not None and old is not None else None
                passed = (delta is not None and
                          (delta >= rule["rare_recall_delta_min"] if group == "rare"
                           else delta >= -rule["majority_recall_drop_max"]))
                entry = {"head": head, "class_index": index, "baseline": old,
                         "current": now, "delta": delta, "pass": passed}
                comparisons[group].append(entry)
                if not passed:
                    failures.append(f"{head}[{index}]")
    return {"status": "pass" if not failures else "fail", "rule": rule,
            "zero_bucket_failures": zero_failures, "comparisons": comparisons,
            "failures": failures}


def _evaluate(adapter, core, loader, device, amp_enabled, *, raw_feature_cache: Any | None = None,
              event_threshold: float = 0.0,
              event_score_sink: dict[str, list[torch.Tensor]] | None = None):
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
    move_confusion = [[0] * 9 for _ in range(9)]
    camera_dx_confusion = [[0] * 5 for _ in range(5)]
    camera_dy_confusion = [[0] * 5 for _ in range(5)]
    intent_confusion = [[0] * 8 for _ in range(8)]
    button_pred = [0] * 6
    button_target = [0] * 6
    event_logits_rows: list[torch.Tensor] = []
    event_target_rows: list[torch.Tensor] = []
    event_mask_rows: list[torch.Tensor] = []
    grounding_rows: dict[str, list[torch.Tensor]] = defaultdict(list)
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
                                          frame_cache=frame_cache, frame_stats=frame_stats,
                                          raw_feature_cache=raw_feature_cache)
            losses.append(float(values["total"].detach().cpu()))
            # Collators intentionally keep supervision tensors on CPU.  Move
            # every tensor participating in boolean indexing to the output
            # device; indexing a CPU target with the CUDA mask raises during
            # the final validation pass after training.
            output_device = output.fast.move_logits.device
            visual_output = getattr(output, "visual", None)
            grounding = getattr(visual_output, "grounding", None)
            if grounding is not None:
                for key, value in (
                    ("present_logits", grounding.present_logits), ("bbox", grounding.bbox),
                    ("side_logits", grounding.side_logits), ("prompt_logits", grounding.prompt_logits),
                    ("reachable_logits", grounding.reachable_logits),
                    ("mask", batch["grounding_mask"].to(output_device)),
                    ("present_target", batch["grounding_present_target"].to(output_device)),
                    ("bbox_mask", batch["grounding_bbox_mask"].to(output_device)),
                    ("bbox_target", batch["grounding_bbox_target"].to(output_device)),
                    ("side_target", batch["grounding_side_target"].to(output_device)),
                    ("prompt_target", batch["grounding_prompt_target"].to(output_device)),
                    ("reachable_target", batch["grounding_reachable_target"].to(output_device)),
                ):
                    grounding_rows[key].append(value.detach().cpu())
            mask = batch["fast_loss_mask"].to(output_device).bool()
            for value in output.fast.move_logits.argmax(-1)[mask].detach().cpu().reshape(-1).tolist():
                move_pred[int(value)] += 1
            targets = batch["move_target"].to(output_device)[mask].detach().cpu().reshape(-1).tolist()
            for value in targets:
                move_target[int(value)] += 1
            target_tensor = batch["move_target"].to(output_device)[mask]
            pred_tensor = output.fast.move_logits.argmax(-1)[mask]
            for target, prediction in zip(target_tensor.detach().cpu().reshape(-1).tolist(),
                                          pred_tensor.detach().cpu().reshape(-1).tolist()):
                move_confusion[int(target)][int(prediction)] += 1
            non_stop = target_tensor != 0
            non_stop_total += int(non_stop.sum())
            non_stop_pred += int(((pred_tensor != 0) & non_stop).sum())
            button_logits = output.fast.button_logits[mask]
            button_targets = batch["button_target"].to(output_device)[mask]
            button_pred_tensor = (button_logits > 0).to(torch.int64)
            explicit_event_logits = getattr(output.fast, "interact_event_logits", None)
            event_logits = (explicit_event_logits
                            if explicit_event_logits is not None
                            else output.fast.button_logits[..., 0])
            event_mask = mask
            if event_mask.shape != event_logits.shape:
                if event_mask.ndim == 1 and event_mask.shape[0] == event_logits.shape[0]:
                    event_mask = event_mask.unsqueeze(-1).expand_as(event_logits)
                else:
                    raise ValueError("fast_loss_mask 必须是 [B] 或与 interact_event_logits 同形状")
            event_predictions = (event_logits[event_mask] >= float(event_threshold)).to(torch.int64)
            if event_predictions.ndim == 2 and button_pred_tensor.shape[0] == event_predictions.shape[0]:
                button_pred_tensor[..., 0] = event_predictions
            elif button_pred_tensor.ndim == 3:
                button_pred_tensor[..., 0] = event_predictions.reshape(
                    button_pred_tensor.shape[0], button_pred_tensor.shape[1]
                )
            else:
                button_pred_tensor[:, 0] = event_predictions.reshape(-1)
            for i in range(button_pred_tensor.shape[-1]):
                button_pred[i] += int(button_pred_tensor[..., i].sum())
                button_target[i] += int((button_targets[..., i] > 0).sum())
            event_logits_rows.append(event_logits.detach().cpu())
            event_target_rows.append(batch["button_target"].to(output_device)[..., 0].detach().cpu())
            event_mask_rows.append(event_mask.detach().cpu())
            if event_score_sink is not None:
                event_score_sink.setdefault("logits", []).append(event_logits[event_mask].detach().cpu())
                event_score_sink.setdefault("targets", []).append(
                    batch["button_target"].to(output_device)[..., 0][event_mask].detach().cpu())
            for logits, target_key, confusion in (
                (getattr(output.fast, "camera_dx_logits", None), "camera_dx_target", camera_dx_confusion),
                (getattr(output.fast, "camera_dy_logits", None), "camera_dy_target", camera_dy_confusion),
            ):
                if logits is not None and target_key in batch:
                    predictions = logits.argmax(-1)[mask].detach().cpu().reshape(-1).tolist()
                    targets = batch[target_key].to(output_device)[mask].detach().cpu().reshape(-1).tolist()
                    for target, prediction in zip(targets, predictions):
                        confusion[int(target)][int(prediction)] += 1
            if output.slow is not None:
                valid = (batch["intent_target"].to(output_device) >= 0)
                slow_predictions = output.slow.intent_logits.argmax(-1)[valid].detach().cpu().tolist()
                slow_targets = batch["intent_target"].to(output_device)[valid].detach().cpu().tolist()
                for value in slow_predictions:
                    intent_pred[int(value)] += 1
                for value in slow_targets:
                    intent_target[int(value)] += 1
                for target, prediction in zip(slow_targets, slow_predictions):
                    intent_confusion[int(target)][int(prediction)] += 1
    # Slow-accuracy pass reuses the same cache (and stats) so its forward
    # re-encodes nothing the loss pass above already encoded; the reported
    # hits therefore cover BOTH traversals of this bounded loader (loss pass +
    # slow-accuracy pass).  Without the cache those two passes would encode
    # roughly twice the unique frames, so hit_rate near 0.5 or higher means
    # the second traversal was almost fully served from cache.
    slow_acc = _slow_accuracy(adapter, core, loader, device, amp_enabled,
                              frame_cache=frame_cache, frame_stats=frame_stats,
                              raw_feature_cache=raw_feature_cache)
    hits = frame_stats["hits"]
    total = frame_stats["encoded"] + hits
    if event_logits_rows:
        event_metrics = _interact_event_metrics(
            torch.cat(event_logits_rows), torch.cat(event_target_rows),
            torch.cat(event_mask_rows),
            threshold=event_threshold,
        )
    else:
        event_metrics = None
    grounding_metrics = (_grounding_metrics(
        **{key: torch.cat(values) for key, values in grounding_rows.items()})
        if grounding_rows else None)
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
            "classification": {
                "move": _classification_metrics(move_confusion),
                "camera_dx": _classification_metrics(camera_dx_confusion, zero_class=2),
                "camera_dy": _classification_metrics(camera_dy_confusion, zero_class=2),
                "intent": _classification_metrics(intent_confusion),
            },
            "button_pred_positive_counts": button_pred, "button_target_positive_counts": button_target,
            "interact_pred_positive": button_pred[0], "interact_target_positive": button_target[0],
            "interact_event": event_metrics,
            "grounding": grounding_metrics,
            "move_nonstop_prediction_rate": (non_stop_pred / non_stop_total if non_stop_total else None)}


def _binary_metrics(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int | None]:
    valid = mask.bool()
    if not bool(valid.any()):
        return {"total": 0, "accuracy": None, "precision": None, "recall": None}
    prediction = logits[valid] >= 0
    truth = target[valid] > 0
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    total = int(valid.sum())
    return {"total": total, "accuracy": float((prediction == truth).float().mean()),
            "precision": (tp / (tp + fp) if tp + fp else None),
            "recall": (tp / (tp + fn) if tp + fn else None)}


def _grounding_metrics(*, present_logits: torch.Tensor, bbox: torch.Tensor, side_logits: torch.Tensor,
                       prompt_logits: torch.Tensor, reachable_logits: torch.Tensor, mask: torch.Tensor,
                       present_target: torch.Tensor, bbox_mask: torch.Tensor, bbox_target: torch.Tensor,
                       side_target: torch.Tensor, prompt_target: torch.Tensor,
                       reachable_target: torch.Tensor) -> dict[str, Any]:
    """Score only rows backed by same-session human grounding annotations."""
    valid = mask.bool()
    bbox_valid = valid & bbox_mask.bool()
    side_accuracy = (float((side_logits[valid].argmax(-1) == side_target[valid]).float().mean())
                     if bool(valid.any()) else None)
    bbox_l1 = (float((bbox[bbox_valid] - bbox_target[bbox_valid]).abs().mean())
               if bool(bbox_valid.any()) else None)
    return {"annotated": int(valid.sum()), "bbox_annotated": int(bbox_valid.sum()),
            "presence": _binary_metrics(present_logits, present_target, valid),
            "prompt": _binary_metrics(prompt_logits, prompt_target, valid),
            "reachable": _binary_metrics(reachable_logits, reachable_target, valid),
            "side_accuracy": side_accuracy, "bbox_l1": bbox_l1}


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
    parser.add_argument("--data", required=True, help="vla_chunks_v5.jsonl")
    parser.add_argument("--val-data", default="", help="可选：逗号/分号分隔的 session v5 JSONL 留出集")
    parser.add_argument("--max-val-samples", type=int, default=64,
                        help="验证集最多编码的样本数，0 表示整集")
    parser.add_argument("--max-eval-samples", type=int, default=64,
                        help="训练后训练集诊断最多编码的样本数，0 表示整集")
    parser.add_argument("--model-path", required=True, help="本地 Qwen3-VL checkpoint")
    parser.add_argument("--init-checkpoint", default="", help="已废弃：ACT 不加载 M2_VG")
    parser.add_argument("--allow-base-init", action="store_true", help="已废弃：ACT 默认从基础模型初始化")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vision-micro-batch-size", type=int, default=8,
                        help="每次送入视觉塔的图像数；2080 Ti 建议从 4/8 试起")
    parser.add_argument("--raw-feature-cache", default="",
                        help="增量预编码的冻结视觉特征目录；启用后训练查表")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--temporal-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-initial-scale", type=float, default=1024.0,
                        help="FP16 GradScaler 初始 scale；2080 Ti 默认 1024 以避免首步溢出")
    parser.add_argument("--history-dropout-p", type=float, default=0.5,
                        help="训练时按样本清零完整 action history 的概率；验证/部署始终为 0")
    parser.add_argument("--visual-aux", type=float, default=1.0,
                        help="history-free visual action expert 的显式监督权重")
    parser.add_argument("--grounding-annotations", default="",
                        help="训练期同 session ACT grounding JSONL；部署不会读取该文件")
    parser.add_argument("--grounding-loss-weight", type=float, default=1.0,
                        help="训练期视觉 grounding 辅助 loss 权重；0 可用于消融")
    parser.add_argument("--visual-center-samples", type=int, default=64,
                        help="从训练集分层抽取的 raw visual center 校准样本数")
    parser.add_argument("--move-stop-weight", type=float, default=0.5)
    parser.add_argument("--move-direction-balance", action="store_true",
                        help="batch>=4 时按 batch 频率；batch<4 时按采样训练集全局频率做 inverse-sqrt balance")
    parser.add_argument("--class-balance", action=argparse.BooleanOptionalAction, default=True,
                        help="对 move/camera dx/camera dy/intent 使用训练子集全局有界 inverse-sqrt 权重")
    parser.add_argument("--image-augmentation", action=argparse.BooleanOptionalAction, default=True,
                        help="训练窗口共享色彩/平移增强；验证和缓存路径始终关闭")
    parser.add_argument("--button-positive-weight", type=float, default=4.0)
    parser.add_argument("--interact-event-max-fpr", type=float, default=0.02,
                        help="仅用训练集拟合 Q 阈值时允许的最大逐步假阳性率")
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
    parser.add_argument("--ablate-frame-avg", action="store_true",
                        help="画面边际贡献 ablation：训练时把 frame_features 替换为整个 batch 的平均特征(抹掉帧身份)，用于与正常训练对照损失曲线")
    parser.add_argument("--no-history", action="store_true",
                        help="时序/历史剥离：将 fast 头接入的上一动作条件 history_actions 置零（不再凭上步动作自相关拟合）")
    parser.add_argument("--log-interval", type=int, default=25,
                        help="训练过程分量日志间隔（步）")
    parser.add_argument("--acceptance-baseline", default="",
                        help="可选：旧 metrics.json，用于类别 recall 改善验收")
    parser.add_argument("--require-acceptance", action="store_true",
                        help="提供 baseline 后，验收失败则以错误退出")
    parser.add_argument("--require-training-behavior", action="store_true",
                        help="要求 30–60 步收敛、有限各头 loss 且无梯度爆炸")
    parser.add_argument("--checkpoint", default="checkpoints/vla_minimal.pt")
    args = parser.parse_args(argv)
    result = train(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
