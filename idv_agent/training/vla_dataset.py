"""Dataset and collator for VLA action-chunk v4 records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from idv_agent.configs.game_mode import GAME_MODE_CHOICES
from idv_agent.configs.subgoal import SUBGOAL_NAMES
from idv_agent.vla.action_chunk import CAMERA_BUCKETS, INTENTS, validate_v4_record


CAMERA_TO_INDEX = {value: index for index, value in enumerate(CAMERA_BUCKETS)}
SUBGOAL_SOURCE_WEIGHT = {"rule": 0.5, "human_override": 1.0, "unknown": 0.0}
HISTORY_ACTION_WIDTH = 9  # move + camera dx/dy + six button states
HISTORY_ACTION_STEPS = 8


class VLASequenceDataset(Dataset):
    """Read one or more v4 JSONL files without loading model-specific pixels.

    Image paths stay lazy so Qwen/SigLIP processors can load them in their own
    adapter.  Validation happens once at dataset construction, not inside every
    training step.
    """

    def __init__(self, jsonl_paths: str | Path | Iterable[str | Path],
                 *, verify_images: bool = True):
        if isinstance(jsonl_paths, (str, Path)):
            jsonl_paths = [jsonl_paths]
        self.records: list[tuple[Path, dict]] = []
        for path_like in jsonl_paths:
            path = Path(path_like)
            if not path.is_file():
                raise ValueError(f"VLA JSONL 不存在: {path}")
            with path.open(encoding="utf-8") as handle:
                for line_no, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_no} JSON 无效") from exc
                    try:
                        validate_v4_record(record)
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line_no}: {exc}") from exc
                    image_root = Path(record.get("source_root", path.parent))
                    if verify_images:
                        for frame in record["observations"]["frames"]:
                            image = image_root / frame["path"]
                            if not image.is_file():
                                raise ValueError(f"{path}:{line_no} 图像不存在: {image}")
                    self.records.append((path.parent, record))
        if not self.records:
            raise ValueError("VLASequenceDataset 没有有效样本")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        root, record = self.records[index]
        # Filtered MVP files may live outside the source session.  The
        # preparer preserves source_root while retaining protocol-relative
        # frame paths, so resolve pixels from that original root.
        source_root = record.get("source_root")
        if source_root:
            root = Path(source_root)
        frames = record["observations"]["frames"]
        slow = record["slow_label"]
        chunk = record["action_chunk"]
        slow_valid = bool(slow.get("valid"))
        history = record["observations"].get("history_actions", [])
        history_values = []
        for action in history[-HISTORY_ACTION_STEPS:]:
            history_values.extend([
                float(action["move_dir"]), float(CAMERA_TO_INDEX[action["camera_dx"]]),
                float(CAMERA_TO_INDEX[action["camera_dy"]]),
                *[float(v) for v in action["buttons"]],
            ])
        history_values.extend([0.0] * (HISTORY_ACTION_STEPS * HISTORY_ACTION_WIDTH - len(history_values)))
        return {
            "episode_id": record["episode_id"],
            "anchor_frame": record["anchor_frame"],
            "frame_paths": [root / frame["path"] for frame in frames],
            "task_instruction": record["task"]["instruction"],
            "mode_id": torch.tensor(GAME_MODE_CHOICES.index(record["mode"]), dtype=torch.long),
            "history_actions": torch.tensor(history_values, dtype=torch.float32),
            "intent_target": torch.tensor(INTENTS.index(slow["intent"]) if slow_valid else -100,
                                          dtype=torch.long),
            "subgoal_target": torch.tensor(SUBGOAL_NAMES.index(slow["subgoal"]) if slow_valid else -100,
                                           dtype=torch.long),
            "subgoal_weight": torch.tensor(
                SUBGOAL_SOURCE_WEIGHT.get(slow.get("subgoal_source", "unknown"), 0.0),
                dtype=torch.float32),
            "move_target": torch.tensor([step["move_dir"] for step in chunk], dtype=torch.long),
            "camera_dx_target": torch.tensor(
                [CAMERA_TO_INDEX[step["camera_dx"]] for step in chunk], dtype=torch.long),
            "camera_dy_target": torch.tensor(
                [CAMERA_TO_INDEX[step["camera_dy"]] for step in chunk], dtype=torch.long),
            "button_target": torch.tensor([step["buttons"] for step in chunk], dtype=torch.float32),
            "duration_target": torch.tensor(
                [step["duration_frames"] for step in chunk], dtype=torch.float32),
            "slow_loss_mask": torch.tensor(record["loss_mask"]["slow"], dtype=torch.float32),
            "fast_loss_mask": torch.tensor(record["loss_mask"]["fast"], dtype=torch.float32),
        }


class VLASequenceCollator:
    """Right-pad temporal metadata while leaving image loading to adapters."""

    def __init__(self, max_frames: int = 8):
        if not 3 <= max_frames <= 8:
            raise ValueError("max_frames 必须在 3..8")
        self.max_frames = int(max_frames)

    def __call__(self, samples: list[dict]) -> dict:
        if not samples:
            raise ValueError("samples 不能为空")
        batch = len(samples)
        valid_mask = torch.zeros((batch, self.max_frames), dtype=torch.bool)
        frame_paths: list[list[Path | None]] = []
        for row, sample in enumerate(samples):
            n_frames = len(sample["frame_paths"])
            if n_frames > self.max_frames:
                raise ValueError("样本帧数超过 max_frames")
            paths = list(sample["frame_paths"]) + [None] * (self.max_frames - n_frames)
            frame_paths.append(paths)
            valid_mask[row, :n_frames] = True
        tensor_keys = (
            "mode_id", "history_actions", "intent_target", "subgoal_target", "subgoal_weight",
            "move_target", "camera_dx_target", "camera_dy_target", "button_target",
            "duration_target", "slow_loss_mask", "fast_loss_mask",
        )
        batch_dict = {key: torch.stack([sample[key] for sample in samples]) for key in tensor_keys}
        batch_dict.update({
            "episode_id": [sample["episode_id"] for sample in samples],
            "anchor_frame": torch.tensor([sample["anchor_frame"] for sample in samples], dtype=torch.long),
            "task_instruction": [sample["task_instruction"] for sample in samples],
            "frame_paths": frame_paths,
            "frame_valid_mask": valid_mask,
        })
        return batch_dict
