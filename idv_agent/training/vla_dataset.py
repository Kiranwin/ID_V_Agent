"""Dataset and collator for VLA action-chunk v5 records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import Dataset

from idv_agent.configs.game_mode import GAME_MODE_CHOICES
from idv_agent.configs.subgoal import SUBGOAL_NAMES
from idv_agent.vla.action_chunk import CAMERA_BUCKETS, INTENTS, validate_v5_record


CAMERA_TO_INDEX = {value: index for index, value in enumerate(CAMERA_BUCKETS)}
SUBGOAL_SOURCE_WEIGHT = {"rule": 0.5, "human_override": 1.0, "unknown": 0.0}
HISTORY_ACTION_WIDTH = 9  # move + camera dx/dy + six button states
HISTORY_ACTION_STEPS = 8
GROUNDING_SIDE_TO_INDEX = {"none": 0, "left": 1, "center": 2, "right": 3}
CAMERA_CONTROL_ANNOTATION_SCHEMA = "act.camera_control_annotation.v2"
CONTROL_PHASES = ("search", "target_acquire", "target_align", "hold")
TARGET_IDS = ("target_cipher", "other_visible", "none")
STEERING_MODES = ("target_center", "path_follow", "search_sweep", "hold")
PATH_STRATEGIES = ("direct", "detour_left", "detour_right", "unknown")
CAMERA_CONTROL_PHASE_TO_INDEX = {value: index for index, value in enumerate(CONTROL_PHASES)}
CAMERA_CONTROL_TARGET_TO_INDEX = {value: index for index, value in enumerate(TARGET_IDS)}
CAMERA_CONTROL_STEERING_TO_INDEX = {value: index for index, value in enumerate(STEERING_MODES)}
CAMERA_CONTROL_PATH_TO_INDEX = {value: index for index, value in enumerate(PATH_STRATEGIES)}


def _empty_grounding_targets() -> dict[str, torch.Tensor]:
    """Return an explicitly masked training-only grounding target."""
    return {
        "grounding_mask": torch.tensor(0.0),
        "grounding_present_target": torch.tensor(0.0),
        "grounding_bbox_mask": torch.tensor(0.0),
        "grounding_bbox_target": torch.zeros(4, dtype=torch.float32),
        "grounding_side_target": torch.tensor(0, dtype=torch.long),
        "grounding_prompt_target": torch.tensor(0.0),
        "grounding_reachable_target": torch.tensor(0.0),
    }


def _empty_camera_control_targets() -> dict[str, torch.Tensor]:
    """Return masked training-only path/camera semantic supervision."""
    return {
        "camera_control_mask": torch.tensor(0.0),
        "camera_control_phase_target": torch.tensor(0, dtype=torch.long),
        "camera_control_target_id_target": torch.tensor(0, dtype=torch.long),
        "camera_control_steering_target": torch.tensor(0, dtype=torch.long),
        "camera_control_path_target": torch.tensor(0, dtype=torch.long),
        "desired_turn_dx_target": torch.tensor(0, dtype=torch.long),
        "desired_turn_dy_target": torch.tensor(0, dtype=torch.long),
    }


def _annotation_paths(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        paths = [item.strip() for item in str(paths).replace(";", ",").split(",") if item.strip()]
    resolved = [Path(path) for path in paths]
    if not resolved:
        raise ValueError("camera-control 标注路径为空")
    return resolved


class GroundingAnnotationIndex:
    """Read completed same-frame ACT grounding labels once, fail closed.

    The annotation key is exactly ``(session, observation_end_frame)``.  It is
    a training-only lookup; images, runtime policy, and deployment never read
    this index or its source JSONL.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise ValueError(f"ACT grounding 标注不存在: {self.path}")
        self.rows: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        for line_no, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
                session, frame = str(row["session"]), int(row["frame"])
                bbox, side = row["cipher_bbox_xyxy_norm"], row["target_side"]
                prompt, reachable = int(row["interact_prompt"]), int(row["cipher_reachable"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{self.path}:{line_no} grounding 标注字段无效") from exc
            key = (session, frame)
            if key in self.rows:
                raise ValueError(f"{self.path}:{line_no} grounding 标注重复: {key}")
            if side not in GROUNDING_SIDE_TO_INDEX or prompt not in (0, 1) or reachable not in (0, 1):
                raise ValueError(f"{self.path}:{line_no} grounding 枚举值无效")
            present = int(bbox is not None)
            if present:
                if (not isinstance(bbox, list) or len(bbox) != 4 or
                        not all(isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
                                for value in bbox) or float(bbox[0]) >= float(bbox[2]) or
                        float(bbox[1]) >= float(bbox[3]) or side == "none"):
                    raise ValueError(f"{self.path}:{line_no} 密码机 bbox/side 无效")
            elif side != "none" or prompt or reachable:
                raise ValueError(f"{self.path}:{line_no} 无密码机框时不能有 interact_prompt/reachable")
            if prompt and (not present or not reachable):
                raise ValueError(f"{self.path}:{line_no} interact_prompt 必须同时有密码机框且 reachable=1")
            self.rows[key] = {
                "grounding_mask": torch.tensor(1.0),
                "grounding_present_target": torch.tensor(float(present)),
                "grounding_bbox_mask": torch.tensor(float(present)),
                "grounding_bbox_target": torch.tensor(bbox if present else [0.0] * 4, dtype=torch.float32),
                "grounding_side_target": torch.tensor(GROUNDING_SIDE_TO_INDEX[side], dtype=torch.long),
                "grounding_prompt_target": torch.tensor(float(prompt)),
                "grounding_reachable_target": torch.tensor(float(reachable)),
            }
        if not self.rows:
            raise ValueError(f"ACT grounding 标注为空: {self.path}")

    def lookup(self, session: str, observation_end_frame: int) -> dict[str, torch.Tensor]:
        row = self.rows.get((str(session), int(observation_end_frame)))
        if row is not None:
            return {key: value.clone() for key, value in row.items()}
        return _empty_grounding_targets()


class CameraControlAnnotationIndex:
    """Read completed human path/control annotations for visual ACT training.

    The labels describe intended visual control and are never exposed to
    runtime.  A record is keyed exactly by the v5 observation endpoint; h0
    replay remains in the JSONL only as auditable evidence.
    """

    def __init__(self, paths: str | Path | Iterable[str | Path]):
        # Import lazily: the preparer uses the training CLI for data discovery,
        # while this dataset is imported by that CLI.  Loading is still fail
        # closed before any annotation becomes a tensor.
        from idv_agent.scripts.prepare_camera_control_annotations import validate as validate_camera_control_annotations

        self.paths = _annotation_paths(paths)
        self.rows: dict[tuple[str, int], dict[str, torch.Tensor]] = {}
        for path in self.paths:
            validate_camera_control_annotations(path, require_complete=True)
            for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not raw.strip():
                    continue
                try:
                    row = json.loads(raw)
                    session, frame = str(row["session"]), int(row["frame"])
                    phase = CAMERA_CONTROL_PHASE_TO_INDEX[row["camera_control_phase"]]
                    target = CAMERA_CONTROL_TARGET_TO_INDEX[row["camera_target_id"]]
                    steering = CAMERA_CONTROL_STEERING_TO_INDEX[row["camera_steering_mode"]]
                    path_target = CAMERA_CONTROL_PATH_TO_INDEX[row["path_strategy"]]
                    dx = CAMERA_TO_INDEX[int(row["desired_turn_dx"])]
                    dy = CAMERA_TO_INDEX[int(row["desired_turn_dy"])]
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{line_no} camera-control 标注字段无效") from exc
                if row.get("schema_version") != CAMERA_CONTROL_ANNOTATION_SCHEMA:
                    raise ValueError(f"{path}:{line_no} camera-control schema 无效")
                key = (session, frame)
                if key in self.rows:
                    raise ValueError(f"{path}:{line_no} camera-control 标注重复: {key}")
                self.rows[key] = {
                    "camera_control_mask": torch.tensor(1.0),
                    "camera_control_phase_target": torch.tensor(phase, dtype=torch.long),
                    "camera_control_target_id_target": torch.tensor(target, dtype=torch.long),
                    "camera_control_steering_target": torch.tensor(steering, dtype=torch.long),
                    "camera_control_path_target": torch.tensor(path_target, dtype=torch.long),
                    "desired_turn_dx_target": torch.tensor(dx, dtype=torch.long),
                    "desired_turn_dy_target": torch.tensor(dy, dtype=torch.long),
                }
        if not self.rows:
            raise ValueError("camera-control 标注为空")

    def lookup(self, session: str, observation_end_frame: int) -> dict[str, torch.Tensor]:
        row = self.rows.get((str(session), int(observation_end_frame)))
        if row is not None:
            return {key: value.clone() for key, value in row.items()}
        return _empty_camera_control_targets()


class VLASequenceDataset(Dataset):
    """Read one or more v5 JSONL files without loading model-specific pixels.

    Image paths stay lazy so Qwen/SigLIP processors can load them in their own
    adapter.  Validation happens once at dataset construction, not inside every
    training step.
    """

    def __init__(self, jsonl_paths: str | Path | Iterable[str | Path],
                 *, verify_images: bool = True,
                 grounding_annotations: str | Path | None = None,
                 camera_control_annotations: str | Path | Iterable[str | Path] | None = None):
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
                        validate_v5_record(record)
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
        self.grounding = (GroundingAnnotationIndex(grounding_annotations)
                          if grounding_annotations is not None else None)
        self.camera_control = (CameraControlAnnotationIndex(camera_control_annotations)
                               if camera_control_annotations is not None else None)

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
        item = {
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
        if self.grounding is not None:
            item.update(self.grounding.lookup(record["episode_id"],
                                              int(record["alignment"]["observation_end_frame"])))
        else:
            item.update(_empty_grounding_targets())
        if self.camera_control is not None:
            item.update(self.camera_control.lookup(record["episode_id"],
                                                   int(record["alignment"]["observation_end_frame"])))
        else:
            item.update(_empty_camera_control_targets())
        return item


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
            "grounding_mask", "grounding_present_target", "grounding_bbox_mask",
            "grounding_bbox_target", "grounding_side_target", "grounding_prompt_target",
            "grounding_reachable_target",
            "camera_control_mask", "camera_control_phase_target",
            "camera_control_target_id_target", "camera_control_steering_target",
            "camera_control_path_target", "desired_turn_dx_target", "desired_turn_dy_target",
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
