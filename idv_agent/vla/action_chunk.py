"""VLA-native action-chunk contract.

This module is deliberately independent from ``configs.schema.ActionCategory``.
The model predicts semantic navigation/interaction tokens plus continuous camera
motion; only the final executor translates them to keyboard/mouse events.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VLA_SCHEMA_VERSION = "vla.action_chunk.v1"

# Fixed lengths keep collation and low-latency rolling inference predictable.
HISTORY_FRAMES = 3
ACTION_CHUNK_HORIZON = 4
MACRO_FRAMES = 6

NAV_ACTIONS = ("forward", "left", "right", "stop_observe")
INTERACTION_ACTIONS = ("none", "tap_q", "hold_decode", "release")
OUTCOMES = ("success", "partial", "failure", "unknown")
SOURCES = ("human", "teacher", "model")


def _err(path: str, message: str) -> ValueError:
    return ValueError(f"VLA record {path}: {message}")


def _finite_number(value: Any, path: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise _err(path, "必须是数字") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise _err(path, "必须是有限数字")
    return number


def _validate_action(action: Any, path: str, *, duration_required: bool = True) -> None:
    if not isinstance(action, Mapping):
        raise _err(path, "必须是对象")
    nav = action.get("nav")
    if nav not in NAV_ACTIONS:
        raise _err(f"{path}.nav", f"必须是 {NAV_ACTIONS}")
    interaction = action.get("interaction", "none")
    if interaction not in INTERACTION_ACTIONS:
        raise _err(f"{path}.interaction", f"必须是 {INTERACTION_ACTIONS}")
    for name in ("camera_dx", "camera_dy"):
        value = _finite_number(action.get(name, 0.0), f"{path}.{name}")
        if not -1.0 <= value <= 1.0:
            raise _err(f"{path}.{name}", "范围必须为 [-1, 1]")
    if duration_required:
        duration = action.get("duration_frames")
        if not isinstance(duration, int) or isinstance(duration, bool) or duration < 1 or duration > 30:
            raise _err(f"{path}.duration_frames", "必须是 1..30 的整数")


def validate_record(record: Mapping[str, Any], *, strict_lengths: bool = True) -> None:
    """Validate one JSON-serializable VLA training/inference record.

    ``auxiliary`` is intentionally ignored by the required model contract: it
    may contain YOLO/state-machine annotations, but inference must work when it
    is absent. Raises ``ValueError`` with a field path on the first violation.
    """
    if not isinstance(record, Mapping):
        raise ValueError("VLA record: 必须是对象")
    if record.get("schema_version") != VLA_SCHEMA_VERSION:
        raise _err("schema_version", f"必须是 {VLA_SCHEMA_VERSION!r}")
    for field in ("episode_id", "anchor_frame"):
        if field not in record:
            raise _err(field, "缺少字段")
    if not isinstance(record["episode_id"], str) or not record["episode_id"]:
        raise _err("episode_id", "必须是非空字符串")
    if not isinstance(record["anchor_frame"], int) or record["anchor_frame"] < 0:
        raise _err("anchor_frame", "必须是非负整数")

    task = record.get("task")
    if not isinstance(task, Mapping) or not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        raise _err("task.instruction", "必须是非空字符串")

    obs = record.get("observations")
    if not isinstance(obs, Mapping):
        raise _err("observations", "必须是对象")
    frames = obs.get("frames")
    if not isinstance(frames, list) or not frames:
        raise _err("observations.frames", "必须是非空数组")
    if strict_lengths and len(frames) != HISTORY_FRAMES:
        raise _err("observations.frames", f"必须正好 {HISTORY_FRAMES} 帧")
    for i, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or not isinstance(frame.get("path"), str) or not frame["path"]:
            raise _err(f"observations.frames[{i}].path", "必须是非空相对路径")
        if frame["path"].startswith(("/", "\\")) or ":" in frame["path"][:3]:
            raise _err(f"observations.frames[{i}].path", "必须使用相对路径")
        if not isinstance(frame.get("frame_index"), int) or frame["frame_index"] < 0:
            raise _err(f"observations.frames[{i}].frame_index", "必须是非负整数")
        _finite_number(frame.get("timestamp_ns"), f"observations.frames[{i}].timestamp_ns")
    fps = _finite_number(obs.get("fps"), "observations.fps")
    if fps <= 0 or fps > 240:
        raise _err("observations.fps", "必须在 (0, 240] 内")

    history = obs.get("history_actions", [])
    if not isinstance(history, list):
        raise _err("observations.history_actions", "必须是数组")
    if strict_lengths and len(history) > HISTORY_FRAMES:
        raise _err("observations.history_actions", f"最多 {HISTORY_FRAMES} 个动作")
    for i, action in enumerate(history):
        _validate_action(action, f"observations.history_actions[{i}]", duration_required=False)

    chunk = record.get("action_chunk")
    if not isinstance(chunk, list) or not chunk:
        raise _err("action_chunk", "必须是非空数组")
    if strict_lengths and len(chunk) != ACTION_CHUNK_HORIZON:
        raise _err("action_chunk", f"必须正好 {ACTION_CHUNK_HORIZON} 步")
    for i, action in enumerate(chunk):
        _validate_action(action, f"action_chunk[{i}]")

    quality = record.get("quality", {})
    if not isinstance(quality, Mapping):
        raise _err("quality", "必须是对象")
    if quality.get("source", "unknown") not in SOURCES:
        raise _err("quality.source", f"必须是 {SOURCES}")
    if quality.get("outcome", "unknown") not in OUTCOMES:
        raise _err("quality.outcome", f"必须是 {OUTCOMES}")

