"""VLA-native action-chunk contract.

This module is deliberately independent from ``configs.schema.ActionCategory``.
The model predicts semantic navigation/interaction tokens plus continuous camera
motion; only the final executor translates them to keyboard/mouse events.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

VLA_SCHEMA_VERSION = "vla.action_chunk.v2"

# Fixed lengths keep collation and low-latency rolling inference predictable.
HISTORY_FRAMES = 3
ACTION_CHUNK_HORIZON = 4
MACRO_FRAMES = 6
# A frame is observed before its label is allowed to become a target.  One
# frame leaves room for capture/inference latency and prevents same-frame label
# leakage; deployments can increase this after measuring latency.
DEFAULT_ACTION_DELAY_FRAMES = 1

# A single categorical movement head covers WASD diagonals without averaging
# them into an ambiguous macro.  Index 0 is released/no movement.
MOVE_DIRECTIONS = (
    "stop", "north", "northeast", "east", "southeast",
    "south", "southwest", "west", "northwest",
)
# Five bins per camera axis are enough for a fast first controller.  Raw mouse
# deltas may remain in auxiliary metadata for later re-bucketing.
CAMERA_BUCKETS = (-2, -1, 0, 1, 2)
BUTTON_NAMES = ("interact", "vault", "item", "heal", "sprint", "crouch")
# Top-level tactical intent; sub-intents are optional metadata, not model heads.
INTENTS = ("decipher", "kite", "rescue", "rotate", "travel", "search", "gate", "idle")
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
    move_dir = action.get("move_dir")
    if not isinstance(move_dir, int) or isinstance(move_dir, bool) or not 0 <= move_dir < len(MOVE_DIRECTIONS):
        raise _err(f"{path}.move_dir", f"必须是 0..{len(MOVE_DIRECTIONS) - 1} 的整数")
    for name in ("camera_dx", "camera_dy"):
        value = action.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value not in CAMERA_BUCKETS:
            raise _err(f"{path}.{name}", f"必须是离散桶 {CAMERA_BUCKETS}")
    buttons = action.get("buttons")
    if not isinstance(buttons, list) or len(buttons) != len(BUTTON_NAMES):
        raise _err(f"{path}.buttons", f"必须是长度为 {len(BUTTON_NAMES)} 的 0/1 数组")
    if any(isinstance(v, bool) or v not in (0, 1) for v in buttons):
        raise _err(f"{path}.buttons", "元素必须为 0 或 1")
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
    timestamps = [int(frame["timestamp_ns"]) for frame in frames]
    if timestamps != sorted(timestamps) or len(set(timestamps)) != len(timestamps):
        raise _err("observations.frames", "timestamp_ns 必须严格递增")
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

    alignment = record.get("alignment")
    if not isinstance(alignment, Mapping):
        raise _err("alignment", "必须是对象")
    delay = alignment.get("action_delay_frames")
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 0 or delay > 30:
        raise _err("alignment.action_delay_frames", "必须是 0..30 的整数")
    if alignment.get("mode") != "causal_future":
        raise _err("alignment.mode", "必须是 causal_future")
    if "action_start_frame" not in alignment or not isinstance(alignment["action_start_frame"], int):
        raise _err("alignment.action_start_frame", "必须是整数")
    for name in ("observation_end_timestamp_ns", "action_start_timestamp_ns", "action_end_timestamp_ns"):
        _finite_number(alignment.get(name), f"alignment.{name}")
    if not (alignment["observation_end_timestamp_ns"] <
            alignment["action_start_timestamp_ns"] <= alignment["action_end_timestamp_ns"]):
        raise _err("alignment", "观测结束、动作起止时间必须严格有序")
    intent = record.get("intent", "")
    if intent and intent not in INTENTS:
        raise _err("intent", f"必须是 {INTENTS}")

    quality = record.get("quality", {})
    if not isinstance(quality, Mapping):
        raise _err("quality", "必须是对象")
    if quality.get("source", "unknown") not in SOURCES:
        raise _err("quality.source", f"必须是 {SOURCES}")
    if quality.get("outcome", "unknown") not in OUTCOMES:
        raise _err("quality.outcome", f"必须是 {OUTCOMES}")
