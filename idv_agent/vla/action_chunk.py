"""VLA-native action-chunk contract.

This module is deliberately independent from ``configs.schema.ActionCategory``.
The model predicts semantic navigation/interaction tokens plus continuous camera
motion; only the final executor translates them to keyboard/mouse events.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from idv_agent.configs.game_mode import GAME_MODE_CHOICES, mode_token
from idv_agent.configs.subgoal import SUBGOAL_NAMES, subgoals_for_intent

VLA_SCHEMA_VERSION = "vla.action_chunk.v3"
VLA_SCHEMA_VERSION_V4 = "vla.action_chunk.v4"
VLA_SCHEMA_VERSION_V5 = "vla.action_chunk.v5"

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
# v5 camera labels are quantized from a macro action's net Raw Input pixels,
# not from a normalized per-frame value.  The same constants also define the
# only legal ACT execution magnitudes, keeping labels and deployment aligned.
# Adjacent command representatives are 0/25/110 px.  Class boundaries sit at
# their midpoints, so a 49--67 px correction remains fine (-/+1), while only
# 68 px or more is a coarse (-/+2) turn.
CAMERA_BUCKET_EDGES_PX = (12.0, 67.5)
CAMERA_BUCKET_COMMAND_PX = {-2: -110.0, -1: -25.0, 0: 0.0, 1: 25.0, 2: 110.0}
BUTTON_NAMES = ("interact", "vault", "item", "heal", "sprint", "crouch")
# Top-level tactical intent; sub-intents are optional metadata, not model heads.
INTENTS = ("decipher", "kite", "rescue", "rotate", "travel", "search", "gate", "idle")
OUTCOMES = ("success", "partial", "failure", "unknown")
SOURCES = ("human", "teacher", "model")
SUBGOAL_SOURCES = ("rule", "human_override", "unknown")


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


def camera_bucket_from_pixels(value: Any) -> int:
    """Quantize one macro action's net camera displacement in pixels."""
    pixels = _finite_number(value, "camera_pixels")
    fine, coarse = CAMERA_BUCKET_EDGES_PX
    if pixels < -coarse:
        return -2
    if pixels < -fine:
        return -1
    if pixels <= fine:
        return 0
    if pixels <= coarse:
        return 1
    return 2


def camera_command_pixels(bucket: int) -> float:
    """Return the v5 physical mouse command for a camera bucket."""
    if isinstance(bucket, bool) or bucket not in CAMERA_BUCKET_COMMAND_PX:
        raise ValueError(f"camera bucket 必须是 {CAMERA_BUCKETS}")
    return CAMERA_BUCKET_COMMAND_PX[bucket]


def _validate_action(action: Any, path: str, *, duration_required: bool = True,
                     camera_pixels_required: bool = False) -> None:
    if not isinstance(action, Mapping):
        raise _err(path, "必须是对象")
    move_dir = action.get("move_dir")
    if not isinstance(move_dir, int) or isinstance(move_dir, bool) or not 0 <= move_dir < len(MOVE_DIRECTIONS):
        raise _err(f"{path}.move_dir", f"必须是 0..{len(MOVE_DIRECTIONS) - 1} 的整数")
    for name in ("camera_dx", "camera_dy"):
        value = action.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value not in CAMERA_BUCKETS:
            raise _err(f"{path}.{name}", f"必须是离散桶 {CAMERA_BUCKETS}")
    if camera_pixels_required:
        for bucket_name, pixel_name in (("camera_dx", "camera_dx_px"),
                                        ("camera_dy", "camera_dy_px")):
            pixels = _finite_number(action.get(pixel_name), f"{path}.{pixel_name}")
            if camera_bucket_from_pixels(pixels) != action[bucket_name]:
                raise _err(f"{path}.{bucket_name}", f"必须与 {pixel_name} 的像素桶一致")
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
    mode = record.get("mode")
    if mode not in GAME_MODE_CHOICES:
        raise _err("mode", f"必须是 {GAME_MODE_CHOICES}")

    task = record.get("task")
    if not isinstance(task, Mapping) or not isinstance(task.get("instruction"), str) or not task["instruction"].strip():
        raise _err("task.instruction", "必须是非空字符串")
    if task.get("mode_token") != mode_token(mode):
        raise _err("task.mode_token", f"必须是 {mode_token(mode)!r}")

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
    expected_start = alignment.get("observation_end_frame", record["anchor_frame"]) + delay + 1
    if alignment["action_start_frame"] != expected_start:
        raise _err("alignment.action_start_frame",
                   "必须等于 observation_end_frame + action_delay_frames + 1；delay 表示两者之间跳过的帧数")
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


def _validate_slow_label(label: Any, path: str) -> None:
    if not isinstance(label, Mapping):
        raise _err(path, "必须是对象")
    valid = label.get("valid")
    if not isinstance(valid, bool):
        raise _err(f"{path}.valid", "必须是布尔值")
    if not valid:
        return
    intent = label.get("intent")
    if intent not in INTENTS:
        raise _err(f"{path}.intent", f"必须是 {INTENTS}")
    subgoal = label.get("subgoal")
    if subgoal not in SUBGOAL_NAMES:
        raise _err(f"{path}.subgoal", "不是 subgoal.v1 词汇")
    if subgoal not in subgoals_for_intent(intent):
        raise _err(f"{path}.subgoal", f"{subgoal!r} 不属于 intent={intent!r} 的候选集合")
    source = label.get("subgoal_source", "unknown")
    if source not in SUBGOAL_SOURCES:
        raise _err(f"{path}.subgoal_source", f"必须是 {SUBGOAL_SOURCES}")
    if source == "rule" and label.get("subgoal_rule_version") != "subgoal.v1":
        raise _err(f"{path}.subgoal_rule_version", "rule 来源必须为 subgoal.v1")


def _validate_fast_slow_record(record: Mapping[str, Any], *, schema_version: str,
                               strict_lengths: bool, camera_pixels_required: bool) -> None:
    """Validate common v4/v5 fast-slow fields without schema drift."""
    if not isinstance(record, Mapping):
        raise ValueError("VLA record: 必须是对象")
    if record.get("schema_version") != schema_version:
        raise _err("schema_version", f"必须是 {schema_version!r}")
    if "intent" in record:
        raise _err("intent", "v4 顶层 intent 已废弃，请使用 slow_label.intent")

    # Reuse all common validation while relaxing v3's fixed history length.
    base = dict(record)
    base["schema_version"] = VLA_SCHEMA_VERSION
    anchor_label = record.get("slow_label", {})
    base["intent"] = anchor_label.get("intent", "") if isinstance(anchor_label, Mapping) else ""
    validate_record(base, strict_lengths=False)

    obs = record["observations"]
    frames = obs["frames"]
    if schema_version == VLA_SCHEMA_VERSION_V5 and strict_lengths and len(frames) != 8:
        raise _err("observations.frames", "v5 必须包含正好 8 帧")
    if schema_version != VLA_SCHEMA_VERSION_V5 and strict_lengths and not 3 <= len(frames) <= 8:
        raise _err("observations.frames", f"{schema_version.rsplit('.', 1)[-1]} 必须包含 3..8 帧")
    history = obs.get("history_actions", [])
    if schema_version == VLA_SCHEMA_VERSION_V5 and strict_lengths and len(history) != 8:
        raise _err("observations.history_actions", "v5 必须包含正好 8 个宏动作 history")
    if schema_version != VLA_SCHEMA_VERSION_V5 and strict_lengths and len(history) > 8:
        raise _err("observations.history_actions", f"{schema_version.rsplit('.', 1)[-1]} 最多 8 个历史动作")
    if camera_pixels_required:
        for i, action in enumerate(history):
            _validate_action(action, f"observations.history_actions[{i}]", duration_required=False,
                             camera_pixels_required=True)
    for i, frame in enumerate(frames):
        if "slow_label" not in frame:
            raise _err(f"observations.frames[{i}].slow_label", "缺少字段")
        _validate_slow_label(frame["slow_label"], f"observations.frames[{i}].slow_label")

    if camera_pixels_required:
        for i, action in enumerate(record["action_chunk"]):
            _validate_action(action, f"action_chunk[{i}]", camera_pixels_required=True)
            if schema_version == VLA_SCHEMA_VERSION_V5 and action["duration_frames"] != MACRO_FRAMES:
                raise _err(f"action_chunk[{i}].duration_frames",
                           f"v5 必须固定为 {MACRO_FRAMES} 帧")

    if "slow_label" not in record:
        raise _err("slow_label", "缺少字段")
    _validate_slow_label(record["slow_label"], "slow_label")

    alignment = record["alignment"]
    action_end_frame = alignment.get("action_end_frame")
    if (not isinstance(action_end_frame, int) or
            isinstance(action_end_frame, bool) or action_end_frame < alignment["action_start_frame"]):
        raise _err("alignment.action_end_frame", "必须是动作起始帧之后的整数")

    masks = record.get("loss_mask")
    if not isinstance(masks, Mapping):
        raise _err("loss_mask", "必须是对象")
    for name in ("slow", "fast"):
        value = masks.get(name)
        if isinstance(value, bool) or value not in (0, 1):
            raise _err(f"loss_mask.{name}", "必须是 0 或 1")


def validate_v4_record(record: Mapping[str, Any], *, strict_lengths: bool = True) -> None:
    """Validate the legacy v4 fast/slow VLA record."""
    _validate_fast_slow_record(record, schema_version=VLA_SCHEMA_VERSION_V4,
                               strict_lengths=strict_lengths, camera_pixels_required=False)


def validate_v5_record(record: Mapping[str, Any], *, strict_lengths: bool = True) -> None:
    """Validate v5 records with pixel-provenanced camera action labels."""
    _validate_fast_slow_record(record, schema_version=VLA_SCHEMA_VERSION_V5,
                               strict_lengths=strict_lengths, camera_pixels_required=True)
