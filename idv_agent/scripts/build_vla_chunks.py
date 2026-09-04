"""Build native VLA action-chunk JSONL from a recorded episode.

The output is intentionally independent of the legacy 20-way action schema:
semantic navigation/interaction tokens and continuous camera deltas are the
only labels consumed by a VLA model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from idv_agent.configs.game_mode import DEFAULT_GAME_MODE, mode_token
from idv_agent.labels.extract import extract_session

from idv_agent.vla.action_chunk import (
    ACTION_CHUNK_HORIZON,
    DEFAULT_ACTION_DELAY_FRAMES,
    HISTORY_FRAMES,
    MACRO_FRAMES,
    VLA_SCHEMA_VERSION,
    VLA_SCHEMA_VERSION_V4,
    VLA_SCHEMA_VERSION_V5,
    camera_bucket_from_pixels,
    BUTTON_NAMES,
    CAMERA_BUCKETS,
    INTENTS,
    MOVE_DIRECTIONS,
    validate_record,
    validate_v4_record,
    validate_v5_record,
)
from idv_agent.configs.subgoal import subgoals_for_intent


_MISSING = object()


def _num(value, default=_MISSING, *, field="value", integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        if default is not _MISSING:
            return default
        raise ValueError(f"{field} 必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须是有限数字")
    if integer and not number.is_integer():
        raise ValueError(f"{field} 必须是整数")
    return number


def _move_dir(rows):
    mx = sum(_num(r.get("move_x")) for r in rows) / len(rows)
    my = sum(_num(r.get("move_y")) for r in rows) / len(rows)
    if abs(mx) < 0.25 and my > 0.35: return 1  # north
    if mx > 0.25 and my > 0.25: return 2       # northeast
    if mx > 0.35 and abs(my) < 0.25: return 3  # east
    if mx > 0.25 and my < -0.25: return 4      # southeast
    if abs(mx) < 0.25 and my < -0.35: return 5 # south
    if mx < -0.25 and my < -0.25: return 6     # southwest
    if mx < -0.35 and abs(my) < 0.25: return 7 # west
    if mx < -0.25 and my > 0.25: return 8      # northwest
    return 0


def _camera_bucket(value):
    value = max(-1.0, min(1.0, _num(value, field="camera")))
    if value < -0.6: return -2
    if value < -0.15: return -1
    if value <= 0.15: return 0
    if value <= 0.6: return 1
    return 2


def _buttons(rows):
    names = {str(r.get("category_name", "")).upper() for r in rows}
    held = "+".join(str(r.get("held_keys", "")) for r in rows).lower()
    values = [0] * len(BUTTON_NAMES)
    values[0] = int(any(n.startswith("INTERACT_") for n in names) or "key:q" in held)
    values[1] = int("VAULT" in names or "key:space" in held)
    values[2] = int(any(n.startswith("ITEM_") for n in names) or "key:f" in held)
    values[3] = int("key:e" in held)
    values[4] = int("key:shift" in held)
    values[5] = int("key:ctrl" in held)
    return values


def _action(rows, frame_indices, *, pixel_camera=False):
    # Camera deltas are frame-wise increments.  Averaging them over a macro
    # window makes normal left/right turns cancel out (and turns a real turn
    # into the ``0`` bucket).  Aggregate the net angular displacement instead;
    # the bucket is applied after accumulation.  This keeps the action-chunk
    # label aligned with the movement keys that are active in the same window.
    camera_dx_px = sum(_num(r.get("cam_dx_px"), field="cam_dx_px") for r in rows)
    camera_dy_px = sum(_num(r.get("cam_dy_px"), field="cam_dy_px") for r in rows)
    camera_dx = (camera_bucket_from_pixels(camera_dx_px) if pixel_camera else
                 _camera_bucket(sum(_num(r.get("cam_dx")) for r in rows)))
    camera_dy = (camera_bucket_from_pixels(camera_dy_px) if pixel_camera else
                 _camera_bucket(sum(_num(r.get("cam_dy")) for r in rows)))
    result = {
        "move_dir": _move_dir(rows),
        "camera_dx": camera_dx,
        "camera_dy": camera_dy,
        "buttons": _buttons(rows),
        "duration_frames": len(frame_indices),
    }
    if pixel_camera:
        result["camera_dx_px"] = float(camera_dx_px)
        result["camera_dy_px"] = float(camera_dy_px)
    return result


def _history_action(row, *, pixel_camera=False):
    """Compact one-step action context (no duration needed for history)."""
    action = _action([row], [int(row.get("frame_idx", 0))], pixel_camera=pixel_camera)
    action.pop("duration_frames", None)
    return action


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 无效") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 必须是对象")
            rows.append(value)
    return rows


def _load_intent_segments(session: Path) -> list[dict]:
    """Load human top-level intent segments for v4.

    JSONL is the canonical format; CSV is accepted for convenient manual
    editing.  Frame ranges are inclusive.
    """
    jsonl = session / "intent_segments.jsonl"
    if jsonl.is_file():
        rows = _load_jsonl(jsonl)
    else:
        csv_path = session / "intent_segments.csv"
        if not csv_path.is_file():
            return []
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    segments = []
    for index, row in enumerate(rows):
        try:
            start = int(row.get("start_frame"))
            end = int(row.get("end_frame"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"intent segment {index} 缺少有效 start_frame/end_frame") from exc
        intent = str(row.get("intent", "")).strip()
        if intent not in INTENTS:
            raise ValueError(f"intent segment {index} 的 intent={intent!r} 不在 {INTENTS}")
        if start < 0 or end < start:
            raise ValueError(f"intent segment {index} 帧范围无效")
        segments.append({"start_frame": start, "end_frame": end,
                         "intent": intent, "segment_id": str(row.get("segment_id") or f"seg_{index:03d}")})
    segments.sort(key=lambda row: (row["start_frame"], row["end_frame"]))
    return segments


def _load_frame_states(session: Path) -> dict[int, dict]:
    rows = _load_jsonl(session / "frame_states.jsonl")
    result = {}
    for row in rows:
        value = row.get("frame", row.get("frame_id"))
        try:
            frame = int(value)
        except (TypeError, ValueError):
            continue
        result[frame] = row
    return result


def _derive_subgoal(intent: str, state: dict, action: dict) -> str:
    """Derive a v1 subgoal without requiring a second manual label."""
    state_name = str(state.get("state", "")).strip().lower()
    category = str(action.get("category_name", "")).upper()
    if intent == "decipher" and (
        state_name in {"qte", "calibration"}
        or str(state.get("decode_calibration", "")).lower() in {"1", "yes", "true"}
    ):
        candidate = "handle_qte"
    elif intent == "decipher":
        if str(state.get("interact_prompt", "")).lower() in {"yes", "1", "true"}:
            candidate = "start_decoding"
        elif category.startswith("INTERACT_"):
            candidate = "start_decoding"
        else:
            candidate = "approach_cipher"
    elif intent == "kite":
        if category == "VAULT":
            candidate = "vault_window"
        elif category == "DROP_BOARD":
            candidate = "drop_pallet"
        else:
            candidate = "maintain_distance"
    elif intent == "rescue":
        candidate = "rescue_teammate"
    elif intent == "rotate":
        candidate = "move_to_zone"
    elif intent == "travel":
        # Movement takes precedence when both signals are present.  A
        # camera-only frame represents sweeping the scene to find the cipher.
        moving = (abs(_num(action.get("move_x"), 0.0)) > 1e-6 or
                  abs(_num(action.get("move_y"), 0.0)) > 1e-6)
        looking = (abs(_num(action.get("cam_dx"), 0.0)) > 1e-6 or
                   abs(_num(action.get("cam_dy"), 0.0)) > 1e-6)
        candidate = "move_to_target" if moving else ("find_cipher" if looking else "move_to_target")
    elif intent == "search":
        if category == "OPEN_CHEST":
            candidate = "open_chest"
        elif category.startswith("ITEM_"):
            candidate = "pickup_item"
        else:
            candidate = "find_chest"
    elif intent == "gate":
        candidate = "open_gate"
    elif intent == "idle":
        candidate = "hide" if state_name in {"hidden", "hiding"} else "observe"
    else:
        candidate = "observe"
    if candidate not in subgoals_for_intent(intent):
        # Fall back safely when a state/action signal is not legal for the
        # manually supplied top-level intent.
        candidate = subgoals_for_intent(intent)[0]
    return candidate


def _slow_label_for_frame(frame_idx: int, segments: list[dict], states: dict[int, dict], action: dict) -> dict:
    for segment in segments:
        if segment["start_frame"] <= frame_idx <= segment["end_frame"]:
            intent = segment["intent"]
            return {
                "valid": True,
                "segment_id": segment["segment_id"],
                "intent": intent,
                "subgoal": _derive_subgoal(intent, states.get(frame_idx, {}), action),
                "subgoal_source": "rule",
                "subgoal_rule_version": "subgoal.v1",
            }
    return {"valid": False, "segment_id": "", "intent": "", "subgoal": "none",
            "subgoal_source": "unknown"}


def build(session: Path, output: Path, *, stride: int = 3,
          anchor_stride: int | None = None, history_stride: int | None = None,
          history: int = HISTORY_FRAMES, horizon: int = ACTION_CHUNK_HORIZON,
          macro_frames: int = MACRO_FRAMES, action_delay_frames: int = DEFAULT_ACTION_DELAY_FRAMES,
          outcome: str = "unknown", schema_version: str = VLA_SCHEMA_VERSION,
          slow_period_s: float = 1.0) -> int:
    if horizon != ACTION_CHUNK_HORIZON:
        raise ValueError("VLA 要求固定 horizon=4")
    if schema_version == VLA_SCHEMA_VERSION and history != HISTORY_FRAMES:
        raise ValueError("VLA v3 要求固定 history=3")
    if schema_version in {VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5} and not 3 <= history <= 8:
        raise ValueError("VLA v4/v5 要求 history 在 3..8")
    if schema_version not in {VLA_SCHEMA_VERSION, VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5}:
        raise ValueError(f"不支持 schema_version={schema_version!r}")
    if slow_period_s <= 0:
        raise ValueError("slow_period_s 必须为正数")
    # ``stride`` remains a compatibility alias.  New callers must be able to
    # decorrelate neighboring anchors without changing the temporal spacing
    # inside each observation/history window.
    if anchor_stride is None:
        # v5's canonical sampling separates anchor decorrelation from the
        # temporal spacing inside the observation window.  Legacy v3/v4
        # callers retain the old ``stride`` alias behavior.
        anchor_stride = 12 if schema_version == VLA_SCHEMA_VERSION_V5 else stride
    if history_stride is None:
        history_stride = stride
    if anchor_stride < 1 or history_stride < 1 or macro_frames < 1 or action_delay_frames < 0:
        raise ValueError("stride/macro_frames 必须为正数")
    try:
        frames = sorted((session / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    except ValueError as exc:
        raise ValueError("frames 下所有 JPG 文件名必须是整数") from exc
    if not frames:
        raise ValueError(f"没有帧: {session / 'frames'}")
    if [int(frame.stem) for frame in frames] != list(range(len(frames))):
        raise ValueError("frames 文件名必须从 0 连续递增")
    action_csv = session / "per_frame_actions.csv"
    # New VLA recordings use Raw Input relative motion as the sole camera
    # source.  Always rebuild the frame-aligned action table from it before
    # chunking so a stale/manual table cannot reintroduce absolute-position
    # semantics or an outdated camera aggregation.
    mouse_delta_csv = session / "mouse_deltas.csv"
    if not mouse_delta_csv.is_file():
        raise ValueError(f"缺少 {mouse_delta_csv}；新 VLA session 必须提供 Raw Input 相对位移")
    extract_session(session)
    if not action_csv.is_file():  # defensive: extract_session writes this file
        raise ValueError(f"动作提取失败，未生成 {action_csv}")
    actions = {}
    with action_csv.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or ())
        required = {"timestamp_ns", "move_x", "move_y", "cam_dx", "cam_dy"}
        if schema_version == VLA_SCHEMA_VERSION_V5:
            required |= {"cam_dx_px", "cam_dy_px"}
        if not required.issubset(fields) or not ({"frame_idx", "frame_id"} & fields):
            raise ValueError("per_frame_actions.csv 缺少必要动作列")
        for line_no, row in enumerate(reader, 2):
            try:
                row["frame_idx"] = int(row.get("frame_idx", row.get("frame_id")))
            except (TypeError, ValueError):
                raise ValueError(f"per_frame_actions.csv:{line_no} frame_idx 无效")
            if row["frame_idx"] in actions:
                raise ValueError(f"per_frame_actions.csv:{line_no} frame_idx 重复")
            for field in ("timestamp_ns", "move_x", "move_y", "cam_dx", "cam_dy"):
                _num(row.get(field), field=f"per_frame_actions.csv:{line_no}.{field}",
                     integer=(field == "timestamp_ns"))
            if _num(row.get("timestamp_ns"), field=f"per_frame_actions.csv:{line_no}.timestamp_ns",
                    integer=True) < 0:
                raise ValueError(f"per_frame_actions.csv:{line_no}.timestamp_ns 必须非负")
            actions[row["frame_idx"]] = row
    expected_frames = set(range(len(frames)))
    actual_frames = set(actions)
    if actual_frames != expected_frames:
        missing = sorted(expected_frames - actual_frames)
        extra = sorted(actual_frames - expected_frames)
        raise ValueError(f"per_frame_actions.csv 必须覆盖全部帧；缺少={missing[:10]} 多余={extra[:10]}")
    action_timestamps = [_num(actions[i].get("timestamp_ns"), field=f"action[{i}].timestamp_ns",
                              integer=True) for i in range(len(frames))]
    if action_timestamps != sorted(action_timestamps) or len(set(action_timestamps)) != len(action_timestamps):
        raise ValueError("per_frame_actions.csv timestamp_ns 必须严格递增")
    frame_ts_path = session / "frame_timestamps.csv"
    if frame_ts_path.is_file():
        with frame_ts_path.open(encoding="utf-8-sig", newline="") as f:
            frame_rows = list(csv.DictReader(f))
        if len(frame_rows) != len(frames):
            raise ValueError("frame_timestamps.csv 数量必须与帧文件一致")
        try:
            frame_ids = [int(row["frame_id"]) for row in frame_rows]
            frame_timestamps = [int(row["timestamp_ns"]) for row in frame_rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("frame_timestamps.csv frame_id/timestamp_ns 无效") from exc
        if frame_ids != list(range(len(frames))):
            raise ValueError("frame_timestamps.csv frame_id 必须从 0 连续递增")
        if action_timestamps != frame_timestamps:
            raise ValueError("per_frame_actions.csv timestamp_ns 必须与 frame_timestamps.csv 对齐")

    if schema_version in {VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5}:
        intent_segments = _load_intent_segments(session)
        if not intent_segments:
            raise ValueError(f"{schema_version.rsplit('.', 1)[-1]} 缺少 intent_segments.csv/jsonl（人工只标顶层 intent 的片段文件）")
        frame_states = _load_frame_states(session)
    else:
        intent_segments = []
        frame_states = {}

    fps = 30.0
    mode = DEFAULT_GAME_MODE
    task_name = "find_cipher_and_decode"
    task_instruction = "找到密码机，靠近并进入破译"
    meta = session / "meta.json"
    if meta.is_file():
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            fps = float(raw.get("effective_fps") or raw.get("target_fps") or fps)
            mode = raw.get("mode", DEFAULT_GAME_MODE)
            task_name = str(raw.get("task_name") or task_name)
            task_instruction = str(raw.get("task_instruction") or task_instruction)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    total_future = horizon * macro_frames
    rows = []
    last_slow_ts = None
    last_slow_segment = None
    for anchor in range((history - 1) * history_stride,
                        len(frames) - action_delay_frames - total_future, anchor_stride):
        history_frames = []
        for offset in range(history - 1, -1, -1):
            idx = anchor - offset * history_stride
            frame = frames[idx]
            ts = _num(actions[idx].get("timestamp_ns"), field=f"action[{idx}].timestamp_ns")
            item = {"path": frame.relative_to(session).as_posix(),
                    "frame_index": idx, "timestamp_ns": int(ts)}
            if schema_version in {VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5}:
                item["slow_label"] = _slow_label_for_frame(idx, intent_segments,
                                                            frame_states, actions.get(idx, {}))
            history_frames.append(item)
        chunk = []
        for step in range(horizon):
            first = anchor + action_delay_frames + step * macro_frames + 1
            indices = list(range(first, first + macro_frames))
            chunk.append(_action([actions[i] for i in indices], indices,
                                 pixel_camera=schema_version == VLA_SCHEMA_VERSION_V5))
        obs_end_ts = int(_num(actions[anchor].get("timestamp_ns"), field=f"action[{anchor}].timestamp_ns"))
        action_start_frame = anchor + action_delay_frames + 1
        action_end_frame = action_start_frame + total_future - 1
        action_start_ts = int(_num(actions[action_start_frame].get("timestamp_ns"), field=f"action[{action_start_frame}].timestamp_ns"))
        action_end_ts = int(_num(actions[action_end_frame].get("timestamp_ns"), field=f"action[{action_end_frame}].timestamp_ns"))
        history_actions = []
        for offset in range(history - 1, -1, -1):
            idx = anchor - offset * history_stride
            if idx in actions:
                history_actions.append(_history_action(
                    actions[idx], pixel_camera=schema_version == VLA_SCHEMA_VERSION_V5))
        record = {
            "schema_version": schema_version,
            "episode_id": session.name,
            "anchor_frame": anchor,
            "mode": mode,
            "task": {"name": task_name,
                      "instruction": task_instruction,
                      "mode_token": mode_token(mode)},
            "observations": {"frames": history_frames, "fps": fps,
                             "history_actions": history_actions},
            "action_chunk": chunk,
            "alignment": {
                "mode": "causal_future",
                "action_delay_frames": action_delay_frames,
                "observation_end_frame": anchor,
                "action_start_frame": anchor + action_delay_frames + 1,
                "action_end_frame": action_end_frame,
                "observation_end_timestamp_ns": obs_end_ts,
                "action_start_timestamp_ns": action_start_ts,
                "action_end_timestamp_ns": action_end_ts,
            },
            "quality": {"source": "teacher", "outcome": outcome},
            "auxiliary": {"legacy_action_schema": False},
        }
        if schema_version == VLA_SCHEMA_VERSION:
            # v3 keeps the legacy optional field; v4 uses slow_label.intent
            # as the single canonical intent source.
            record["intent"] = ""
        if schema_version in {VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5}:
            slow_label = _slow_label_for_frame(anchor, intent_segments, frame_states, actions.get(anchor, {}))
            record["slow_label"] = slow_label
            anchor_ts = obs_end_ts
            segment_id = slow_label.get("segment_id") if slow_label.get("valid") else None
            slow_due = bool(slow_label.get("valid")) and (
                last_slow_ts is None
                or anchor_ts - last_slow_ts >= int(slow_period_s * 1e9)
                or segment_id != last_slow_segment
            )
            if slow_due:
                last_slow_ts = anchor_ts
                last_slow_segment = segment_id
            record["loss_mask"] = {"slow": int(slow_due), "fast": 1}
            (validate_v5_record(record) if schema_version == VLA_SCHEMA_VERSION_V5
             else validate_v4_record(record))
        else:
            validate_record(record)
        rows.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) +
                      ("\n" if rows else ""), encoding="utf-8")
    print(f"[vla] episode={session.name} samples={len(rows)} output={output}")
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stride", type=int, default=3,
                        help="兼容旧调用：同时设置 anchor/history stride")
    parser.add_argument("--anchor-stride", type=int, default=None,
                        help="相邻训练样本 anchor 的帧间隔")
    parser.add_argument("--history-stride", type=int, default=None,
                        help="同一观测窗口内历史帧的帧间隔")
    parser.add_argument("--history", type=int, default=None,
                        help="历史帧数；v3 默认 3，v4 默认 8（范围 3..8）")
    parser.add_argument("--macro-frames", type=int, default=MACRO_FRAMES)
    parser.add_argument("--action-delay-frames", type=int, default=DEFAULT_ACTION_DELAY_FRAMES,
                        help="观测结束到动作标签起点的延迟帧数，默认 1")
    parser.add_argument("--outcome", choices=("success", "partial", "failure", "unknown"), default="unknown")
    parser.add_argument("--schema-version", choices=(VLA_SCHEMA_VERSION, VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5),
                        default=VLA_SCHEMA_VERSION)
    parser.add_argument("--slow-period-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    history = args.history if args.history is not None else (
        8 if args.schema_version in {VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5} else HISTORY_FRAMES
    )
    output = args.output or args.session / "vla_chunks.jsonl"
    build(args.session, output, stride=args.stride,
          anchor_stride=args.anchor_stride, history_stride=args.history_stride,
          history=history,
          macro_frames=args.macro_frames,
          action_delay_frames=args.action_delay_frames,
          outcome=args.outcome,
          schema_version=args.schema_version,
          slow_period_s=args.slow_period_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
