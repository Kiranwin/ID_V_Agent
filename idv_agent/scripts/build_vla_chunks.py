"""Build native VLA action-chunk JSONL from a recorded episode.

The output is intentionally independent of the legacy 20-way action schema:
semantic navigation/interaction tokens and continuous camera deltas are the
only labels consumed by a VLA model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from idv_agent.vla.action_chunk import (
    ACTION_CHUNK_HORIZON,
    DEFAULT_ACTION_DELAY_FRAMES,
    HISTORY_FRAMES,
    MACRO_FRAMES,
    VLA_SCHEMA_VERSION,
    BUTTON_NAMES,
    CAMERA_BUCKETS,
    MOVE_DIRECTIONS,
    validate_record,
)


def _num(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
    value = max(-1.0, min(1.0, _num(value)))
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


def _action(rows, frame_indices):
    n = max(len(rows), 1)
    return {
        "move_dir": _move_dir(rows),
        "camera_dx": _camera_bucket(sum(_num(r.get("cam_dx")) for r in rows) / n),
        "camera_dy": _camera_bucket(sum(_num(r.get("cam_dy")) for r in rows) / n),
        "buttons": _buttons(rows),
        "duration_frames": len(frame_indices),
    }


def _history_action(row):
    """Compact one-step action context (no duration needed for history)."""
    action = _action([row], [int(row.get("frame_idx", 0))])
    action.pop("duration_frames", None)
    return action


def build(session: Path, output: Path, *, stride: int = 3,
          history: int = HISTORY_FRAMES, horizon: int = ACTION_CHUNK_HORIZON,
          macro_frames: int = MACRO_FRAMES, action_delay_frames: int = DEFAULT_ACTION_DELAY_FRAMES,
          outcome: str = "unknown") -> int:
    if history != HISTORY_FRAMES or horizon != ACTION_CHUNK_HORIZON:
        raise ValueError("VLA v1 要求固定 history=3、horizon=4")
    if stride < 1 or macro_frames < 1 or action_delay_frames < 0:
        raise ValueError("stride/macro_frames 必须为正数")
    frames = sorted((session / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise ValueError(f"没有帧: {session / 'frames'}")
    actions = {}
    action_csv = session / "per_frame_actions.csv"
    if not action_csv.is_file():
        raise ValueError(f"缺少 {action_csv}，请先运行 extract")
    with action_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["frame_idx"] = int(row.get("frame_idx", row.get("frame_id")))
            except (TypeError, ValueError):
                continue
            actions[row["frame_idx"]] = row

    fps = 30.0
    meta = session / "meta.json"
    if meta.is_file():
        try:
            raw = json.loads(meta.read_text(encoding="utf-8"))
            fps = float(raw.get("effective_fps") or raw.get("target_fps") or fps)
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    total_future = horizon * macro_frames
    rows = []
    for anchor in range((history - 1) * stride,
                        len(frames) - action_delay_frames - total_future, stride):
        history_frames = []
        for offset in range(history - 1, -1, -1):
            idx = anchor - offset * stride
            frame = frames[idx]
            ts = _num(actions.get(idx, {}).get("timestamp_ns"), 0)
            history_frames.append({"path": frame.relative_to(session).as_posix(),
                                   "frame_index": idx, "timestamp_ns": int(ts)})
        chunk = []
        for step in range(horizon):
            first = anchor + action_delay_frames + step * macro_frames + 1
            indices = list(range(first, first + macro_frames))
            chunk.append(_action([actions.get(i, {"frame_idx": i}) for i in indices], indices))
        obs_end_ts = int(_num(actions.get(anchor, {}).get("timestamp_ns"), 0))
        action_start_frame = anchor + action_delay_frames + 1
        action_end_frame = action_start_frame + total_future - 1
        action_start_ts = int(_num(actions.get(action_start_frame, {}).get("timestamp_ns"), 0))
        action_end_ts = int(_num(actions.get(action_end_frame, {}).get("timestamp_ns"), 0))
        # Synthetic/minimal test sessions may have missing timestamps; keep
        # the contract valid while real recordings retain their exact clocks.
        if action_start_ts <= obs_end_ts:
            action_start_ts = obs_end_ts + 1
        if action_end_ts < action_start_ts:
            action_end_ts = action_start_ts
        history_actions = []
        for offset in range(history - 1, -1, -1):
            idx = anchor - offset * stride
            if idx in actions:
                history_actions.append(_history_action(actions[idx]))
        record = {
            "schema_version": VLA_SCHEMA_VERSION,
            "episode_id": session.name,
            "anchor_frame": anchor,
            "intent": "",
            "task": {"name": "find_cipher_and_decode",
                      "instruction": "找到密码机，靠近并进入破译"},
            "observations": {"frames": history_frames, "fps": fps,
                             "history_actions": history_actions},
            "action_chunk": chunk,
            "alignment": {
                "mode": "causal_future",
                "action_delay_frames": action_delay_frames,
                "observation_end_frame": anchor,
                "action_start_frame": anchor + action_delay_frames + 1,
                "observation_end_timestamp_ns": obs_end_ts,
                "action_start_timestamp_ns": action_start_ts,
                "action_end_timestamp_ns": action_end_ts,
            },
            "quality": {"source": "teacher", "outcome": outcome},
            "auxiliary": {"legacy_action_schema": False},
        }
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
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--macro-frames", type=int, default=MACRO_FRAMES)
    parser.add_argument("--action-delay-frames", type=int, default=DEFAULT_ACTION_DELAY_FRAMES,
                        help="观测结束到动作标签起点的延迟帧数，默认 1")
    parser.add_argument("--outcome", choices=("success", "partial", "failure", "unknown"), default="unknown")
    args = parser.parse_args(argv)
    output = args.output or args.session / "vla_chunks.jsonl"
    build(args.session, output, stride=args.stride,
          macro_frames=args.macro_frames,
          action_delay_frames=args.action_delay_frames,
          outcome=args.outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
