"""Infer frame-level VG states from raw input events and prompt boxes.

The canonical raw recorder currently writes ``events.csv``.  ``events.jsonl``
is also accepted for sessions exported by other tools.  No manual state entry
is required: Q held + an interact-prompt box means decoding; WASD held means
walking; otherwise the frame is idle.  Chased is deliberately merged into
walking by default until a reliable regulator detector exists.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STATE_CHOICES = ("decoding", "idle", "walking", "chased")
MOVE_KEYS = frozenset({"key:w", "key:a", "key:s", "key:d"})
INTERACT_KEY = "key:q"
PROMPT_CLASS_ID = 2


def _read_events(session: Path) -> list[dict[str, Any]]:
    csv_path = session / "events.csv"
    jsonl_path = session / "events.jsonl"
    if csv_path.is_file():
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if jsonl_path.is_file():
        rows = []
        with jsonl_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{jsonl_path}:{line_no} JSON 无效") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{jsonl_path}:{line_no} 必须是对象")
                rows.append(row)
        return rows
    raise ValueError(f"缺少 {session / 'events.csv'} 或 {jsonl_path}")


def _ts(row: dict[str, Any], path: str) -> int:
    try:
        value = int(float(row.get("timestamp_ns")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}.timestamp_ns 无效") from exc
    if value < 0:
        raise ValueError(f"{path}.timestamp_ns 必须非负")
    return value


@dataclass(frozen=True)
class Interval:
    code: str
    start: int
    end: int

    def contains(self, timestamp_ns: int) -> bool:
        return self.start <= timestamp_ns <= self.end


def _build_intervals(events: list[dict[str, Any]], end_ts: int) -> list[Interval]:
    pending: dict[str, int] = {}
    intervals: list[Interval] = []
    ordered = sorted(enumerate(events), key=lambda item: _ts(item[1], f"events[{item[0]}]"))
    for index, event in ordered:
        kind = str(event.get("kind", "")).lower()
        code = str(event.get("code", "")).lower()
        timestamp = _ts(event, f"events[{index}]")
        if kind in {"key_down", "mouse_down", "button_down"}:
            pending.setdefault(code, timestamp)
        elif kind in {"key_up", "mouse_up", "button_up"}:
            start = pending.pop(code, None)
            if start is not None and timestamp >= start:
                intervals.append(Interval(code, start, timestamp))
    for code, start in pending.items():
        intervals.append(Interval(code, start, end_ts + 1))
    return sorted(intervals, key=lambda item: item.start)


def _prompt_present(label_path: Path) -> bool:
    if not label_path.is_file():
        raise ValueError(f"缺少 YOLO 标签: {label_path}")
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_no} 不是 5 列 YOLO 格式")
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_no} class_id 无效") from exc
        if class_id == PROMPT_CLASS_ID:
            return True
    return False


def _mouse_motion(session: Path) -> tuple[list[int], list[int], list[int]]:
    path = session / "mouse_deltas.csv"
    if not path.is_file():
        return [], [], []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    timestamps, xs, ys = [], [], []
    for index, row in enumerate(rows):
        try:
            timestamps.append(int(float(row["timestamp_ns"])))
            xs.append(int(float(row["dx"])))
            ys.append(int(float(row["dy"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{index + 2} 鼠标位移无效") from exc
    return timestamps, xs, ys


def _mouse_changed_between(timestamp_ns: int, previous_ns: int | None,
                           timestamps: list[int], xs: list[int], ys: list[int]) -> bool:
    if previous_ns is None or not timestamps:
        return False
    start = max(0, bisect_right(timestamps, previous_ns) - 1)
    end = bisect_right(timestamps, timestamp_ns)
    if end - start < 2:
        return False
    return any(xs[i] or ys[i] for i in range(start, end))


def _load_action_states(session: Path) -> dict[int, dict[str, str]]:
    """Load event-derived per-frame action semantics when available.

    The game starts decoding with a short Q tap and then continues without a
    held Q key.  ``labels.extract``/``state_machine`` already turns that
    implicit interval into ``INTERACT_HOLD`` rows; using those rows here keeps
    frame state inference event-derived while avoiding the old Q+prompt-only
    undercount.
    """
    path = session / "per_frame_actions.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        result: dict[int, dict[str, str]] = {}
        for index, row in enumerate(rows, 2):
            try:
                frame = int(row.get("frame_idx", row.get("frame_id")))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{index} frame_idx 无效") from exc
            result[frame] = {
                "category_name": str(row.get("category_name", "")),
                "text_action": str(row.get("text_action", "")),
            }
    return result


def _action_implies_decoding(action: dict[str, str] | None) -> bool:
    if not action:
        return False
    category = action.get("category_name", "").strip().upper()
    text_action = action.get("text_action", "").strip().upper()
    return category == "INTERACT_HOLD" or "INTERACT_HOLD" in text_action or "DECODE" in text_action


def _held(intervals: list[Interval], timestamp_ns: int) -> set[str]:
    return {item.code for item in intervals if item.contains(timestamp_ns)}


def _chased_heuristic(held: set[str], mouse_changed: bool, recent_motions: list[bool],
                      window_frames: int) -> bool:
    """Conservative optional approximation; disabled by default."""
    active_turn = bool(held & MOVE_KEYS) and mouse_changed
    recent_motions.append(active_turn)
    del recent_motions[:-max(1, window_frames)]
    return (active_turn and len(recent_motions) >= window_frames
            and sum(recent_motions) >= math.ceil(window_frames * 0.6))


def infer(dataset: Path, *, repo_root: Path, output: Path, overwrite: bool = False,
          chased_heuristic: bool = False, chased_window_frames: int = 15) -> dict[str, Any]:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"缺少 {manifest}")
    if output.exists() and not overwrite:
        raise ValueError(f"已存在 {output}；如需重建请加 --overwrite")
    if chased_window_frames < 3:
        raise ValueError("chased_window_frames 必须 >= 3")

    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows:
        raise ValueError("manifest.csv 没有图片")

    sessions: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        source = Path(row["source"])
        source_abs = source if source.is_absolute() else repo_root / source
        session = source_abs.parent.parent
        sid = str(row["session"])
        if sid not in sessions:
            events = _read_events(session)
            frame_ts_path = session / "frame_timestamps.csv"
            frame_ts = []
            if frame_ts_path.is_file():
                with frame_ts_path.open(encoding="utf-8-sig", newline="") as handle:
                    frame_rows = list(csv.DictReader(handle))
                    frame_ts = [int(float(item["timestamp_ns"])) for item in frame_rows]
                    frame_ts_by_index = {
                        int(item["frame_id"]): int(float(item["timestamp_ns"])) for item in frame_rows
                    }
            else:
                frame_ts_by_index = {}
            end_ts = max(frame_ts or [int(float(e.get("timestamp_ns", 0))) for e in events] or [0])
            sessions[sid] = {
                "session": session,
                "intervals": _build_intervals(events, end_ts),
                "mouse": _mouse_motion(session),
                "action_states": _load_action_states(session),
                "frame_ts_by_index": frame_ts_by_index,
                "last_ts": None,
                "recent_motions": [],
            }

    rows = []
    counts = {state: 0 for state in STATE_CHOICES}
    for row in manifest_rows:
        sid = str(row["session"])
        item = sessions[sid]
        image_path = Path(row["image"])
        image_abs = image_path if image_path.is_absolute() else repo_root / image_path
        label_path = dataset / "labels" / row["split"] / f"{image_abs.stem}.txt"
        try:
            frame_index = int(image_abs.stem.rsplit("_", 1)[1])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"无法从图片名解析 frame: {image_abs.name}") from exc
        frame_ts = None
        source = Path(row["source"])
        source_abs = source if source.is_absolute() else repo_root / source
        # The frame timestamp CSV is authoritative; fall back to matching the
        # source filename when a minimal export omitted it.
        frame_ts = item["frame_ts_by_index"].get(frame_index)
        if frame_ts is None:
            frame_ts = int(source_abs.stem)
        held = _held(item["intervals"], frame_ts)
        mouse_ts, mouse_x, mouse_y = item["mouse"]
        mouse_changed = _mouse_changed_between(frame_ts, item["last_ts"], mouse_ts, mouse_x, mouse_y)
        item["last_ts"] = frame_ts
        prompt = _prompt_present(label_path)
        action_state = item["action_states"].get(frame_index)
        # Q is a tap in the real game: after the tap, decoding persists even
        # when Q is released and the prompt disappears.  Prefer the
        # event-derived state-machine label when present, while retaining the
        # raw Q+prompt rule for minimal/exported sessions without actions CSV.
        action_decoding = _action_implies_decoding(action_state)
        decoding = action_decoding or (INTERACT_KEY in held and prompt)
        chased = chased_heuristic and _chased_heuristic(
            held, mouse_changed, item["recent_motions"], chased_window_frames)
        if decoding:
            state = "decoding"
        elif chased:
            state = "chased"
        elif held or mouse_changed:
            state = "walking"
        else:
            state = "idle"
        counts[state] += 1
        rows.append({
            "session": sid,
            "frame": frame_index,
            "state": state,
            "evidence": {
                "q_held": INTERACT_KEY in held,
                "interact_prompt": prompt,
                "action_category": action_state.get("category_name", "") if action_state else "",
                "action_text": action_state.get("text_action", "") if action_state else "",
                "action_decoding": action_decoding,
                "move_keys_held": sorted(held & MOVE_KEYS),
                "mouse_changed": mouse_changed,
                "state_rule": "action_interact_hold|q_held+prompt|chased_heuristic|active_input|idle",
            },
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"frames": len(rows), "states": counts, "chased_heuristic": chased_heuristic}
    print(f"[states] frames={len(rows)} states={counts} output={output.resolve()}")
    if not chased_heuristic:
        print("[states] chased 未启用，追击帧按 walking 处理")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="从事件流和 interact_prompt 自动生成帧状态")
    parser.add_argument("dataset", type=Path, help="包含 manifest.csv/images/labels 的数据集")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--chased-heuristic", action="store_true")
    parser.add_argument("--chased-window-frames", type=int, default=15)
    args = parser.parse_args(argv)
    try:
        infer(args.dataset, repo_root=args.repo_root.resolve(),
              output=args.output or args.dataset / "frame_states.jsonl",
              overwrite=args.overwrite, chased_heuristic=args.chased_heuristic,
              chased_window_frames=args.chased_window_frames)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
