"""Create an editable candidate intent-segment file for one raw VLA session.

The output is deliberately a small number of candidate intervals.  Humans
only fill/adjust the top-level intent; subgoals remain rule-derived.  Candidate
boundaries use strong action/state transitions as hints and must be reviewed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from idv_agent.vla.action_chunk import INTENTS


def _frame_ids(session: Path) -> list[int]:
    frames = sorted(session.glob("frames/*.jpg"), key=lambda p: int(p.stem))
    if not frames:
        raise ValueError(f"没有找到帧: {session / 'frames'}")
    return [int(path.stem) for path in frames]


def _action_boundaries(session: Path, frame_ids: list[int], min_segment_frames: int) -> dict[int, str]:
    """Return candidate boundaries with human-readable reasons."""
    action_path = session / "per_frame_actions.csv"
    if not action_path.is_file():
        return {}
    with action_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_frame = {}
    for row in rows:
        try:
            by_frame[int(row.get("frame_idx", row.get("frame_id")))] = row
        except (TypeError, ValueError):
            continue
    reasons: dict[int, str] = {}
    previous_category = None
    previous_decode = False
    for frame in frame_ids:
        row = by_frame.get(frame, {})
        category = str(row.get("category_name", "")).upper()
        decoding = category == "INTERACT_HOLD" or "decode" in str(row.get("text_action", "")).lower()
        if decoding and not previous_decode and frame >= min_segment_frames:
            reasons[frame] = "进入破译动作（候选 decipher）"
        elif previous_category and category != previous_category:
            # Only retain changes that are likely to represent a semantic phase
            # shift; ordinary LOOK/MOVE alternation is intentionally ignored.
            semantic = {"INTERACT_HOLD", "INTERACT_TAP", "VAULT", "ITEM_1", "SKILL_1_TAP"}
            if category in semantic or previous_category in semantic:
                if frame >= min_segment_frames:
                    reasons.setdefault(frame, f"动作阶段变化 {previous_category}->{category}")
        previous_category = category
        previous_decode = decoding
    return reasons


def _state_boundaries(session: Path, frame_ids: list[int], min_segment_frames: int) -> dict[int, str]:
    """Use an optional session-local state sidecar as a weak boundary hint."""
    path = session / "frame_states.jsonl"
    if not path.is_file():
        return {}
    import json
    states = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            states[int(row["frame"])] = str(row["state"])
        except (KeyError, TypeError, ValueError):
            continue
    reasons = {}
    previous = None
    for frame in frame_ids:
        state = states.get(frame, "")
        if previous and state and state != previous and frame >= min_segment_frames:
            reasons.setdefault(frame, f"帧状态变化 {previous}->{state}")
        if state:
            previous = state
    return reasons


def init(session: Path, output: Path, *, min_segment_frames: int = 15,
         overwrite: bool = False) -> int:
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames 必须为正数")
    if output.exists() and not overwrite:
        raise ValueError(f"已存在 {output}；如需重建请加 --overwrite")
    frame_ids = _frame_ids(session)
    boundaries = {frame_ids[0]: "session 起点"}
    boundaries.update(_action_boundaries(session, frame_ids, min_segment_frames))
    boundaries.update(_state_boundaries(session, frame_ids, min_segment_frames))
    boundaries[frame_ids[-1] + 1] = "session 终点（exclusive）"
    starts = sorted(boundaries)
    rows = []
    for index, start in enumerate(starts[:-1]):
        end = starts[index + 1] - 1
        if end < start:
            continue
        rows.append({
            "segment_id": f"seg_{len(rows):03d}",
            "start_frame": start,
            "end_frame": end,
            "intent": "",
            "candidate_reason": boundaries[start],
            "notes": "填写/确认 intent；必要时调整 start_frame/end_frame",
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("segment_id", "start_frame", "end_frame",
                                                    "intent", "candidate_reason", "notes"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[intent] candidate_segments={len(rows)} output={output.resolve()}")
    print("[intent] 请填写 intent：" + ", ".join(INTENTS))
    print("[intent] 候选边界只是提示；按‘当前要完成什么’调整，不要按每次按键切段")
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成可编辑的 VLA 意图片段候选表")
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-segment-frames", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        init(args.session, args.output or args.session / "intent_segments.csv",
             min_segment_frames=args.min_segment_frames, overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
