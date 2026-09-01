"""Validate manually reviewed VLA intent segments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from idv_agent.vla.action_chunk import INTENTS


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        raise ValueError(f"缺少 {path}")
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} JSON 无效") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_no} 必须是对象")
        rows.append(row)
    return rows


def validate(path: Path, *, frame_start: int | None = None, frame_end: int | None = None,
             require_full_coverage: bool = False, allow_empty: bool = False) -> int:
    rows = _load(path)
    if not rows:
        raise ValueError("没有意图片段")
    normalized = []
    for index, row in enumerate(rows):
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"片段 {index} 缺少有效 start_frame/end_frame") from exc
        intent = str(row.get("intent", "")).strip()
        if start < 0 or end < start:
            raise ValueError(f"片段 {index} 帧范围无效: {start}..{end}")
        if not intent and not allow_empty:
            raise ValueError(f"片段 {index} 尚未填写 intent")
        if intent and intent not in INTENTS:
            raise ValueError(f"片段 {index} intent={intent!r} 不在 {INTENTS}")
        normalized.append((start, end, intent, str(row.get("segment_id", f"seg_{index:03d}"))))
    normalized.sort()
    for previous, current in zip(normalized, normalized[1:]):
        if current[0] <= previous[1]:
            raise ValueError(f"片段重叠: {previous[3]} 与 {current[3]}")
        if require_full_coverage and current[0] != previous[1] + 1:
            raise ValueError(f"片段存在空洞: {previous[1] + 1}..{current[0] - 1}")
    if frame_start is not None and normalized[0][0] != frame_start:
        raise ValueError(f"首片段必须从 frame={frame_start} 开始，当前为 {normalized[0][0]}")
    if frame_end is not None and normalized[-1][1] != frame_end:
        raise ValueError(f"末片段必须结束于 frame={frame_end}，当前为 {normalized[-1][1]}")
    if require_full_coverage and (frame_start is None or frame_end is None):
        raise ValueError("require_full_coverage 需要同时提供 --frame-start 和 --frame-end")
    print(f"[intent] valid segments={len(normalized)} intents=" +
          ", ".join(sorted({item[2] for item in normalized if item[2]})))
    return len(normalized)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="校验 VLA 意图片段")
    parser.add_argument("segments", type=Path)
    parser.add_argument("--frame-start", type=int, default=None)
    parser.add_argument("--frame-end", type=int, default=None)
    parser.add_argument("--require-full-coverage", action="store_true")
    parser.add_argument("--allow-empty", action="store_true", help="只检查边界，允许 intent 为空")
    args = parser.parse_args(argv)
    try:
        validate(args.segments, frame_start=args.frame_start, frame_end=args.frame_end,
                 require_full_coverage=args.require_full_coverage, allow_empty=args.allow_empty)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
