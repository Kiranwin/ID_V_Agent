"""Validate a raw VLA recording before annotation/conversion."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def validate(session: Path) -> list[str]:
    errors = []
    meta_path = session / "meta.json"
    if not meta_path.is_file():
        return ["缺少 meta.json"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"meta.json 无法解析: {exc}"]
    if meta.get("recording_type") != "vla_raw":
        errors.append("recording_type 不是 vla_raw（旧 session 不可作为 VLA 原始集）")
    for name in ("events.csv", "mouse_positions.csv", "frame_timestamps.csv"):
        if not (session / name).is_file():
            errors.append(f"缺少 {name}")
    frames = sorted((session / "frames").glob("*.jpg"), key=lambda p: int(p.stem))
    ts_path = session / "frame_timestamps.csv"
    timestamps = []
    if ts_path.is_file():
        try:
            with ts_path.open(encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    timestamps.append((int(row["frame_id"]), int(row["timestamp_ns"])))
        except Exception as exc:
            errors.append(f"frame_timestamps.csv 无法解析: {exc}")
    if len(frames) != len(timestamps):
        errors.append(f"帧数({len(frames)})与时间戳数({len(timestamps)})不一致")
    if timestamps:
        ids = [x[0] for x in timestamps]
        times = [x[1] for x in timestamps]
        if ids != list(range(len(ids))):
            errors.append("frame_id 必须从 0 连续递增")
        if times != sorted(times) or len(set(times)) != len(times):
            errors.append("timestamp_ns 必须严格递增")
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    errors = validate(args.session)
    if errors:
        for error in errors:
            print(f"[vla-raw] ERROR: {error}")
        return 1
    print(f"[vla-raw] OK: {args.session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
