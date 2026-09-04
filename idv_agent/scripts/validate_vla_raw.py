"""Validate a raw VLA recording before annotation/conversion."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from idv_agent.configs.game_mode import GAME_MODE_CHOICES
from idv_agent.vla.action_chunk import VLA_SCHEMA_VERSION


def _number(value, field: str, *, integer: bool = False) -> float | int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} 必须是数字") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须是有限数字")
    if integer:
        if not number.is_integer():
            raise ValueError(f"{field} 必须是整数")
        return int(number)
    return number


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        if not required.issubset(headers):
            missing = ",".join(sorted(required - headers))
            raise ValueError(f"缺少列: {missing}")
        return list(reader)


def _validate_event_rows(rows: list[dict[str, str]], *, start_ts: int | None,
                         end_ts: int | None) -> None:
    previous = None
    for index, row in enumerate(rows, 2):
        timestamp = _number(row.get("timestamp_ns"), f"events.csv:{index}.timestamp_ns", integer=True)
        if timestamp < 0:
            raise ValueError(f"events.csv:{index}.timestamp_ns 必须非负")
        if previous is not None and timestamp < previous:
            raise ValueError("events.csv timestamp_ns 必须非降序")
        if start_ts is not None and timestamp < start_ts:
            raise ValueError(f"events.csv:{index}.timestamp_ns 早于 meta.start_ts_ns")
        if end_ts is not None and timestamp > end_ts:
            raise ValueError(f"events.csv:{index}.timestamp_ns 超出 meta.end_ts_ns")
        if row.get("kind") not in {"key_down", "key_up", "mouse_down", "mouse_up", "scroll"}:
            raise ValueError(f"events.csv:{index}.kind 不是支持的事件类型")
        if not str(row.get("code") or "").strip():
            raise ValueError(f"events.csv:{index}.code 不能为空")
        _number(row.get("value"), f"events.csv:{index}.value")
        previous = timestamp


def _validate_mouse_rows(rows: list[dict[str, str]], *, start_ts: int | None,
                         end_ts: int | None) -> None:
    previous = None
    for index, row in enumerate(rows, 2):
        timestamp = _number(row.get("timestamp_ns"), f"mouse_deltas.csv:{index}.timestamp_ns", integer=True)
        if timestamp < 0:
            raise ValueError(f"mouse_deltas.csv:{index}.timestamp_ns 必须非负")
        if previous is not None and timestamp < previous:
            raise ValueError("mouse_deltas.csv timestamp_ns 必须非降序")
        if start_ts is not None and timestamp < start_ts:
            raise ValueError(f"mouse_deltas.csv:{index}.timestamp_ns 早于 meta.start_ts_ns")
        if end_ts is not None and timestamp > end_ts:
            raise ValueError(f"mouse_deltas.csv:{index}.timestamp_ns 超出 meta.end_ts_ns")
        _number(row.get("dx"), f"mouse_deltas.csv:{index}.dx")
        _number(row.get("dy"), f"mouse_deltas.csv:{index}.dy")
        previous = timestamp


def validate(session: Path) -> list[str]:
    errors = []
    meta_path = session / "meta.json"
    if not meta_path.is_file():
        return ["缺少 meta.json"]
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"meta.json 无法解析: {exc}"]
    if not isinstance(meta, dict):
        return ["meta.json 顶层必须是对象"]
    if meta.get("recording_type") != "vla_raw":
        errors.append("recording_type 不是 vla_raw（旧 session 不可作为 VLA 原始集）")
    if meta.get("vla_schema_version") != VLA_SCHEMA_VERSION:
        errors.append(f"vla_schema_version 必须是 {VLA_SCHEMA_VERSION}")
    if meta.get("mode") not in GAME_MODE_CHOICES:
        errors.append(f"mode 必须是 {GAME_MODE_CHOICES}")
    start_ts = end_ts = None
    for name in ("start_ts_ns", "end_ts_ns"):
        if name in meta and meta[name] is not None:
            try:
                value = _number(meta[name], f"meta.{name}", integer=True)
                if value < 0:
                    raise ValueError("必须非负")
                if name == "start_ts_ns":
                    start_ts = value
                else:
                    end_ts = value
            except ValueError as exc:
                errors.append(str(exc))
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        errors.append("meta.start_ts_ns 不得晚于 meta.end_ts_ns")
    for name in ("num_frames",):
        if name in meta and meta[name] is not None:
            try:
                if _number(meta[name], f"meta.{name}", integer=True) < 0:
                    raise ValueError("必须非负")
            except ValueError as exc:
                errors.append(str(exc))
    for name in ("effective_fps", "target_fps"):
        if name in meta and meta[name] is not None:
            try:
                fps = _number(meta[name], f"meta.{name}")
                if fps <= 0 or fps > 240:
                    raise ValueError("必须在 (0, 240] 内")
            except ValueError as exc:
                errors.append(str(exc))
    for name in ("events.csv", "mouse_deltas.csv", "frame_timestamps.csv"):
        if not (session / name).is_file():
            errors.append(f"缺少 {name}")
    for name, required, validator in (
        ("events.csv", {"timestamp_ns", "kind", "code", "value"}, _validate_event_rows),
        ("mouse_deltas.csv", {"timestamp_ns", "dx", "dy"}, _validate_mouse_rows),
    ):
        path = session / name
        if not path.is_file():
            continue
        try:
            rows = _read_csv(path, required)
            validator(rows, start_ts=start_ts, end_ts=end_ts)
        except Exception as exc:
            errors.append(f"{name} 无法解析: {exc}")

    frames = []
    frame_dir = session / "frames"
    for path in sorted(frame_dir.glob("*.jpg")) if frame_dir.is_dir() else []:
        try:
            frame_id = int(path.stem)
        except ValueError:
            errors.append(f"帧文件名必须是整数: {path.name}")
            continue
        if frame_id < 0 or path.stat().st_size == 0:
            errors.append(f"帧文件无效: {path.name}")
            continue
        frames.append((frame_id, path))
    frames.sort(key=lambda item: item[0])
    ts_path = session / "frame_timestamps.csv"
    timestamps = []
    if ts_path.is_file():
        try:
            for row in _read_csv(ts_path, {"frame_id", "timestamp_ns"}):
                frame_id = _number(row.get("frame_id"), "frame_timestamps.csv.frame_id", integer=True)
                timestamp = _number(row.get("timestamp_ns"), "frame_timestamps.csv.timestamp_ns", integer=True)
                if frame_id < 0 or timestamp < 0:
                    raise ValueError("frame_id 和 timestamp_ns 必须非负")
                timestamps.append((frame_id, timestamp))
        except Exception as exc:
            errors.append(f"frame_timestamps.csv 无法解析: {exc}")
    if len(frames) != len(timestamps):
        errors.append(f"帧数({len(frames)})与时间戳数({len(timestamps)})不一致")
    frame_ids = [frame_id for frame_id, _ in frames]
    if timestamps:
        ids = [x[0] for x in timestamps]
        times = [x[1] for x in timestamps]
        if ids != list(range(len(ids))):
            errors.append("frame_id 必须从 0 连续递增")
        if times != sorted(times) or len(set(times)) != len(times):
            errors.append("timestamp_ns 必须严格递增")
        if frame_ids != ids:
            errors.append("帧文件名必须与 frame_timestamps.csv 的 frame_id 集合一致")
        if start_ts is not None and times and times[0] < start_ts:
            errors.append("首帧 timestamp_ns 早于 meta.start_ts_ns")
        if end_ts is not None and times and times[-1] > end_ts:
            errors.append("末帧 timestamp_ns 晚于 meta.end_ts_ns")
    if "num_frames" in meta and meta["num_frames"] is not None:
        try:
            if _number(meta["num_frames"], "meta.num_frames", integer=True) != len(frames):
                errors.append("meta.num_frames 与帧文件数量不一致")
        except ValueError:
            pass
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
