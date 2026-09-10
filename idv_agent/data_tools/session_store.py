"""Safe filesystem access and summaries for raw VLA sessions."""

from __future__ import annotations

import csv
import json
from bisect import bisect_left
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from idv_agent.scripts.validate_vla_raw import validate as validate_raw


MAX_PREVIEWS = 12
MAX_EVENT_ROWS = 120


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _frame_ids(session: Path) -> list[int]:
    frame_dir = session / "frames"
    ids = []
    for path in frame_dir.glob("*.jpg"):
        try:
            ids.append(int(path.stem))
        except ValueError:
            continue
    return sorted(ids)


def _metadata(session: Path) -> dict[str, Any]:
    path = session / "meta.json"
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sample_ids(frame_ids: list[int], limit: int = MAX_PREVIEWS) -> list[int]:
    if len(frame_ids) <= limit:
        return frame_ids
    positions = [round(i * (len(frame_ids) - 1) / (limit - 1)) for i in range(limit)]
    return [frame_ids[position] for position in positions]


def _event_key(code: str) -> str:
    if code.startswith("key:"):
        return code[4:].upper()
    if code.startswith("btn:"):
        return code[4:].upper()
    return code.upper()


def _nearest_frame(timestamp_ns: int, frame_timestamps: list[tuple[int, int]]) -> int | None:
    if not frame_timestamps:
        return None
    timestamp_values = [timestamp for timestamp, _ in frame_timestamps]
    index = bisect_left(timestamp_values, timestamp_ns)
    if index == 0:
        return frame_timestamps[0][1]
    if index == len(frame_timestamps):
        return frame_timestamps[-1][1]
    before = frame_timestamps[index - 1]
    after = frame_timestamps[index]
    return before[1] if timestamp_ns - before[0] <= after[0] - timestamp_ns else after[1]


def _merge_consecutive_keydowns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated key-down reports for display without changing raw data."""
    merged: list[dict[str, Any]] = []
    for event in events:
        if (
            merged
            and event.get("kind") == "key_down"
            and merged[-1].get("kind") == "key_down"
            and event.get("code") == merged[-1].get("code")
        ):
            previous = merged[-1]
            previous["repeat_count"] += event.get("repeat_count", 1)
            previous["end_timestamp_ns"] = event["timestamp_ns"]
            previous["end_time_s"] = event["time_s"]
            previous["end_frame_id"] = event["frame_id"]
            continue
        item = dict(event)
        item["repeat_count"] = 1
        item["end_timestamp_ns"] = event["timestamp_ns"]
        item["end_time_s"] = event["time_s"]
        item["end_frame_id"] = event["frame_id"]
        merged.append(item)
    return merged


class SessionStore:
    """Expose only sessions below one configured raw-session root."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def resolve_session(self, name: str) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"invalid session: {name!r}")
        session = (self.root / name).resolve()
        try:
            session.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"invalid session: {name!r}") from exc
        if not session.is_dir():
            raise FileNotFoundError(f"session not found: {name}")
        return session

    def list_sessions(self) -> list[dict[str, Any]]:
        if not self.root.is_dir():
            return []
        items = []
        for session in sorted((path for path in self.root.iterdir() if path.is_dir()), key=lambda p: p.name):
            if (session / "frames").is_dir():
                items.append(self._session_item(session))
        return items

    def _session_item(self, session: Path) -> dict[str, Any]:
        frames = _frame_ids(session)
        meta = _metadata(session)
        timestamps = _read_csv(session / "frame_timestamps.csv")
        events = _read_csv(session / "events.csv")
        duration = 0.0
        if len(timestamps) >= 2:
            try:
                duration = max(0.0, (int(timestamps[-1]["timestamp_ns"]) - int(timestamps[0]["timestamp_ns"])) / 1e9)
            except (KeyError, TypeError, ValueError):
                pass
        raw_errors = validate_raw(session) if (session / "meta.json").is_file() else ["缺少 meta.json"]
        return {
            "name": session.name,
            "frame_count": len(frames),
            "duration_s": round(duration, 3),
            "fps": float(meta.get("effective_fps") or meta.get("target_fps") or 0),
            "raw_valid": not raw_errors,
            "event_count": len(events),
            "key_event_count": sum(1 for row in events if str(row.get("kind", "")).startswith("key_")),
        }

    def summary(self, name: str) -> dict[str, Any]:
        session = self.resolve_session(name)
        frames = _frame_ids(session)
        meta = _metadata(session)
        timestamps = _read_csv(session / "frame_timestamps.csv")
        events = _read_csv(session / "events.csv")
        previews = [
            {"frame_id": frame_id, "path": f"frames/{frame_id:08d}.jpg"}
            for frame_id in _sample_ids(frames)
        ]
        frame_timestamps = []
        for row in timestamps:
            try:
                frame_timestamps.append((int(row["timestamp_ns"]), int(row["frame_id"])))
            except (KeyError, TypeError, ValueError):
                continue
        frame_timestamps.sort()
        first_timestamp = frame_timestamps[0][0] if frame_timestamps else None
        frame_times = {
            str(frame_id): round((timestamp - first_timestamp) / 1e9, 3)
            for timestamp, frame_id in frame_timestamps
        } if first_timestamp is not None else {}
        key_events = [row for row in events if str(row.get("kind", "")).startswith("key_")]
        key_counts = Counter(_event_key(str(row.get("code", ""))) for row in key_events)
        event_items = []
        for row in events[-MAX_EVENT_ROWS:]:
            try:
                timestamp_ns = int(row.get("timestamp_ns", 0))
            except (TypeError, ValueError):
                continue
            event_items.append({
                "timestamp_ns": timestamp_ns,
                "time_s": round((timestamp_ns - first_timestamp) / 1e9, 3) if first_timestamp is not None else None,
                "kind": str(row.get("kind", "")),
                "code": str(row.get("code", "")),
                "key": _event_key(str(row.get("code", ""))),
                "value": row.get("value", ""),
                "frame_id": _nearest_frame(timestamp_ns, frame_timestamps),
            })
        duration = self._session_item(session)["duration_s"]
        if not duration and len(timestamps) >= 2:
            try:
                duration = max(0.0, (int(timestamps[-1]["timestamp_ns"]) - int(timestamps[0]["timestamp_ns"])) / 1e9)
            except (KeyError, TypeError, ValueError):
                pass
        return {
            **self._session_item(session),
            "metadata": meta,
            "raw_errors": validate_raw(session),
            "frame_start": frames[0] if frames else None,
            "frame_end": frames[-1] if frames else None,
            "frame_ids": frames,
            "frame_times": frame_times,
            "duration_s": round(duration, 3),
            "event_count": len(events),
            "key_event_count": len(key_events),
            "key_counts": dict(sorted(key_counts.items())),
            "events": _merge_consecutive_keydowns(event_items),
            "previews": previews,
        }
