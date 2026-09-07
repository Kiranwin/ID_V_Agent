"""Safe persistence for the small, human-authored camera-control label set."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from idv_agent.scripts.prepare_camera_control_annotations import (
    ANNOTATION_SCHEMA,
    validate,
)


SPLITS = ("train", "val")
EDITABLE_FIELDS = (
    "camera_control_phase",
    "camera_target_id",
    "camera_steering_mode",
    "path_strategy",
    "desired_turn_dx",
    "desired_turn_dy",
    "status",
)
IMMUTABLE_FIELDS = (
    "schema_version", "id", "split", "session", "frame", "image_path",
    "existing_grounding", "replay_camera_dx", "replay_camera_dy",
)


class CameraControlStore:
    """Load v2 templates and atomically save an ``.annotated.jsonl`` sibling.

    The pending template is never changed through the web UI.  This makes the
    label work recoverable and prevents a browser request from changing replay
    evidence or grounding context.
    """

    def __init__(self, annotation_dir: Path, raw_root: Path):
        self.annotation_dir = Path(annotation_dir).resolve()
        self.raw_root = Path(raw_root).resolve()
        if not self.annotation_dir.is_dir():
            raise ValueError(f"camera-control 标注目录不存在: {self.annotation_dir}")

    def _pending_path(self, split: str) -> Path:
        self._check_split(split)
        return self.annotation_dir / f"camera_control_{split}.v2.pending.jsonl"

    def _annotated_path(self, split: str) -> Path:
        self._check_split(split)
        return self.annotation_dir / f"camera_control_{split}.v2.annotated.jsonl"

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in SPLITS:
            raise ValueError("split 必须是 train 或 val")

    @staticmethod
    def _read_rows(path: Path) -> list[dict[str, Any]]:
        if not path.is_file():
            raise FileNotFoundError(f"camera-control 标注不存在: {path}")
        validate(path, require_complete=False)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(row.get("schema_version") != ANNOTATION_SCHEMA for row in rows):
            raise ValueError("camera-control 标注 schema 不匹配")
        return rows

    def rows(self, split: str) -> tuple[list[dict[str, Any]], Path]:
        pending = self._read_rows(self._pending_path(split))
        annotated = self._annotated_path(split)
        if annotated.is_file():
            rows = self._read_rows(annotated)
            self._check_annotated_matches_template(pending, rows)
            path = annotated
        else:
            rows, path = pending, self._pending_path(split)
        if any(row.get("split") != split for row in rows):
            raise ValueError(f"{path} 包含错误 split")
        return rows, path

    @staticmethod
    def _check_annotated_matches_template(pending: list[dict[str, Any]],
                                          annotated: list[dict[str, Any]]) -> None:
        """An editable copy cannot add, drop, or rewrite replay evidence."""
        source = {str(row["id"]): row for row in pending}
        candidate = {str(row["id"]): row for row in annotated}
        if set(source) != set(candidate):
            raise ValueError("annotated 标注与 pending 模板的记录集合不一致")
        for identifier, source_row in source.items():
            changed = [field for field in IMMUTABLE_FIELDS
                       if candidate[identifier].get(field) != source_row.get(field)]
            if changed:
                raise ValueError(f"annotated 标注修改了只读字段: {identifier} ({', '.join(changed)})")

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {"splits": {}}
        for split in SPLITS:
            rows, source = self.rows(split)
            complete = sum(row.get("status") == "complete" for row in rows)
            result["splits"][split] = {
                "rows": rows,
                "complete": complete,
                "pending": len(rows) - complete,
                "source": source.name,
                "output": self._annotated_path(split).name,
            }
        return result

    def context_frames(self, split: str, identifier: str) -> list[dict[str, Any]]:
        rows, _ = self.rows(split)
        row = next((item for item in rows if item["id"] == identifier), None)
        if row is None:
            raise FileNotFoundError("camera-control annotation not found")
        session_name = str(row["session"])
        session = (self.raw_root / session_name).resolve()
        if session.parent != self.raw_root or not session.is_dir():
            raise ValueError("annotation session is outside raw root")
        target = int(row["frame"])
        # Canonical v5/m26 causal alignment uses one skipped frame after the
        # observation and then a six-frame h0 macro action.  Annotators need
        # to inspect that recorded outcome, not the already-consumed history.
        h0_start = target + 2
        result = []
        for frame in range(h0_start, h0_start + 6):
            path = (session / "frames" / f"{frame:08d}.jpg").resolve()
            if path.parent != (session / "frames").resolve():
                raise ValueError("invalid frame path")
            result.append({"frame": frame, "exists": path.is_file()})
        return result

    def frame_path(self, split: str, identifier: str, frame: int) -> Path:
        rows, _ = self.rows(split)
        row = next(item for item in rows if item["id"] == identifier)
        allowed = {int(row["frame"])} | {
            item["frame"] for item in self.context_frames(split, identifier)
        }
        if frame not in allowed:
            raise ValueError("frame 不属于当前观察帧或 h0 录制回放")
        session = (self.raw_root / str(row["session"])).resolve()
        path = (session / "frames" / f"{frame:08d}.jpg").resolve()
        if path.parent != (session / "frames").resolve() or not path.is_file():
            raise FileNotFoundError("frame not found")
        return path

    def update(self, split: str, identifier: str, editable: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(editable, Mapping):
            raise ValueError("annotation 必须是对象")
        unknown = set(editable) - set(EDITABLE_FIELDS)
        if unknown:
            raise ValueError("不能修改只读字段: " + ", ".join(sorted(unknown)))
        rows, _ = self.rows(split)
        found = False
        updated: list[dict[str, Any]] = []
        for row in rows:
            if row["id"] != identifier:
                updated.append(row)
                continue
            found = True
            new_row = dict(row)
            new_row.update({key: editable[key] for key in EDITABLE_FIELDS if key in editable})
            updated.append(new_row)
        if not found:
            raise FileNotFoundError("camera-control annotation not found")
        output = self._annotated_path(split)
        self._atomic_validate_write(output, updated)
        return next(row for row in updated if row["id"] == identifier)

    @staticmethod
    def _atomic_validate_write(path: Path, rows: list[dict[str, Any]]) -> None:
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            validate(temporary, require_complete=False)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
