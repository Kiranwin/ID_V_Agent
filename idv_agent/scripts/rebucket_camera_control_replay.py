"""Synchronize camera-control replay evidence after a v5 bucket migration.

Human ``desired_turn_*`` labels are semantic supervision and are deliberately
left untouched.  Only read-only replay evidence is refreshed from canonical
v5 h0 actions, after the raw sessions have been rebuilt with new bucket edges.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from idv_agent.scripts.prepare_camera_control_annotations import validate


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"camera-control 标注不存在: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("camera-control 标注为空")
    return rows


def _load_chunks(path: Path) -> dict[int, dict[str, Any]]:
    by_anchor: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = int(record["anchor_frame"])
        if key in by_anchor:
            raise ValueError(f"{path} anchor 重复: {key}")
        by_anchor[key] = record
    return by_anchor


def _h0_replay(raw_root: Path, session_name: str, anchor: int,
               cache: dict[str, dict[int, dict[str, Any]]],
               supplemental: dict[tuple[str, int], dict[str, Any]]) -> tuple[int, int]:
    session = (raw_root / session_name).resolve()
    if session.parent != raw_root or not session.is_dir():
        raise ValueError(f"标注 session 不在 raw root 中: {session_name}")
    if session_name not in cache:
        chunks = session / "vla_chunks_v5.jsonl"
        if not chunks.is_file():
            raise FileNotFoundError(f"缺少 v5 chunks: {chunks}")
        cache[session_name] = _load_chunks(chunks)
    record = cache[session_name].get(anchor) or supplemental.get((session_name, anchor))
    if record is None:
        raise ValueError(f"{session_name} 缺少标注 anchor={anchor} 的 v5 chunk")
    action = record.get("action_chunk", [None])[0]
    if not isinstance(action, dict):
        raise ValueError(f"{session_name} anchor={anchor} 缺少 h0 action")
    return int(action["camera_dx"]), int(action["camera_dy"])


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        validate(temporary, require_complete=all(row.get("status") == "complete" for row in rows))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _supplemental_records(root: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if root is None:
        return {}
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"supplemental data root 不存在: {root}")
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for path in root.glob("*/*.jsonl"):
        for anchor, record in _load_chunks(path).items():
            key = (str(record.get("episode_id", "")), anchor)
            if not key[0]:
                raise ValueError(f"{path} 缺少 episode_id")
            prior = records.get(key)
            if prior is not None and prior["action_chunk"][0] != record["action_chunk"][0]:
                raise ValueError(f"supplemental data 对 {key} 有冲突 h0 action")
            records[key] = record
    return records


def rebucket(input_path: str | Path, raw_root: str | Path, output: str | Path,
             *, backup: bool = False, supplemental_data: str | Path | None = None) -> dict[str, Any]:
    """Write a label copy with replay buckets taken from rebuilt v5 h0 rows."""
    input_path, raw_root, output = Path(input_path), Path(raw_root).resolve(), Path(output)
    if not raw_root.is_dir():
        raise ValueError(f"raw root 不存在: {raw_root}")
    rows = _rows(input_path)
    cache: dict[str, dict[int, dict[str, Any]]] = {}
    supplemental = _supplemental_records(Path(supplemental_data) if supplemental_data else None)
    changes = 0
    transitions: dict[str, int] = {}
    for row in rows:
        old = (int(row["replay_camera_dx"]), int(row["replay_camera_dy"]))
        new = _h0_replay(raw_root, str(row["session"]), int(row["frame"]), cache, supplemental)
        transitions[f"dx:{old[0]}->{new[0]}"] = transitions.get(f"dx:{old[0]}->{new[0]}", 0) + 1
        transitions[f"dy:{old[1]}->{new[1]}"] = transitions.get(f"dy:{old[1]}->{new[1]}", 0) + 1
        row["replay_camera_dx"], row["replay_camera_dy"] = new
        changes += int(old != new)
    output.parent.mkdir(parents=True, exist_ok=True)
    if backup and output.resolve() == input_path.resolve():
        backup_path = output.with_name(output.stem + ".before_rebucket675" + output.suffix)
        if backup_path.exists():
            raise FileExistsError(f"保护性备份已存在，拒绝覆盖: {backup_path}")
        shutil.copy2(input_path, backup_path)
    _atomic_write(output, rows)
    return {"input": str(input_path.resolve()), "output": str(output.resolve()),
            "rows": len(rows), "changed_rows": changes, "transitions": dict(sorted(transitions.items())),
            "supplemental_data": str(Path(supplemental_data).resolve()) if supplemental_data else None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--supplemental-data", type=Path,
                        help="MVP train/val JSONL 根目录；仅回退 event-centered anchor")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--in-place", action="store_true",
                        help="原子替换 input；先创建 .before_rebucket675.jsonl 备份")
    args = parser.parse_args(argv)
    if args.in_place == (args.output is not None):
        parser.error("必须二选一：--output 或 --in-place")
    output = args.input if args.in_place else args.output
    report = rebucket(args.input, args.raw_root, output, backup=args.in_place,
                      supplemental_data=args.supplemental_data)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
