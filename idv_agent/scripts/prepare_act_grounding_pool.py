"""Export same-session frames for manual ACT visual-grounding annotation.

The MVP action chunks record replayed mouse movement, but do not say where a
cipher was on the decision frame.  This command creates a *copy* of the small,
high-value subset needed to add that missing supervision:

* every Q event frame and its six-frame context on each side;
* deterministic representative decision frames for every non-zero camera
  bucket, independently for train and validation splits.

The output intentionally uses the repository's existing YOLO layout.  Boxes
are annotated through X-AnyLabeling sidecars then converted with
``convert_anylabeling_to_yolo``.  Fields that cannot be expressed as boxes
live in the separate editable ``act_grounding_annotations.jsonl`` template.
Neither source JSONL nor source frames are modified.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CLASSES = ("cipher_visible", "cipher_highlight", "interact_prompt")
SCHEMA_VERSION = "act.grounding_annotation_pool.v1"
AUXILIARY_SCHEMA_VERSION = "act.grounding_auxiliary.v1"


def _json_records(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no} JSON 无效") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_no} 样本必须是对象")
        records.append(record)
    return records


def _source_session(record: dict[str, Any], dataset_file: Path) -> Path:
    value = record.get("source_root")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{dataset_file}: 样本缺少 source_root，不能定位原始帧")
    session = Path(value)
    if not session.is_dir():
        raise ValueError(f"{dataset_file}: source_root 不存在: {session}")
    if not (session / "frames").is_dir():
        raise ValueError(f"{dataset_file}: source_root 缺少 frames/: {session}")
    return session


def _integer(record: dict[str, Any], path: str) -> int:
    value: Any = record
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"样本缺少 {path}")
        value = value[part]
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"样本 {path} 必须是整数") from exc


def _action_starts(record: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    actions = record.get("action_chunk")
    if not isinstance(actions, list) or not actions:
        raise ValueError("样本缺少非空 action_chunk")
    start = _integer(record, "alignment.action_start_frame")
    result = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise ValueError(f"action_chunk[{index}] 必须是对象")
        result.append((start, action))
        duration = action.get("duration_frames")
        if not isinstance(duration, int) or duration <= 0:
            raise ValueError(f"action_chunk[{index}].duration_frames 必须是正整数")
        start += duration
    return result


def _reason_key(reason: dict[str, Any]) -> tuple:
    return tuple((key, json.dumps(value, sort_keys=True, ensure_ascii=False))
                 for key, value in sorted(reason.items()))


def _add(selected: dict[tuple[str, str, int], list[dict[str, Any]]], *, split: str,
         session: Path, frame: int, reason: dict[str, Any]) -> None:
    path = session / "frames" / f"{frame:08d}.jpg"
    if not path.is_file():
        raise ValueError(f"选中的来源帧不存在: {path}")
    reasons = selected[(split, session.name, frame)]
    if _reason_key(reason) not in {_reason_key(item) for item in reasons}:
        reasons.append(reason)


def _evenly_spaced(items: list[tuple[Path, int, dict[str, Any]]], count: int) -> list[tuple[Path, int, dict[str, Any]]]:
    """Pick deterministic coverage points without selecting only one burst."""
    if len(items) <= count:
        return items
    positions = [round(i * (len(items) - 1) / (count - 1)) for i in range(count)] if count > 1 else [len(items) // 2]
    return [items[index] for index in positions]


def _dataset_files(dataset: Path) -> list[tuple[str, Path]]:
    result = []
    for split in ("train", "val"):
        directory = dataset / split
        if not directory.is_dir():
            continue
        result.extend((split, path) for path in sorted(directory.glob("*.jsonl")))
    if not result:
        raise ValueError(f"未找到 {dataset}/train/*.jsonl 或 {dataset}/val/*.jsonl")
    return result


def _ensure_new_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"标注输出目录已有文件，拒绝覆盖: {output}")


def prepare(dataset: Path, output: Path, *, camera_per_bucket: int = 8,
            q_radius_frames: int = 6) -> dict[str, Any]:
    """Create an annotation pool while preserving canonical split membership."""
    dataset, output = Path(dataset), Path(output)
    if camera_per_bucket <= 0:
        raise ValueError("camera_per_bucket 必须为正整数")
    if q_radius_frames < 0:
        raise ValueError("q_radius_frames 不能小于 0")
    _ensure_new_output(output)

    selected: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    camera_candidates: dict[tuple[str, str, int], list[tuple[Path, int, dict[str, Any]]]] = defaultdict(list)
    source_sessions: dict[str, Path] = {}
    q_events = 0
    source_rows = 0

    for split, dataset_file in _dataset_files(dataset):
        for record in _json_records(dataset_file):
            source_rows += 1
            session = _source_session(record, dataset_file)
            previous = source_sessions.setdefault(session.name, session)
            if previous != session:
                raise ValueError(f"同名 session 指向不同 source_root: {session.name}")
            decision_frame = _integer(record, "alignment.observation_end_frame")
            for action_start, action in _action_starts(record):
                buttons = action.get("buttons")
                if not isinstance(buttons, list) or len(buttons) < 1:
                    raise ValueError("action_chunk.buttons 必须是非空数组")
                if int(buttons[0]):
                    q_events += 1
                    for offset in range(-q_radius_frames, q_radius_frames + 1):
                        frame = action_start + offset
                        if frame < 0:
                            continue
                        _add(selected, split=split, session=session, frame=frame,
                             reason={"kind": "q_window", "q_frame": action_start, "offset": offset})
                for axis in ("camera_dx", "camera_dy"):
                    bucket = action.get(axis)
                    if not isinstance(bucket, int):
                        raise ValueError(f"action_chunk.{axis} 必须是整数")
                    if bucket:
                        reason = {
                            "kind": "camera_nonzero", "axis": axis,
                            f"{axis}_bucket": bucket,
                            "decision_frame": decision_frame,
                            "action_start_frame": action_start,
                        }
                        camera_candidates[(split, axis, bucket)].append((session, decision_frame, reason))

    for (split, _axis, _bucket), candidates in camera_candidates.items():
        unique = {(session.name, frame): (session, frame, reason)
                  for session, frame, reason in candidates}
        for session, frame, reason in _evenly_spaced(
                sorted(unique.values(), key=lambda item: (item[0].name, item[1])), camera_per_bucket):
            _add(selected, split=split, session=session, frame=frame, reason=reason)

    manifest: list[dict[str, Any]] = []
    auxiliary: list[dict[str, Any]] = []
    for split, session_name, frame in sorted(selected):
        session = source_sessions[session_name]
        stem = f"{session_name}_{frame:08d}"
        image = output / "images" / split / f"{stem}.jpg"
        label = output / "labels" / split / f"{stem}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(session / "frames" / f"{frame:08d}.jpg", image)
        label.write_text("", encoding="utf-8")
        reasons = sorted(selected[(split, session_name, frame)], key=_reason_key)
        row = {
            "schema_version": SCHEMA_VERSION,
            "id": stem,
            "image": image.relative_to(output).as_posix(),
            "label": label.relative_to(output).as_posix(),
            "split": split,
            "session": session_name,
            "frame": frame,
            "source_image": str((session / "frames" / f"{frame:08d}.jpg").resolve()),
            "reasons": reasons,
        }
        manifest.append(row)
        auxiliary.append({
            "schema_version": AUXILIARY_SCHEMA_VERSION,
            "id": stem,
            "split": split,
            "session": session_name,
            "frame": frame,
            "cipher_bbox_xyxy_norm": None,
            "cipher_reachable": None,
            "target_side": None,
            "interact_prompt": None,
            "selection_reasons": reasons,
        })

    (output / "dataset.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"names: {list(CLASSES)!r}\n", encoding="utf-8")
    (output / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest), encoding="utf-8")
    (output / "act_grounding_annotations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in auxiliary), encoding="utf-8")
    by_split = Counter(row["split"] for row in manifest)
    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset": str(dataset.resolve()),
        "output": str(output.resolve()),
        "source_rows": source_rows,
        "q_events": q_events,
        "frames": {split: by_split.get(split, 0) for split in ("train", "val")},
        "camera_candidates": {f"{split}.{axis}.{bucket}": len(items)
                              for (split, axis, bucket), items in sorted(camera_candidates.items())},
        "camera_per_bucket": camera_per_bucket,
        "q_radius_frames": q_radius_frames,
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    if not all(isinstance(item, (int, float)) and 0.0 <= float(item) <= 1.0 for item in value):
        return False
    return float(value[0]) < float(value[2]) and float(value[1]) < float(value[3])


def validate_annotations(pool: Path) -> dict[str, Any]:
    """Fail closed until every exported frame has complete auxiliary labels.

    A no-cipher frame is represented by ``cipher_bbox_xyxy_norm=null`` and
    ``cipher_reachable=0``.  It is still mandatory to set target_side and
    interact_prompt explicitly so missing work can never silently become a
    negative sample.
    """
    pool = Path(pool)
    manifest_path = pool / "manifest.jsonl"
    annotations_path = pool / "act_grounding_annotations.jsonl"
    if not manifest_path.is_file() or not annotations_path.is_file():
        raise ValueError("标注池缺少 manifest.jsonl 或 act_grounding_annotations.jsonl")
    manifest = _json_records(manifest_path)
    annotations = _json_records(annotations_path)
    expected = {str(row.get("id")): row for row in manifest}
    actual: dict[str, dict[str, Any]] = {}
    for line_no, row in enumerate(annotations, 1):
        identifier = row.get("id")
        if not isinstance(identifier, str) or identifier not in expected:
            raise ValueError(f"辅助标注第 {line_no} 行 id 不在 manifest: {identifier!r}")
        if identifier in actual:
            raise ValueError(f"辅助标注 id 重复: {identifier}")
        actual[identifier] = row
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"辅助标注与 manifest 不一致: 缺少={missing[:3]} 多余={extra[:3]}")
    for identifier, row in actual.items():
        bbox, reachable = row.get("cipher_bbox_xyxy_norm"), row.get("cipher_reachable")
        side, prompt = row.get("target_side"), row.get("interact_prompt")
        if side not in ("left", "center", "right", "none") or prompt not in (0, 1):
            raise ValueError(f"辅助标注未完成或无效: {identifier}；target_side=left/center/right/none，interact_prompt=0/1")
        if reachable not in (0, 1):
            raise ValueError(f"辅助标注未完成或无效: {identifier}；cipher_reachable 必须为 0/1")
        if bbox is None:
            if reachable != 0 or side != "none":
                raise ValueError(f"无密码机框时必须 cipher_reachable=0 且 target_side=none: {identifier}")
        elif not _valid_bbox(bbox):
            raise ValueError(f"cipher_bbox_xyxy_norm 无效: {identifier}")
        elif side == "none":
            raise ValueError(f"有密码机框时 target_side 不能为 none: {identifier}")
    counts = Counter(str(row["split"]) for row in actual.values())
    return {"frames": len(actual), "splits": {split: counts.get(split, 0) for split in ("train", "val")}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="导出 ACT 密码机视觉 grounding 人工标注池")
    parser.add_argument("dataset", type=Path, help="canonical MVP v5 数据目录，含 train/ 和 val/")
    parser.add_argument("--output", type=Path, default=None, help="新的、空的标注池目录")
    parser.add_argument("--camera-per-bucket", type=int, default=8,
                        help="每个 split/axis/非零 bucket 最多抽取的决策帧数")
    parser.add_argument("--q-radius-frames", type=int, default=6, help="每个 Q 前后导出的帧数")
    parser.add_argument("--validate-annotations", action="store_true",
                        help="仅校验已填写的 act_grounding_annotations.jsonl；dataset 参数仍作 pool 路径")
    args = parser.parse_args(argv)
    try:
        if args.validate_annotations:
            report = validate_annotations(args.dataset)
        else:
            if args.output is None:
                parser.error("导出标注池必须提供 --output")
            report = prepare(args.dataset, args.output, camera_per_bucket=args.camera_per_bucket,
                             q_radius_frames=args.q_radius_frames)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
