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
import math
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


def _yolo_objects(path: Path) -> list[tuple[int, list[float]]]:
    """Read canonical YOLO rows into normalized xyxy boxes without guessing."""
    if not path.is_file():
        raise ValueError(f"缺少 YOLO 标签文件: {path}")
    objects = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no} 必须是 5 列 YOLO 格式")
        try:
            class_id = int(fields[0])
            xc, yc, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} 含有不能解析的数字") from exc
        if class_id not in range(len(CLASSES)):
            raise ValueError(f"{path}:{line_no} class_id 不在 0..{len(CLASSES) - 1}")
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0
                   for value in (xc, yc, width, height)) or width <= 0 or height <= 0:
            raise ValueError(f"{path}:{line_no} YOLO 坐标无效")
        box = [max(0.0, xc - width / 2), max(0.0, yc - height / 2),
               min(1.0, xc + width / 2), min(1.0, yc + height / 2)]
        if not _valid_bbox(box):
            raise ValueError(f"{path}:{line_no} 裁剪后无有效框")
        objects.append((class_id, [round(value, 6) for value in box]))
    return objects


def _target_from_objects(objects: list[tuple[int, list[float]]]) -> tuple[list[float] | None, str, int, Counter]:
    """Derive one ACT target from human YOLO evidence.

    ``cipher_visible`` wins over highlight because it is the real machine
    geometry. Within a class, the largest visible component gives a stable
    target when the machine is split by the player/UI. A prompt proves that
    the interaction is currently reachable, even in the exceptional fully
    occluded frame where no machine box remains.
    """
    classes = Counter(class_id for class_id, _box in objects)
    candidates = [box for class_id, box in objects if class_id == 0]
    source_class = 0
    if not candidates:
        candidates = [box for class_id, box in objects if class_id == 1]
        source_class = 1
    if not candidates:
        return None, "none", int(classes.get(2, 0) > 0), classes
    box = max(candidates, key=lambda item: (item[2] - item[0]) * (item[3] - item[1]))
    center_x = (box[0] + box[2]) / 2
    side = "left" if center_x < 1 / 3 else "right" if center_x > 2 / 3 else "center"
    # Prompt is the only approved direct signal of immediate reachability;
    # visibility alone must not manufacture an interaction label.
    return box, side, int(classes.get(2, 0) > 0), classes


def populate_annotations_from_yolo(pool: Path) -> dict[str, Any]:
    """Replace only generated fields using completed YOLO labels.

    Missing X-AnyLabeling JSON has already been explicitly declared by the
    user as a completed no-cipher image. It becomes an empty YOLO txt after
    conversion and is therefore a deliberate negative, not an annotation gap.
    The prior sidecar is copied once before replacement and never overwritten.
    """
    pool = Path(pool)
    manifest_path = pool / "manifest.jsonl"
    annotations_path = pool / "act_grounding_annotations.jsonl"
    backup_path = pool / "act_grounding_annotations.before_yolo_auto.jsonl"
    if not manifest_path.is_file() or not annotations_path.is_file():
        raise ValueError("标注池缺少 manifest.jsonl 或 act_grounding_annotations.jsonl")
    if backup_path.exists():
        raise ValueError(f"已存在自动生成前备份，拒绝覆盖: {backup_path}")
    manifest = _json_records(manifest_path)
    annotations = _json_records(annotations_path)
    by_id = {str(row.get("id")): row for row in annotations}
    if len(by_id) != len(annotations) or set(by_id) != {str(row.get("id")) for row in manifest}:
        raise ValueError("辅助标注与 manifest 不一致；拒绝自动生成")
    generated = []
    class_totals = Counter()
    no_cipher = 0
    for item in manifest:
        identifier = str(item["id"])
        label_value = item.get("label")
        if not isinstance(label_value, str):
            raise ValueError(f"manifest id={identifier} 缺少 label")
        objects = _yolo_objects(pool / label_value)
        bbox, side, reachable, classes = _target_from_objects(objects)
        class_totals.update(classes)
        no_cipher += int(bbox is None)
        row = dict(by_id[identifier])
        row.update({
            "cipher_bbox_xyxy_norm": bbox,
            "cipher_reachable": reachable,
            "target_side": side,
            "interact_prompt": int(classes.get(2, 0) > 0),
            "annotation_source": "yolo_auto_from_completed_pool.v1",
        })
        generated.append(row)
    shutil.copy2(annotations_path, backup_path)
    annotations_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated), encoding="utf-8")
    return {
        "frames": len(generated),
        "cipher_visible": class_totals.get(0, 0),
        "cipher_highlight": class_totals.get(1, 0),
        "interact_prompt": class_totals.get(2, 0),
        "no_cipher": no_cipher,
    }


def validate_annotations(pool: Path) -> dict[str, Any]:
    """Fail closed until every exported frame has complete auxiliary labels.

    A no-cipher frame is represented by ``cipher_bbox_xyxy_norm=null`` and
    ``target_side=none``. ``cipher_reachable=1`` is valid with no box when a
    prompt is visible but the player has fully occluded the machine.
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
            if side != "none":
                raise ValueError(f"无密码机框时 target_side 必须为 none: {identifier}")
            if reachable == 1 and prompt != 1:
                raise ValueError(f"无密码机框却可达时必须有 interact_prompt=1: {identifier}")
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
    parser.add_argument("--populate-from-yolo", action="store_true",
                        help="将完成的 YOLO 标签自动生成辅助语义标注，并先创建一次不可覆盖备份")
    args = parser.parse_args(argv)
    try:
        if args.validate_annotations and args.populate_from_yolo:
            parser.error("--validate-annotations 与 --populate-from-yolo 只能二选一")
        if args.validate_annotations:
            report = validate_annotations(args.dataset)
        elif args.populate_from_yolo:
            report = populate_annotations_from_yolo(args.dataset)
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
