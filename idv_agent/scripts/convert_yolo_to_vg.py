"""Convert canonical YOLO labels into model-agnostic VG grounding JSONL.

YOLO txt remains the only human-maintained bounding-box source. This command
creates two reproducible products:

* ``vg_annotations.jsonl``: one structured annotation record per image;
* ``vg_grounding.jsonl``: grounding/presence question/answer examples rendered from those records.

The structured boxes keep normalized and pixel ``xyxy`` coordinates so a later
Qwen/VLA adapter can choose its own textual box token format without relabeling.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from PIL import Image


CLASSES = ("cipher_visible", "cipher_highlight", "interact_prompt")
STATE_CHOICES = ("decoding", "idle", "walking", "chased")
STATE_TO_INTENT_HINT = {
    "decoding": "decipher",
    "idle": "idle",
    "walking": "travel",
    "chased": "kite",
}
SCHEMA_VERSION = "vg.grounding.v2"
_FRAME_RE = re.compile(r"^(?P<session>.+)_(?P<frame>\d+)$")


def _resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _parse_label(path: Path, width: int, height: int) -> list[dict]:
    objects = []
    if not path.is_file():
        raise ValueError(f"缺少标签文件: {path}")
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no} 必须是 5 列 YOLO 格式")
        try:
            class_id = int(parts[0])
            xc, yc, bw, bh = (float(x) for x in parts[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no} 坐标无法解析") from exc
        if not 0 <= class_id < len(CLASSES):
            raise ValueError(f"{path}:{line_no} 非法 class_id={class_id}")
        if not all(0.0 <= x <= 1.0 for x in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
            raise ValueError(f"{path}:{line_no} 坐标必须在 0..1 且宽高为正")
        x1, y1 = _clip01(xc - bw / 2), _clip01(yc - bh / 2)
        x2, y2 = _clip01(xc + bw / 2), _clip01(yc + bh / 2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{path}:{line_no} 框裁剪后无有效面积")
        source_class = CLASSES[class_id]
        if source_class == "cipher_visible":
            label, visibility = "cipher_machine", "visible"
        elif source_class == "cipher_highlight":
            label, visibility = "cipher_machine", "highlighted"
        else:
            label, visibility = "interact_prompt", "visible"
        objects.append({
            "label": label,
            "source_class": source_class,
            "class_id": class_id,
            "bbox_xyxy_norm": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
            "bbox_xyxy_px": [round(x1 * width, 2), round(y1 * height, 2),
                              round(x2 * width, 2), round(y2 * height, 2)],
            "visibility": visibility,
        })
    return objects


def _region(objects: list[dict]) -> str:
    if not objects:
        return "unknown"
    cx = sum((o["bbox_xyxy_norm"][0] + o["bbox_xyxy_norm"][2]) / 2 for o in objects) / len(objects)
    if cx < 1 / 3:
        return "left"
    if cx > 2 / 3:
        return "right"
    return "center"


def _box_text(box: list[float]) -> str:
    # Canonical, model-agnostic representation. Backbone-specific renderers may
    # later map this to Qwen special tokens or integer 0..1000 coordinates.
    return "[" + ", ".join(f"{x:.4f}" for x in box) + "]"


def _examples(annotation: dict) -> list[dict]:
    image = annotation["image"]
    objects = annotation["objects"]
    # Frame state is supervised separately in frame_qa.jsonl. Keeping it out
    # of vg_grounding avoids duplicate state_qa samples and keeps this file's
    # contract limited to spatial grounding/presence.
    examples = []
    for obj in objects:
        if obj["label"] == "cipher_machine":
            question = "密码机在哪里？"
            answer = f"密码机位于画面{annotation['screen_region']}，框坐标为 {_box_text(obj['bbox_xyxy_norm'])}。"
        elif obj["label"] == "interact_prompt":
            question = "扳手交互提示在哪里？当前可以按 Q 吗？"
            answer = f"扳手交互提示位于画面{annotation['screen_region']}；可以按 Q。框坐标为 {_box_text(obj['bbox_xyxy_norm'])}。"
        else:
            question = "扳手交互提示在哪里？当前可以按 Q 吗？"
            answer = f"扳手交互提示位于画面{annotation['screen_region']}；可以按 Q。框坐标为 {_box_text(obj['bbox_xyxy_norm'])}。"
        examples.append({
            "schema_version": SCHEMA_VERSION,
            "id": f"{annotation['id']}__{obj['class_id']}",
            "task": "grounding_qa",
            "image": image,
            "question": question,
            "answer": answer,
            "target": obj,
            "objects": objects,
            "state": annotation["state"],
            "intent_hint": annotation["intent_hint"],
            "intent_source": annotation.get("intent_source", "unknown"),
            "source_session": annotation["source_session"],
            "source_frame": annotation["source_frame"],
            "split": annotation["split"],
        })
    if not objects:
        examples.append({
            "schema_version": SCHEMA_VERSION,
            "id": f"{annotation['id']}__negative",
            "task": "presence_qa",
            "image": image,
            "question": "画面中是否有可确认的密码机或交互提示？",
            "answer": "没有可确认的密码机或交互提示。",
            "target": None,
            "objects": [],
            "state": annotation["state"],
            "intent_hint": annotation["intent_hint"],
            "intent_source": annotation.get("intent_source", "unknown"),
            "source_session": annotation["source_session"],
            "source_frame": annotation["source_frame"],
            "split": annotation["split"],
        })
    return examples


def _load_states(path: Path) -> dict[tuple[str, int], str]:
    if not path.is_file():
        raise ValueError(f"缺少帧状态文件: {path}；请先填写 state=decoding/idle/walking/chased")
    states = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            item = json.loads(raw)
            session, frame, state = item["session"], int(item["frame"]), item["state"]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_no} 状态行必须含 session/frame/state") from exc
        if state not in STATE_CHOICES:
            raise ValueError(f"{path}:{line_no} state 必须是 {STATE_CHOICES}，当前为 {state!r}")
        key = (str(session), frame)
        if key in states:
            raise ValueError(f"{path}:{line_no} 重复状态 {key}")
        states[key] = state
    return states


def _load_intent_segments(path: Path | None) -> list[dict]:
    if path is None or not path.is_file():
        return []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    result = []
    for index, row in enumerate(rows):
        try:
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            intent = str(row["intent"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{index + 1} 缺少有效 start_frame/end_frame/intent") from exc
        if start < 0 or end < start or not intent:
            raise ValueError(f"{path}:{index + 1} 片段范围或 intent 无效")
        result.append({"start_frame": start, "end_frame": end,
                       "intent": intent, "segment_id": str(row.get("segment_id") or f"seg_{index:03d}")})
    result.sort(key=lambda row: (row["start_frame"], row["end_frame"]))
    for previous, current in zip(result, result[1:]):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError(f"{path} 片段重叠: {previous} / {current}")
    return result


def _intent_for_frame(frame: int, segments: list[dict], state: str) -> tuple[str, str]:
    for segment in segments:
        if segment["start_frame"] <= frame <= segment["end_frame"]:
            return segment["intent"], "intent_segments"
    return STATE_TO_INTENT_HINT[state], "state_weak_hint"


def _segments_for_session(session: str, *, intent_segments: Path | None,
                          intent_root: Path | None) -> list[dict]:
    if intent_segments is not None:
        return _load_intent_segments(intent_segments)
    if intent_root is None:
        return []
    for name in ("intent_segments.csv", "intent_segments.jsonl"):
        path = intent_root / session / name
        if path.is_file():
            return _load_intent_segments(path)
    raise ValueError(f"session={session} 缺少 intent_segments.csv/jsonl: {intent_root / session}")


def convert(dataset: Path, annotations_out: Path, grounding_out: Path,
            repo_root: Path, state_file: Path | None = None,
            intent_segments: Path | None = None,
            intent_root: Path | None = None,
            require_intent_coverage: bool = False) -> tuple[int, int]:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"缺少 {manifest}")
    state_file = state_file or dataset / "frame_states.jsonl"
    states = _load_states(state_file)
    segment_cache: dict[str, list[dict]] = {}
    annotations, examples = [], []
    expected_state_keys: set[tuple[str, int]] = set()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_path = _resolve(repo_root, row["image"])
            label_path = _resolve(repo_root, row["label"])
            if not image_path.is_file():
                raise ValueError(f"缺少图片: {image_path}")
            with Image.open(image_path) as image:
                width, height = image.size
            match = _FRAME_RE.match(image_path.stem)
            if not match:
                raise ValueError(f"图片名无法解析 session/frame: {image_path.name}")
            objects = _parse_label(label_path, width, height)
            session = row["session"]
            frame_index = int(match.group("frame"))
            expected_state_keys.add((session, frame_index))
            state = states.get((session, frame_index))
            if state is None:
                raise ValueError(f"缺少帧状态: session={session} frame={frame_index}")
            if session not in segment_cache:
                segment_cache[session] = _segments_for_session(
                    session, intent_segments=intent_segments, intent_root=intent_root)
            segments = segment_cache[session]
            if require_intent_coverage and not any(
                    segment["start_frame"] <= frame_index <= segment["end_frame"]
                    for segment in segments):
                raise ValueError(f"session={session} frame={frame_index} 不在人工 intent_segments 覆盖范围内")
            intent_hint, intent_source = _intent_for_frame(frame_index, segments, state)
            record = {
                "schema_version": SCHEMA_VERSION,
                "id": image_path.stem,
                "task": "vg_annotation",
                "image": image_path.relative_to(repo_root).as_posix(),
                "width": width,
                "height": height,
                "source_session": session,
                "source_frame": frame_index,
                "split": row["split"],
                "state": state,
                "intent_hint": intent_hint,
                "intent_source": intent_source,
                "screen_region": _region([o for o in objects if o["label"] == "cipher_machine"]),
                "objects": objects,
                "annotation_source": "yolo_txt",
            }
            annotations.append(record)
            examples.extend(_examples(record))
    missing = expected_state_keys - set(states)
    extra = set(states) - expected_state_keys
    if missing:
        preview = sorted(missing)[:5]
        raise ValueError(f"缺少帧状态 {len(missing)} 条，例如 {preview}")
    if extra:
        preview = sorted(extra)[:5]
        raise ValueError(f"帧状态存在未在 manifest 中的条目 {len(extra)} 条，例如 {preview}")
    annotations_out.parent.mkdir(parents=True, exist_ok=True)
    grounding_out.parent.mkdir(parents=True, exist_ok=True)
    annotations_out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in annotations) + "\n", encoding="utf-8")
    grounding_out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in examples) + "\n", encoding="utf-8")
    print(f"[vg] annotations={len(annotations)} grounding_examples={len(examples)}")
    print(f"[vg] annotations_out={annotations_out.resolve()}")
    print(f"[vg] grounding_out={grounding_out.resolve()}")
    return len(annotations), len(examples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="YOLO 标注 → VG grounding JSONL")
    parser.add_argument("dataset", type=Path, help="包含 manifest.csv/images/labels 的 YOLO 数据集目录")
    parser.add_argument("--annotations-out", type=Path, default=None)
    parser.add_argument("--grounding-out", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--state-file", type=Path, default=None,
                        help="帧级状态 JSONL；默认 dataset/frame_states.jsonl")
    parser.add_argument("--intent-segments", type=Path, default=None,
                        help="人工意图片段 CSV/JSONL；用于统一 vg_annotations.intent_hint")
    parser.add_argument("--intent-root", type=Path, default=None,
                        help="包含各 session/intent_segments.csv 的 raw session 根目录")
    parser.add_argument("--allow-weak-intent", action="store_true",
                        help="允许分段未覆盖的帧回退到状态弱提示（默认严格报错）")
    args = parser.parse_args(argv)
    annotations_out = args.annotations_out or args.dataset / "vg_annotations.jsonl"
    grounding_out = args.grounding_out or args.dataset / "vg_grounding.jsonl"
    try:
        if args.intent_segments is not None and args.intent_root is not None:
            parser.error("--intent-segments 与 --intent-root 只能二选一")
        convert(args.dataset, annotations_out, grounding_out, args.repo_root.resolve(), args.state_file,
                args.intent_segments, args.intent_root,
                require_intent_coverage=bool(args.intent_segments or args.intent_root) and not args.allow_weak_intent)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
