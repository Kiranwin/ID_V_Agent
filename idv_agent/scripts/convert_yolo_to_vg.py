"""Convert canonical YOLO labels into model-agnostic VG grounding JSONL.

YOLO txt remains the only human-maintained bounding-box source. This command
creates two reproducible products:

* ``vg_annotations.jsonl``: one structured annotation record per image;
* ``vg_grounding.jsonl``: question/answer examples rendered from those records.

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
    examples = [{
        "schema_version": SCHEMA_VERSION,
        "id": f"{annotation['id']}__state",
        "task": "state_qa",
        "image": image,
        "question": "当前帧中角色处于什么状态？",
        "answer": f"当前帧状态为 {annotation['state']}。",
        "target": None,
        "objects": objects,
        "state": annotation["state"],
        "intent_hint": annotation["intent_hint"],
        "source_session": annotation["source_session"],
        "source_frame": annotation["source_frame"],
        "split": annotation["split"],
    }]
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


def convert(dataset: Path, annotations_out: Path, grounding_out: Path,
            repo_root: Path, state_file: Path | None = None) -> tuple[int, int]:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"缺少 {manifest}")
    state_file = state_file or dataset / "frame_states.jsonl"
    states = _load_states(state_file)
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
                "intent_hint": STATE_TO_INTENT_HINT[state],
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
    args = parser.parse_args(argv)
    annotations_out = args.annotations_out or args.dataset / "vg_annotations.jsonl"
    grounding_out = args.grounding_out or args.dataset / "vg_grounding.jsonl"
    try:
        convert(args.dataset, annotations_out, grounding_out, args.repo_root.resolve(), args.state_file)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
