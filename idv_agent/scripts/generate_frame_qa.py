"""Generate templated frame QA examples from VG annotations and states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"缺少 {path}")
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
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


def _load_states(path: Path) -> dict[tuple[str, int], str]:
    states = {}
    for row in _load_jsonl(path):
        try:
            key = (str(row["session"]), int(row["frame"]))
            state = str(row["state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path} 状态行必须含 session/frame/state") from exc
        if state not in {"decoding", "idle", "walking", "chased"}:
            raise ValueError(f"{path} 存在非法状态 {state!r}")
        if key in states:
            raise ValueError(f"{path} 重复状态 {key}")
        states[key] = state
    return states


def _load_intent_segments(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.is_file():
        return []
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 无效") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no} 必须是对象")
            rows.append(value)
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
        result.append({"start_frame": start, "end_frame": end, "intent": intent})
    result.sort(key=lambda row: (row["start_frame"], row["end_frame"]))
    for previous, current in zip(result, result[1:]):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError(f"{path} 片段重叠: {previous} / {current}")
    return result


def _segments_for_session(session: str, *, intent_segments: Path | None,
                          intent_root: Path | None,
                          cache: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if session in cache:
        return cache[session]
    if intent_segments is not None:
        cache[session] = _load_intent_segments(intent_segments)
    elif intent_root is not None:
        for name in ("intent_segments.csv", "intent_segments.jsonl"):
            path = intent_root / session / name
            if path.is_file():
                cache[session] = _load_intent_segments(path)
                break
        else:
            raise ValueError(f"session={session} 缺少 intent_segments.csv/jsonl: {intent_root / session}")
    else:
        cache[session] = []
    return cache[session]


def _load_intent(dataset: Path, meta_path: Path | None, override: str | None,
                 annotation: dict[str, Any], segments: list[dict[str, Any]]) -> tuple[str, str]:
    if override:
        return override, "cli"
    frame = annotation.get("source_frame")
    try:
        frame = int(frame)
    except (TypeError, ValueError):
        frame = None
    if frame is not None:
        for segment in segments:
            if segment["start_frame"] <= frame <= segment["end_frame"]:
                return segment["intent"], "intent_segments"
    candidates = []
    if meta_path and meta_path.is_file():
        candidates.append(meta_path)
    candidates.append(dataset / "meta.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} JSON 无效") from exc
        for key in ("intent", "task_intent", "primary_intent"):
            value = str(meta.get(key, "")).strip()
            if value:
                return value, f"meta:{key}"
    value = str(annotation.get("intent_hint", "")).strip()
    return value or "idle", str(annotation.get("intent_source", "annotation.intent_hint"))


def _region(objects: list[dict[str, Any]]) -> str:
    if not objects:
        return "未知位置"
    centers = [((obj["bbox_xyxy_norm"][0] + obj["bbox_xyxy_norm"][2]) / 2) for obj in objects]
    center = sum(centers) / len(centers)
    return "左侧" if center < 1 / 3 else "右侧" if center > 2 / 3 else "中央"


def _qa_for_annotation(annotation: dict[str, Any], state: str, intent: str,
                       intent_source: str) -> list[dict[str, Any]]:
    image = annotation["image"]
    objects = annotation.get("objects", [])
    ciphers = [obj for obj in objects if obj.get("label") == "cipher_machine"]
    highlights = [obj for obj in ciphers if obj.get("source_class") == "cipher_highlight"
                  or obj.get("visibility") == "highlighted"]
    location = _region(ciphers)
    common = {
        "schema_version": "vg.frame_qa.v1",
        "image": image,
        "source_session": annotation.get("source_session"),
        "source_frame": annotation.get("source_frame"),
        "split": annotation.get("split"),
        "state": state,
        "intent": intent,
        "intent_source": intent_source,
    }
    return [
        {**common, "id": f"{annotation['id']}__state", "task": "state_qa",
         "question": "当前处于什么状态？", "answer": state, "target": state},
        {**common, "id": f"{annotation['id']}__cipher_presence", "task": "presence_qa",
         "question": "画面中有密码机吗？",
         "answer": f"有，位于画面{location}。" if ciphers else "无。",
         "target": ciphers or None},
        {**common, "id": f"{annotation['id']}__cipher_highlight", "task": "attribute_qa",
         "question": "密码机高亮了吗？", "answer": "是" if highlights else "否",
         "target": highlights or None},
        {**common, "id": f"{annotation['id']}__intent", "task": "intent_qa",
         "question": "当前意图是什么？", "answer": intent, "target": intent},
    ]


def generate(dataset: Path, *, annotations: Path, states: Path,
             output: Path, meta: Path | None = None, intent: str | None = None,
             intent_segments: Path | None = None, intent_root: Path | None = None) -> int:
    state_map = _load_states(states)
    annotations_rows = _load_jsonl(annotations)
    segment_cache: dict[str, list[dict[str, Any]]] = {}
    result = []
    for annotation in annotations_rows:
        key = (str(annotation.get("source_session")), int(annotation.get("source_frame")))
        if key not in state_map:
            raise ValueError(f"缺少帧状态: {key}")
        session = str(annotation.get("source_session", ""))
        segments = _segments_for_session(session, intent_segments=intent_segments,
                                         intent_root=intent_root, cache=segment_cache)
        frame_intent, source = _load_intent(dataset, meta, intent, annotation, segments)
        result.extend(_qa_for_annotation(annotation, state_map[key], frame_intent, source))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in result) +
                      ("\n" if result else ""), encoding="utf-8")
    print(f"[frame-qa] annotations={len(annotations_rows)} examples={len(result)} output={output.resolve()}")
    return len(result)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VG annotations + frame states → frame QA JSONL")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--states", type=Path, default=None)
    parser.add_argument("--meta", type=Path, default=None)
    parser.add_argument("--intent-segments", type=Path, default=None,
                        help="按 source_frame 映射人工意图片段（CSV/JSONL）")
    parser.add_argument("--intent-root", type=Path, default=None,
                        help="包含各 session/intent_segments.csv 的 raw session 根目录")
    parser.add_argument("--intent", default=None, help="覆盖 meta 中的 session 主意图")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        generate(args.dataset,
                 annotations=args.annotations or args.dataset / "vg_annotations.jsonl",
                 states=args.states or args.dataset / "frame_states.jsonl",
                 output=args.output or args.dataset / "frame_qa.jsonl",
                 meta=args.meta, intent=args.intent,
                 intent_segments=args.intent_segments, intent_root=args.intent_root)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
