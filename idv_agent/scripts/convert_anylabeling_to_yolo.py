"""Convert X-AnyLabeling/LabelMe sidecar JSON into YOLO labels.

X-AnyLabeling writes annotations next to labeled images, but it does not write
an empty JSON for negative images.  This converter treats a missing sidecar as
an empty label file, while reporting the count so an incomplete annotation run
is visible.  It never deletes the source JSON files.

Example:
    python -m idv_agent.scripts.convert_anylabeling_to_yolo \
        data/vg_cipher_20260901_105143 --split train
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


CLASSES = ("cipher_visible", "cipher_highlight", "interact_prompt")
CLASS_TO_ID = {name: index for index, name in enumerate(CLASSES)}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _number(value: Any, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path} 必须是数字") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path} 必须是有限数字")
    return result


def _bbox_from_shape(shape: dict, path: str, width: int, height: int) -> tuple[float, float, float, float]:
    if shape.get("shape_type") != "rectangle":
        raise ValueError(f"{path}.shape_type={shape.get('shape_type')!r}；当前只支持 rectangle")
    points = shape.get("points")
    if not isinstance(points, list) or len(points) < 2:
        raise ValueError(f"{path}.points 必须至少含两个点")
    coordinates = []
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"{path}.points[{index}] 必须是 [x, y]")
        coordinates.append((_number(point[0], f"{path}.points[{index}][0]"),
                           _number(point[1], f"{path}.points[{index}][1]")))
    x1 = max(0.0, min(width, min(point[0] for point in coordinates)))
    y1 = max(0.0, min(height, min(point[1] for point in coordinates)))
    x2 = max(0.0, min(width, max(point[0] for point in coordinates)))
    y2 = max(0.0, min(height, max(point[1] for point in coordinates)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{path}.points 裁剪到图像范围后无有效面积")
    return x1, y1, x2, y2


def _shape_to_yolo(shape: dict, path: str, width: int, height: int) -> str:
    label = shape.get("label")
    if label not in CLASS_TO_ID:
        raise ValueError(f"{path}.label={label!r} 不在 {CLASSES}")
    x1, y1, x2, y2 = _bbox_from_shape(shape, path, width, height)
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{CLASS_TO_ID[label]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def _convert_json(json_path: Path, image_path: Path) -> tuple[list[str], Counter[str]]:
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{json_path} JSON 无效: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{json_path} 顶层必须是对象")
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except OSError as exc:
        raise ValueError(f"无法读取图片 {image_path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"图片尺寸无效: {image_path}={width}x{height}")
    declared_width = payload.get("imageWidth")
    declared_height = payload.get("imageHeight")
    if declared_width is not None and int(declared_width) != width:
        raise ValueError(f"{json_path}.imageWidth={declared_width} 与图片宽度 {width} 不一致")
    if declared_height is not None and int(declared_height) != height:
        raise ValueError(f"{json_path}.imageHeight={declared_height} 与图片高度 {height} 不一致")
    shapes = payload.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"{json_path}.shapes 必须是数组")
    lines = []
    labels = Counter()
    for index, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            raise ValueError(f"{json_path}.shapes[{index}] 必须是对象")
        label = shape.get("label")
        lines.append(_shape_to_yolo(shape, f"{json_path}.shapes[{index}]", width, height))
        labels[str(label)] += 1
    return lines, labels


def _image_files(dataset: Path, split: str | None) -> list[tuple[str, Path]]:
    image_root = dataset / "images"
    if not image_root.is_dir():
        raise ValueError(f"缺少图片目录: {image_root}")
    splits = [split] if split else sorted(path.name for path in image_root.iterdir() if path.is_dir())
    if not splits:
        raise ValueError(f"没有找到 images/<split> 目录: {image_root}")
    result = []
    for current_split in splits:
        directory = image_root / current_split
        if not directory.is_dir():
            raise ValueError(f"缺少 split 图片目录: {directory}")
        result.extend((current_split, path) for path in sorted(directory.iterdir())
                      if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not result:
        raise ValueError("没有找到可转换的图片")
    return result


def convert(dataset: Path, *, split: str | None = None, overwrite: bool = False,
            dry_run: bool = False) -> dict[str, Any]:
    """Convert all sidecars under a standard YOLO dataset directory.

    Missing JSON sidecars intentionally produce empty txt files.  All source
    files are validated before any output is written.
    """
    dataset = Path(dataset)
    image_files = _image_files(dataset, split)
    pending: list[tuple[Path, str]] = []
    json_count = 0
    missing_json_count = 0
    boxes = Counter()
    for current_split, image_path in image_files:
        json_path = image_path.with_suffix(".json")
        label_path = dataset / "labels" / current_split / f"{image_path.stem}.txt"
        if json_path.is_file():
            lines, current_boxes = _convert_json(json_path, image_path)
            json_count += 1
            boxes.update(current_boxes)
        else:
            lines = []
            missing_json_count += 1
        if label_path.is_file() and label_path.stat().st_size > 0 and not overwrite:
            raise ValueError(f"已有非空标签，拒绝覆盖: {label_path}；需要覆盖时加 --overwrite")
        pending.append((label_path, "\n".join(lines) + ("\n" if lines else "")))

    if not dry_run:
        for label_path, content in pending:
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(content, encoding="utf-8")
    summary = {
        "images": len(image_files),
        "json": json_count,
        "missing_json_as_empty": missing_json_count,
        "boxes": sum(boxes.values()),
        "labels": dict(boxes),
        "written": not dry_run,
    }
    print(f"[x-anylabeling] images={summary['images']} json={json_count} "
          f"missing_json_as_empty={missing_json_count} boxes={summary['boxes']}")
    print("[x-anylabeling] boxes=" + ", ".join(f"{name}={boxes.get(name, 0)}" for name in CLASSES))
    if missing_json_count:
        print("[x-anylabeling] 注意：没有 JSON 的图片已按空标签处理；请确认这些图片确实没有目标")
    print(f"[x-anylabeling] {'检查完成，未写文件' if dry_run else 'YOLO 标签已写入'}: {dataset / 'labels'}")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="X-AnyLabeling JSON → YOLO txt")
    parser.add_argument("dataset", type=Path, help="包含 images/<split> 的 YOLO 数据集目录")
    parser.add_argument("--split", choices=("train", "val", "test"), default=None,
                        help="只转换一个 split；默认转换所有 images/<split>")
    parser.add_argument("--overwrite", action="store_true",
                        help="允许覆盖已有非空 YOLO 标签")
    parser.add_argument("--dry-run", action="store_true", help="只校验和统计，不写标签")
    args = parser.parse_args(argv)
    try:
        convert(args.dataset, split=args.split, overwrite=args.overwrite, dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
