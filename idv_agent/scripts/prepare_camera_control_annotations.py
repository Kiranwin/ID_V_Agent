"""Prepare and validate human camera-control annotations for ambiguous ACT frames.

This tool never modifies canonical v5 chunks, raw sessions, or existing
grounding labels.  It selects only same-frame ``interact_prompt=0`` decisions
already present in the ACT grounding pool, writes a recoverable JSONL template,
and validates a completed copy before it can be consumed by a later trainer.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from idv_agent.scripts.train_vla import _dataset_paths
from idv_agent.training.vla_dataset import GroundingAnnotationIndex, VLASequenceDataset
from idv_agent.vla.action_chunk import CAMERA_BUCKETS


ANNOTATION_SCHEMA = "act.camera_control_annotation.v1"
CONTROL_PHASES = ("search", "target_acquire", "target_align", "hold")
TARGET_IDS = ("target_cipher", "other_visible", "none")


def _template_row(*, record: dict[str, Any], source_root: Path,
                  grounding: dict[str, Any], split: str) -> dict[str, Any]:
    frame = int(record["alignment"]["observation_end_frame"])
    episode = str(record["episode_id"])
    bbox = grounding["cipher_bbox_xyxy_norm"]
    action = record["action_chunk"][0]
    return {
        "schema_version": ANNOTATION_SCHEMA,
        "id": f"{episode}_{frame:08d}",
        "split": split,
        "session": episode,
        "frame": frame,
        "image_path": str(source_root / next(
            item["path"] for item in record["observations"]["frames"]
            if int(item["frame_index"]) == frame)),
        "existing_grounding": {
            "cipher_bbox_xyxy_norm": bbox,
            "target_side": grounding["target_side"],
            "interact_prompt": int(grounding["interact_prompt"]),
            "cipher_reachable": int(grounding["cipher_reachable"]),
        },
        # Preserve replay as evidence; annotators must not overwrite it.
        "replay_camera_dx": int(action["camera_dx"]),
        "replay_camera_dy": int(action["camera_dy"]),
        "camera_control_phase": None,
        "camera_target_id": None,
        "desired_turn_dx": None,
        "desired_turn_dy": None,
        "status": "pending",
    }


def prepare(*, data: str | list[str], grounding_annotations: str | Path,
            split: str, output: str | Path) -> dict[str, Any]:
    """Write one pending annotation row per ambiguous same-frame decision."""
    labels = GroundingAnnotationIndex(grounding_annotations)
    dataset = VLASequenceDataset(_dataset_paths(data), verify_images=True)
    rows = []
    for root, record in dataset.records:
        source_root = Path(record.get("source_root", root))
        frame = int(record["alignment"]["observation_end_frame"])
        row = labels.rows.get((str(record["episode_id"]), frame))
        if row is None:
            continue
        # Convert retained tensors once for user-facing immutable context.
        bbox = row["grounding_bbox_target"].tolist() if bool(row["grounding_bbox_mask"]) else None
        side = ("none", "left", "center", "right")[int(row["grounding_side_target"])]
        readable = {
            "cipher_bbox_xyxy_norm": bbox,
            "target_side": side,
            "interact_prompt": int(row["grounding_prompt_target"].item()),
            "cipher_reachable": int(row["grounding_reachable_target"].item()),
        }
        if readable["interact_prompt"]:
            continue
        rows.append(_template_row(record=record, source_root=source_root,
                                  grounding=readable, split=split))
    if not rows:
        raise ValueError("未找到 interact_prompt=0 的同帧 grounding 决策")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"schema": ANNOTATION_SCHEMA, "output": str(output.resolve()), "rows": len(rows),
            "split": split, "replay_dx": dict(Counter(row["replay_camera_dx"] for row in rows)),
            "replay_dy": dict(Counter(row["replay_camera_dy"] for row in rows))}


def validate(path: str | Path, *, require_complete: bool = True) -> dict[str, Any]:
    """Validate a template or completed annotation file without modifying it."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"camera-control 标注不存在: {path}")
    seen: set[str] = set()
    complete = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            identifier = str(row["id"])
            phase, target, status = row["camera_control_phase"], row["camera_target_id"], row["status"]
            dx, dy = row["desired_turn_dx"], row["desired_turn_dy"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no} camera-control 字段无效") from exc
        if row.get("schema_version") != ANNOTATION_SCHEMA or not identifier or identifier in seen:
            raise ValueError(f"{path}:{line_no} schema/id 无效或重复")
        seen.add(identifier)
        if status == "pending":
            if require_complete:
                raise ValueError(f"{path}:{line_no} 仍为 pending")
            continue
        if status != "complete" or phase not in CONTROL_PHASES or target not in TARGET_IDS:
            raise ValueError(f"{path}:{line_no} phase/target/status 无效")
        if dx not in CAMERA_BUCKETS or dy not in CAMERA_BUCKETS:
            raise ValueError(f"{path}:{line_no} desired_turn_dx/dy 必须是 -2..2")
        if phase == "hold" and (dx != 0 or dy != 0):
            raise ValueError(f"{path}:{line_no} hold 必须标记 desired_turn=(0,0)")
        if target == "none" and phase == "target_align":
            raise ValueError(f"{path}:{line_no} target_align 需要可识别的控制目标")
        complete += 1
    if not seen:
        raise ValueError("camera-control 标注为空")
    return {"schema": ANNOTATION_SCHEMA, "rows": len(seen), "complete": complete,
            "pending": len(seen) - complete, "require_complete": require_complete}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("prepare")
    make.add_argument("--data", required=True)
    make.add_argument("--grounding-annotations", required=True)
    make.add_argument("--split", choices=("train", "val"), required=True)
    make.add_argument("--output", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--input", required=True)
    check.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args(argv)
    result = (prepare(data=args.data, grounding_annotations=args.grounding_annotations,
                      split=args.split, output=args.output)
              if args.command == "prepare" else
              validate(args.input, require_complete=not args.allow_pending))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
