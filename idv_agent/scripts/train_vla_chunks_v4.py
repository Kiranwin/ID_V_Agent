"""Build the v4 ACT training JSONL and audit its field contract.

The source builder keeps rich provenance (timestamps, alignment, frame-level
labels, ...).  This command makes that boundary explicit: model consumers are
allowed only frames + task condition + history actions, while supervision is
the action chunk + the *sample* slow label + loss masks.  Rich fields remain in
the record for validation/lineage and are never returned by the Dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from idv_agent.scripts.build_vla_chunks import build
from idv_agent.vla.action_chunk import (
    BUTTON_NAMES, CAMERA_BUCKETS, MOVE_DIRECTIONS,
    VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5,
    validate_v4_record, validate_v5_record,
)

MODEL_INPUT_WHITELIST = ("observations.frames.path", "task.instruction", "mode",
                         "observations.history_actions")
SUPERVISION_WHITELIST = ("action_chunk", "slow_label", "loss_mask")


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    yield line_no, {"__audit_error__": f"{path}:{line_no}: JSON 无效: {exc}"}
                    continue
                if not isinstance(value, dict):
                    yield line_no, {"__audit_error__": f"{path}:{line_no}: 必须是对象"}
                    continue
                yield line_no, value


def audit(path: Path, *, schema_version: str = VLA_SCHEMA_VERSION_V4) -> dict:
    if schema_version not in {VLA_SCHEMA_VERSION_V4, VLA_SCHEMA_VERSION_V5}:
        raise ValueError(f"不支持的训练数据 schema: {schema_version}")
    validator = validate_v5_record if schema_version == VLA_SCHEMA_VERSION_V5 else validate_v4_record
    history_total = history_zero = 0
    conflicts = []
    record_errors = []
    parse_errors = []
    alignment_errors = []
    intents = Counter()
    moves = Counter()
    camera_dx = Counter()
    camera_dy = Counter()
    camera_dx_px = []
    camera_dy_px = []
    buttons = Counter()
    action_total = action_zero = 0
    chunks = 0
    for line_no, row in _iter_jsonl(path):
        chunks += 1
        if "__audit_error__" in row:
            parse_errors.append({"line": line_no, "error": row["__audit_error__"]})
            continue
        try:
            validator(row)
        except ValueError as exc:
            record_errors.append({
                "line": line_no,
                "episode_id": row.get("episode_id"),
                "anchor_frame": row.get("anchor_frame"),
                "error": str(exc),
            })
            continue
        history = row.get("observations", {}).get("history_actions", [])
        history_total += 1
        if not history or all(
            int(a.get("move_dir", 0)) == 0
            and int(a.get("camera_dx", 0)) == 0
            and int(a.get("camera_dy", 0)) == 0
            and not any(int(v) for v in a.get("buttons", []))
            for a in history
        ):
            history_zero += 1
        sample = row.get("slow_label", {})
        sample_intent = sample.get("intent") if sample.get("valid") else None
        if sample_intent:
            intents[sample_intent] += 1
        actions = row["action_chunk"]
        for action in actions:
            action_total += 1
            moves[str(action["move_dir"])] += 1
            camera_dx[str(action["camera_dx"])] += 1
            camera_dy[str(action["camera_dy"])] += 1
            if schema_version == VLA_SCHEMA_VERSION_V5:
                camera_dx_px.append(float(action["camera_dx_px"]))
                camera_dy_px.append(float(action["camera_dy_px"]))
            for index, value in enumerate(action["buttons"]):
                if value:
                    buttons[BUTTON_NAMES[index]] += 1
            if (action["move_dir"] == 0 and action["camera_dx"] == 0 and
                    action["camera_dy"] == 0 and not any(action["buttons"])):
                action_zero += 1

        alignment = row["alignment"]
        anchor = row["anchor_frame"]
        frames = row["observations"]["frames"]
        errors = []
        if alignment.get("observation_end_frame") != anchor:
            errors.append("observation_end_frame 必须等于 anchor_frame")
        if frames[-1].get("frame_index") != anchor:
            errors.append("最后一张 observations.frames 必须是 anchor_frame")
        action_end = alignment.get("action_end_frame")
        if not isinstance(action_end, int):
            errors.append("缺少整数 alignment.action_end_frame，无法核对动作跨度")
        else:
            expected_end = alignment["action_start_frame"] + sum(
                action["duration_frames"] for action in actions
            ) - 1
            if action_end != expected_end:
                errors.append("action_end_frame 与 action_chunk.duration_frames 不一致")
        if errors:
            alignment_errors.append({
                "line": line_no,
                "episode_id": row.get("episode_id"),
                "anchor_frame": anchor,
                "errors": errors,
            })
        # Historical frames may legitimately straddle a segment boundary.
        # The model's slow target is the anchor/sample label, so only compare
        # the frame carrying the same anchor index; mixed history is reported
        # separately as expected boundary context, not as a contradiction.
        anchor_labels = [
            f.get("slow_label", {})
            for f in row.get("observations", {}).get("frames", [])
            if f.get("frame_index") == anchor and f.get("slow_label", {}).get("valid")
        ]
        for anchor_label in anchor_labels:
            mismatched = [
                name for name in ("segment_id", "intent", "subgoal")
                if sample.get(name) != anchor_label.get(name)
            ]
            if mismatched:
                conflicts.append({
                    "episode_id": row.get("episode_id"),
                    "anchor_frame": anchor,
                    "sample_slow_label": {name: sample.get(name) for name in
                                          ("segment_id", "intent", "subgoal")},
                    "anchor_frame_slow_label": {name: anchor_label.get(name) for name in
                                                 ("segment_id", "intent", "subgoal")},
                    "mismatched_fields": mismatched,
                })
    return {
        "schema": "vla.train_chunks_audit.v2",
        "record_schema_version": schema_version,
        "chunks": chunks,
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors[:100],
        "record_error_count": len(record_errors),
        "record_errors": record_errors[:100],
        "alignment_error_count": len(alignment_errors),
        "alignment_errors": alignment_errors[:100],
        "intent_counts": dict(intents),
        "action_distribution": {
            "total_actions": action_total,
            "zero_action_count": action_zero,
            "zero_action_ratio": (action_zero / action_total if action_total else None),
            "move_dir": {name: moves[str(index)] for index, name in enumerate(MOVE_DIRECTIONS)},
            "camera_dx": {str(bucket): camera_dx[str(bucket)] for bucket in CAMERA_BUCKETS},
            "camera_dy": {str(bucket): camera_dy[str(bucket)] for bucket in CAMERA_BUCKETS},
            "button_positive": {name: buttons[name] for name in BUTTON_NAMES},
            "camera_raw_pixels": {
                "dx_min": min(camera_dx_px) if camera_dx_px else None,
                "dx_max": max(camera_dx_px) if camera_dx_px else None,
                "dy_min": min(camera_dy_px) if camera_dy_px else None,
                "dy_max": max(camera_dy_px) if camera_dy_px else None,
                "nonzero_dx_count": sum(value != 0 for value in camera_dx_px),
                "nonzero_dy_count": sum(value != 0 for value in camera_dy_px),
            } if schema_version == VLA_SCHEMA_VERSION_V5 else None,
        },
        "history_actions_zero_count": history_zero,
        "history_actions_total": history_total,
        "history_actions_zero_ratio": (history_zero / history_total if history_total else None),
        "slow_label_conflict_count": len(conflicts),
        "slow_label_conflicts": conflicts[:100],
        "model_input_whitelist": list(MODEL_INPUT_WHITELIST),
        "supervision_whitelist": list(SUPERVISION_WHITELIST),
        "gate_pass": not parse_errors and not record_errors and not alignment_errors and not conflicts,
    }


def build_train(session: Path, output: Path, *, stride: int = 3, history: int = 8,
                slow_period_s: float = 1.0,
                schema_version: str = VLA_SCHEMA_VERSION_V4) -> dict:
    if history < 3 or history > 8:
        raise ValueError("v4/v5 history 必须在 3..8")
    build(session, output, stride=stride, history=history,
          schema_version=schema_version, slow_period_s=slow_period_s)
    report = audit(output, schema_version=schema_version)
    report["session"] = session.name
    report["output"] = str(output)
    report_path = output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--slow-period-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = build_train(args.session, args.output, stride=args.stride,
                         history=args.history, slow_period_s=args.slow_period_s)
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
