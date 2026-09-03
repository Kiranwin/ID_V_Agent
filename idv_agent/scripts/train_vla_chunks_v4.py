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
from idv_agent.vla.action_chunk import VLA_SCHEMA_VERSION_V4

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
                    raise ValueError(f"{path}:{line_no}: JSON 无效") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: 必须是对象")
                yield value


def audit(path: Path) -> dict:
    history_total = history_zero = 0
    conflicts = []
    intents = Counter()
    chunks = 0
    for row in _iter_jsonl(path):
        chunks += 1
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
        # Historical frames may legitimately straddle a segment boundary.
        # The model's slow target is the anchor/sample label, so only compare
        # the frame carrying the same anchor index; mixed history is reported
        # separately as expected boundary context, not as a contradiction.
        anchor = row.get("anchor_frame")
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
        "schema": "vla.train_chunks_audit.v1",
        "chunks": chunks,
        "intent_counts": dict(intents),
        "history_actions_zero_count": history_zero,
        "history_actions_total": history_total,
        "history_actions_zero_ratio": (history_zero / history_total if history_total else None),
        "slow_label_conflict_count": len(conflicts),
        "slow_label_conflicts": conflicts[:100],
        "model_input_whitelist": list(MODEL_INPUT_WHITELIST),
        "supervision_whitelist": list(SUPERVISION_WHITELIST),
        "gate_pass": not conflicts,
    }


def build_train(session: Path, output: Path, *, stride: int = 3, history: int = 8,
                slow_period_s: float = 1.0) -> dict:
    if history < 3 or history > 8:
        raise ValueError("v4 history 必须在 3..8")
    build(session, output, stride=stride, history=history,
          schema_version=VLA_SCHEMA_VERSION_V4, slow_period_s=slow_period_s)
    report = audit(output)
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
