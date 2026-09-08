"""Fail-closed readiness audit for the canonical MVP ACT training set.

This is intentionally an audit, not a builder: raw recordings, per-frame
actions, v5 chunks, annotation sidecars, and class-balance artifacts remain
immutable inputs.  It proves that every training-only label attaches to the
same causal observation endpoint as its ACT record and cannot cross the
session-held-out train/validation split.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from idv_agent.scripts.prepare_camera_control_annotations import validate as validate_control
from idv_agent.scripts.train_vla import _dataset_paths
from idv_agent.scripts.validate_vla_raw import validate as validate_raw
from idv_agent.training.vla_dataset import GroundingAnnotationIndex
from idv_agent.vla.action_chunk import (
    CAMERA_BUCKET_COMMAND_PX,
    CAMERA_BUCKET_EDGES_PX,
    CAMERA_BUCKETS,
    VLA_SCHEMA_VERSION_V5,
    validate_v5_record,
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
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


def _dataset_records(dataset: Path) -> tuple[dict[str, dict[tuple[str, int], dict[str, Any]]], list[str]]:
    records: dict[str, dict[tuple[str, int], dict[str, Any]]] = {"train": {}, "val": {}}
    errors: list[str] = []
    for split in records:
        for path in sorted((dataset / split).glob("*.jsonl")):
            for line_no, row in enumerate(_jsonl(path), 1):
                try:
                    validate_v5_record(row)
                    key = (str(row["episode_id"]), int(row["alignment"]["observation_end_frame"]))
                    if key in records[split]:
                        raise ValueError("同 split 重复 observation endpoint")
                    source = Path(str(row.get("source_root", "")))
                    if not source.is_dir():
                        raise ValueError("source_root 不存在")
                    if source.name != key[0]:
                        raise ValueError("source_root 名称与 episode_id 不一致")
                    records[split][key] = row
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append(f"{path}:{line_no}: {exc}")
    for split, values in records.items():
        if not values:
            errors.append(f"{dataset / split}: 没有有效 v5 记录")
    overlap = set(records["train"]) & set(records["val"])
    if overlap:
        errors.append(f"train/val observation endpoint 重叠: {sorted(overlap)[:3]}")
    sessions = [{episode for episode, _ in rows} for rows in records.values()]
    if sessions[0] & sessions[1]:
        errors.append(f"train/val session 重叠: {sorted(sessions[0] & sessions[1])[:3]}")
    return records, errors


def _grounding_rows(path: Path) -> list[dict[str, Any]]:
    # The Dataset index validates bbox/prompt/reachability invariants.  Keep
    # raw rows too because split membership is an audit property, not a model
    # tensor property.
    GroundingAnnotationIndex(path)
    return _jsonl(path)


def _check_grounding(rows: list[dict[str, Any]], records: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> list[str]:
    errors: list[str] = []
    seen: set[tuple[str, int]] = set()
    for line_no, row in enumerate(rows, 1):
        try:
            split = str(row["split"])
            key = (str(row["session"]), int(row["frame"]))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"grounding:{line_no}: split/session/frame 无效: {exc}")
            continue
        # The pool deliberately includes Q-neighbourhood frames in addition to
        # ACT decision frames.  Non-endpoint rows are valid visual-label
        # coverage; the Dataset simply masks them for this ACT run.  Their
        # session still must belong to the declared held-out split.
        split_sessions = {episode for episode, _ in records.get(split, {})}
        if split not in records or key[0] not in split_sessions:
            errors.append(f"grounding:{line_no}: session={key[0]} 不属于 {split} canonical split")
        if key in seen:
            errors.append(f"grounding:{line_no}: 重复 decision frame {key}")
        seen.add(key)
    return errors


def _check_control(path: Path, split: str, records: dict[str, dict[tuple[str, int], dict[str, Any]]]) -> tuple[list[str], Counter]:
    validate_control(path, require_complete=True)
    errors: list[str] = []
    desired_dx: Counter = Counter()
    seen: set[tuple[str, int]] = set()
    for line_no, row in enumerate(_jsonl(path), 1):
        try:
            if row.get("split") != split:
                raise ValueError(f"标注 split={row.get('split')!r}，期望 {split!r}")
            key = (str(row["session"]), int(row["frame"]))
            canonical = records[split][key]
            h0 = canonical["action_chunk"][0]
            if int(row["replay_camera_dx"]) != int(h0["camera_dx"]):
                raise ValueError("replay_camera_dx 与 canonical h0 camera_dx 不一致")
            if int(row["replay_camera_dy"]) != int(h0["camera_dy"]):
                raise ValueError("replay_camera_dy 与 canonical h0 camera_dy 不一致")
            desired_dx[int(row["desired_turn_dx"])] += 1
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path.name}:{line_no}: {exc}")
            continue
        if key in seen:
            errors.append(f"{path.name}:{line_no}: 重复 decision frame {key}")
        seen.add(key)
    return errors, desired_dx


def _class_balance_check(path: Path, train_records: dict[tuple[str, int], dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, int]]]:
    if not path.is_file():
        return [f"缺少 class balance artifact: {path}"], {}
    artifact = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if artifact.get("method") != "inverse_sqrt_clamped_mean1" or artifact.get("execution_horizon") != 1:
        errors.append("class balance 必须是 execution_horizon=1 的 bounded inverse-sqrt")
    if artifact.get("clamp") != [0.35, 3.0] or artifact.get("normalized_per_head") is not True:
        errors.append("class balance clamp/按头归一化 与训练协议不一致")
    if artifact.get("camera_bucket_edges_px") != list(CAMERA_BUCKET_EDGES_PX):
        errors.append("class balance 相机桶边界与代码不一致")
    if artifact.get("camera_bucket_command_px") != {str(key): value for key, value in CAMERA_BUCKET_COMMAND_PX.items()}:
        errors.append("class balance 相机执行代表值与代码不一致")
    observed = {"move": Counter(), "camera_dx": Counter(), "camera_dy": Counter(), "intent": Counter()}
    for row in train_records.values():
        h0 = row["action_chunk"][0]
        observed["move"][int(h0["move_dir"])] += 1
        observed["camera_dx"][int(h0["camera_dx"])] += 1
        observed["camera_dy"][int(h0["camera_dy"])] += 1
        label = row.get("slow_label", {})
        if label.get("valid"):
            observed["intent"][str(label.get("intent"))] += 1
    heads = artifact.get("heads", {})
    expected_orders = {"move": list(range(9)), "camera_dx": list(CAMERA_BUCKETS),
                       "camera_dy": list(CAMERA_BUCKETS)}
    for name, order in expected_orders.items():
        counts = heads.get(name, {}).get("counts")
        expected = [float(observed[name][value]) for value in order]
        if counts != expected:
            errors.append(f"class balance {name} h0 统计与 canonical train 不一致")
    return errors, {name: {str(key): value for key, value in sorted(counter.items())}
                    for name, counter in observed.items()}


def audit(*, dataset: str | Path, grounding_annotations: str | Path,
          control_train: str | Path, control_val: str | Path,
          class_balance: str | Path, validate_raw_sessions: bool = False) -> dict[str, Any]:
    dataset = Path(dataset)
    records, errors = _dataset_records(dataset)
    grounding = _grounding_rows(Path(grounding_annotations))
    errors.extend(_check_grounding(grounding, records))
    train_errors, train_desired = _check_control(Path(control_train), "train", records)
    val_errors, val_desired = _check_control(Path(control_val), "val", records)
    errors.extend(train_errors)
    errors.extend(val_errors)
    balance_errors, h0_counts = _class_balance_check(Path(class_balance), records["train"])
    errors.extend(balance_errors)
    raw_errors: dict[str, list[str]] = {}
    if validate_raw_sessions:
        sources = {Path(str(row["source_root"])) for split in records.values() for row in split.values()}
        for source in sorted(sources):
            result = validate_raw(source)
            if result:
                raw_errors[source.name] = result
        if raw_errors:
            errors.append(f"raw session 校验失败: {list(raw_errors)[:3]}")
    return {
        "schema": "mvp.training_data_audit.v1",
        "dataset": str(dataset.resolve()),
        "record_schema": VLA_SCHEMA_VERSION_V5,
        "camera_bucket_edges_px": list(CAMERA_BUCKET_EDGES_PX),
        "camera_bucket_command_px": {str(key): value for key, value in CAMERA_BUCKET_COMMAND_PX.items()},
        "splits": {name: {"chunks": len(rows), "sessions": len({key[0] for key in rows})}
                   for name, rows in records.items()},
        "grounding": {"rows": len(grounding), "by_split": dict(Counter(str(row.get("split")) for row in grounding))},
        "camera_control": {"train_rows": sum(train_desired.values()), "val_rows": sum(val_desired.values()),
                           "train_desired_turn_dx": dict(sorted(train_desired.items())),
                           "val_desired_turn_dx": dict(sorted(val_desired.items()))},
        "h0_counts": h0_counts,
        "raw_validation": {"enabled": validate_raw_sessions, "failures": raw_errors},
        "errors": errors[:200],
        "error_count": len(errors),
        "gate_pass": not errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="canonical MVP v5 dataset root")
    parser.add_argument("--grounding-annotations", required=True)
    parser.add_argument("--camera-control-train", required=True)
    parser.add_argument("--camera-control-val", required=True)
    parser.add_argument("--class-balance", required=True)
    parser.add_argument("--validate-raw-sessions", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit(dataset=args.dataset, grounding_annotations=args.grounding_annotations,
                   control_train=args.camera_control_train, control_val=args.camera_control_val,
                   class_balance=args.class_balance, validate_raw_sessions=args.validate_raw_sessions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
