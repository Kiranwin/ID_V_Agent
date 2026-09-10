"""Prepare timestamped MVP annotation tasks without changing raw recordings.

This is an annotation/export contract, not a compatibility converter to v5.
Only reviewed rows are exportable; missing facts never become negatives.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from idv_agent.scripts.validate_vla_raw import validate as validate_raw
from idv_agent.vla.prompt_q import Q_CONTRACT, prompt_q_target

SCHEMA = "idv.mvp_annotation.v6"
TRAIN_SCHEMA = "idv.mvp_training.v6"
PHASES = ("search", "approach", "align", "interact", "maintain_decode")
PROTOCOL = {
    "id": "m28_sparse_visual_v6_prompt_q_v1", "capture_hz": 20,
    "vision_max_hz": 5, "history_max": 8, "history_min": 3,
    "nominal_visual_interval_ms": 200, "action_ms": 200,
    "nominal_observation_to_action_ms": 200,
    "timing_source": "design_assumption_requires_runtime_measurement",
    "camera_codebook": "legacy5_provisional", "top_intent": "decipher",
    "q_contract": Q_CONTRACT,
}
TRAIN_PROTOCOL = {
    "id": "m29_sgan_v6_training_v2",
    "capture_hz": 20,
    "vision_max_hz": 5,
    "history_max": 8,
    "history_min": 3,
    "action_delay_ms": 200,
    "action_duration_ms": 200,
    "observation_age_model_input": False,
    "phase_vocabulary": ["search", "approach", "align", "interact", "maintain_decode"],
    "phase_mapping": {},
    "q_contract": Q_CONTRACT,
}
CAMERA_COMMANDS = (-110, -25, 0, 25, 110)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_manifest(session: Path) -> dict:
    files = [session / name for name in
             ("meta.json", "events.csv", "mouse_deltas.csv", "frame_timestamps.csv")]
    files += sorted((session / "frames").glob("*.jpg"))
    return {path.relative_to(session).as_posix(): digest(path) for path in files}


def quantize(value: float, commands=CAMERA_COMMANDS) -> int:
    # Midpoint ties prefer the smaller correction, with a stable sign tie.
    return min(commands, key=lambda command: (abs(value - command), abs(command), command))


def replay_window(events: list[dict], mouse: list[dict], start: int, end: int) -> dict:
    """Integrate (start,end] events exactly; keyboard repeats are not edges."""
    keys: set[str] = set()
    changes = []
    q_edges = []
    duration = {name: 0 for name in ("key:w", "key:a", "key:s", "key:d")}
    previous = start
    for event in events:
        ts = int(event["timestamp_ns"])
        if ts > end:
            break
        code, kind = event["code"], event["kind"]
        if ts > start:
            for key in duration:
                duration[key] += max(0, ts - previous) * int(key in keys)
            previous = ts
        was_down = code in keys
        if kind in {"key_down", "mouse_down"}:
            keys.add(code)
        elif kind in {"key_up", "mouse_up"}:
            keys.discard(code)
        if start < ts <= end and (code in keys) != was_down:
            changes.append({"timestamp_ns": ts, "code": code, "down": code in keys})
            if code == "key:q" and code in keys:
                q_edges.append(ts)
    for key in duration:
        duration[key] += max(0, end - previous) * int(key in keys)
    selected = [row for row in mouse if start < int(row["timestamp_ns"]) <= end]
    dx = sum(int(row["dx"]) for row in selected)
    dy = sum(int(row["dy"]) for row in selected)
    occupancy = {key: round(value / (end - start), 6) for key, value in duration.items()}
    stable = all(value <= 0.2 or value >= 0.8 for value in occupancy.values())
    x = int(occupancy["key:d"] >= 0.8) - int(occupancy["key:a"] >= 0.8)
    y = int(occupancy["key:w"] >= 0.8) - int(occupancy["key:s"] >= 0.8)
    move_map = {(0, 0): 0, (0, 1): 1, (1, 1): 2, (1, 0): 3,
                (1, -1): 4, (0, -1): 5, (-1, -1): 6, (-1, 0): 7, (-1, 1): 8}
    opposing = (occupancy["key:w"] >= 0.8 and occupancy["key:s"] >= 0.8 or
                 occupancy["key:a"] >= 0.8 and occupancy["key:d"] >= 0.8)
    return {
        "start_ns_exclusive": start, "end_ns_inclusive": end,
        "key_occupancy": occupancy,
        "move_proposal": move_map[(x, y)] if stable and not opposing else None,
        "key_transitions": changes, "q_down_ns": q_edges,
        "mouse_dx_counts": dx, "mouse_dy_counts": dy,
        "mouse_abs_dx_counts": sum(abs(int(row["dx"])) for row in selected),
        "mouse_abs_dy_counts": sum(abs(int(row["dy"])) for row in selected),
        "quantized_command_proposal": [quantize(dx), quantize(dy)],
    }


def empty_labels() -> dict:
    return {
        "facts": {
            "scope": None, "cipher_visibility": None, "target_bbox": None,
            "target_track_id": None, "target_evidence": None,
            "interact_prompt": None, "prompt_bbox": None, "prompt_target_link": None,
            "decoding": None, "decoding_evidence": [],
            "path": None,
        },
        "decision": {
            "phase": None, "steering": None, "target_choice_reason": None,
            "track_valid_in_history": None,
            "reviewed_through_frame": None,
        },
        "action_review": {
            "source": None, "move": None, "camera_command": None, "q": None,
            "reason": None,
        },
        "outcome": {
            "status": None, "first_decode_frame": None, "confirmed_through_frame": None,
            "evidence": [], "target_progress": None,
        },
    }


def prepare(session: Path, output: Path) -> dict:
    session = session.resolve()
    errors = validate_raw(session)
    if errors:
        raise ValueError("; ".join(errors))
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    if float(meta.get("target_fps", 0)) != 20:
        raise ValueError("New MVP workspaces require target_fps=20; no frame duplication conversion")
    if output.exists():
        raise ValueError("Output already exists; preserve annotations and choose a new version")
    ft = read_csv(session / "frame_timestamps.csv")
    times = [int(row["timestamp_ns"]) for row in ft]
    events = read_csv(session / "events.csv")
    mouse = read_csv(session / "mouse_deltas.csv")
    anchors: dict[int, set[str]] = {}
    deadline = times[0]
    while deadline <= times[-1]:
        i = bisect.bisect_right(times, deadline) - 1
        if i >= 0:
            anchors.setdefault(i, set()).add("periodic_200ms")
        deadline += 200_000_000
    # Additional annotation views are review-only until the sampler allocates
    # their exposure. Q targets come from current-frame interact_prompt, not
    # from the demonstrated Q timing. Post-Q frames can still be Q positives.
    pressed: set[str] = set()
    q_edges = []
    for row in events:
        if row["kind"] == "key_down":
            if row["code"] == "key:q" and row["code"] not in pressed:
                q_edges.append(int(row["timestamp_ns"]))
            pressed.add(row["code"])
        elif row["kind"] == "key_up":
            pressed.discard(row["code"])
    for q in q_edges:
        for offset in (-400, -200, -50, 200, 1000):
            i = bisect.bisect_right(times, q + offset * 1_000_000) - 1
            if 0 <= i < len(times) and q + offset * 1_000_000 <= times[-1]:
                anchors.setdefault(i, set()).add(f"q_neighbour_{offset}ms")
    rows = []
    for index, reasons in sorted(anchors.items()):
        observed = times[index]
        history = sorted({bisect.bisect_right(times, observed - i * 200_000_000) - 1
                          for i in range(8)} - {-1})
        start, end = observed + 200_000_000, observed + 400_000_000
        after = bisect.bisect_left(times, end)
        issues = []
        if len(history) < 3:
            issues.append("insufficient_history")
        if any(times[right] - times[left] > 350_000_000 for left, right in zip(history, history[1:])):
            issues.append("visual_history_gap")
        if after >= len(times):
            issues.append("incomplete_action_window")
        if end + 1_000_000_000 > times[-1]:
            issues.append("insufficient_one_second_outcome_tail")
        row = {
            "schema": SCHEMA, "id": f"{session.name}:{index:08d}",
            "session_id": session.name, "frame": index, "observed_ns": observed,
            "history_frames": history,
            "history_timestamps_ns": [times[i] for i in history],
            "executed_context": {
                "last_q_down_ns": max((q for q in q_edges if q <= observed), default=None),
                "source": "human_events_not_model_prediction",
            },
            "sampling_reasons": sorted(reasons),
            "timing": {"action_start_ns": start, "action_end_ns": end,
                       "after_frame": after if after < len(times) else None},
            "replay": replay_window(events, mouse, start, end) if end <= times[-1] else None,
            "eligibility_issues": issues, "labels": empty_labels(),
            "review": {"status": "pending", "annotator": None, "reviewer": None,
                       "interact_prompt": {"status": "pending", "annotator": None, "reviewer": None}},
        }
        rows.append(row)
    output.mkdir(parents=True)
    manifest = {
        "schema": "idv.mvp_workspace.v6", "session_id": session.name,
        "raw_root": str(session), "raw_sha256": raw_manifest(session),
        "protocol": PROTOCOL, "split": "unassigned", "scenario_group": None,
        "status": "annotation_pending", "frames": len(ft),
        "post_q_tail_ms": [round((times[-1] - q) / 1e6, 3) for q in q_edges],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decisions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {"workspace": str(output.resolve()), "rows": len(rows), "pending": len(rows),
            "post_q_tail_ms": manifest["post_q_tail_ms"]}


def _box(value: Any, name: str) -> None:
    if value is None:
        return
    if (not isinstance(value, list) or len(value) != 4 or
            any(isinstance(x, bool) or not isinstance(x, (int, float)) or
                not math.isfinite(x) or not 0 <= x <= 1 for x in value) or
            not value[0] < value[2] or not value[1] < value[3]):
        raise ValueError(f"{name}: expected normalized xyxy bbox or null")


def validate_labels(row: dict) -> None:
    labels = row["labels"]
    facts, decision, action, outcome = (labels[key] for key in ("facts", "decision", "action_review", "outcome"))
    if facts["scope"] not in {"in_scope", "out_of_scope", "uncertain"}:
        raise ValueError("facts.scope is required")
    if facts["cipher_visibility"] not in {"visible", "highlight_only", "absent", "occluded", "unknown"}:
        raise ValueError("cipher_visibility is required")
    for field in ("interact_prompt", "decoding"):
        if facts[field] is not None and type(facts[field]) is not bool:
            raise ValueError(f"{field}: use true/false/null")
    _box(facts["target_bbox"], "target_bbox")
    _box(facts["prompt_bbox"], "prompt_bbox")
    if facts["cipher_visibility"] in {"visible", "highlight_only"} and facts["target_bbox"] is None:
        raise ValueError("visible/highlight target requires a box")
    if facts["cipher_visibility"] in {"absent", "occluded", "unknown"} and facts["target_bbox"] is not None:
        raise ValueError("unobserved target must not have an invented box")
    # The visible interaction UI alone establishes the Q token target. Its
    # box/target association can help grounding but is not a Q prerequisite.
    target = prompt_q_target(facts["interact_prompt"])
    if action["q"] is not None and (type(action["q"]) is not bool or action["q"] != target["q"]):
        raise ValueError("action q must equal interact_prompt; leave null for automatic derivation")
    if facts["decoding"] is True and not facts["decoding_evidence"]:
        raise ValueError("Q key alone is not decoding evidence")
    for evidence in facts["decoding_evidence"]:
        if isinstance(evidence, dict) and "frame" in evidence:
            frame = evidence["frame"]
            if type(frame) is not int or not 0 <= frame <= row["frame"]:
                raise ValueError("current decoding evidence cannot use future frames")
    if decision["phase"] not in (*PHASES, None):
        raise ValueError("invalid phase")
    if decision["steering"] not in {"target_center", "path_follow", "search_sweep", "hold", None}:
        raise ValueError("invalid steering")
    if facts["path"] not in {"direct", "detour_left", "detour_right", "blocked", "unknown", None}:
        raise ValueError("invalid path")
    if decision["phase"] == "maintain_decode" and facts["decoding"] is not True:
        raise ValueError("maintain_decode requires observed decoding")
    if decision["reviewed_through_frame"] != row["frame"]:
        raise ValueError("decision review must stop at observation endpoint, not a future Q frame")
    if action["source"] not in {"accept_replay", "correction", "exclude"}:
        raise ValueError("action source required; no automatic acceptance")
    if action["source"] != "exclude":
        if facts["scope"] != "in_scope" or decision["phase"] is None:
            raise ValueError("action targets require an in-scope reviewed phase")
        if any(issue in row["eligibility_issues"] for issue in
               ("insufficient_history", "incomplete_action_window", "visual_history_gap")):
            raise ValueError("action cannot train without history and complete action window")
        if type(action["move"]) is not int or not 0 <= action["move"] <= 8:
            raise ValueError("move must be 0..8")
        if not isinstance(action["camera_command"], list) or len(action["camera_command"]) != 2 or any(
                type(v) is not int or v not in CAMERA_COMMANDS for v in action["camera_command"]):
            raise ValueError("camera command must match versioned codebook")
        if action["source"] == "accept_replay":
            replay = row["replay"]
            if action["camera_command"] != replay["quantized_command_proposal"]:
                raise ValueError("accepted replay differs from extracted evidence; use correction with a reason")
            if action["move"] != replay["move_proposal"]:
                raise ValueError("mixed/changed movement is not accepted replay; correct or exclude")
        if action["source"] == "correction" and not action["reason"]:
            raise ValueError("correction needs a reason and remains unexecuted counterfactual")
    if outcome["status"] not in {"confirmed_decode", "no_entry", "insufficient_tail", "unknown", None}:
        raise ValueError("outcome status required")
    if outcome["status"] == "confirmed_decode":
        first, last = outcome["first_decode_frame"], outcome["confirmed_through_frame"]
        after = row["timing"]["after_frame"]
        if (type(first) is not int or type(last) is not int or after is None or
                first < after or last < first or not outcome["evidence"]):
            raise ValueError("decoding outcome needs post-action evidence and a confirmation interval")


def prompt_is_reviewed(row: dict) -> bool:
    review = row["review"]
    if review["status"] == "reviewed":
        if not review.get("annotator") or not review.get("reviewer"):
            raise ValueError("review provenance required")
        return True
    prompt_review = review.get("interact_prompt", {})
    if prompt_review.get("status", "pending") == "pending":
        return False
    if prompt_review.get("status") != "reviewed" or not prompt_review.get("annotator") or not prompt_review.get("reviewer"):
        raise ValueError("interact_prompt review provenance required")
    return True


def is_deprecated_review(row: dict) -> bool:
    """A human-marked deprecated decision endpoint never enters training."""
    action = row.get("labels", {}).get("action_review", {})
    review = row.get("review", {})
    text = " ".join(str(value or "") for value in (
        action.get("reason"), review.get("reason"), review.get("notes")))
    return "废弃" in text


def validate_workspace(workspace: Path, *, require_reviewed=False, prompt_only=False) -> dict:
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("protocol drift; create a versioned workspace")
    if raw_manifest(Path(manifest["raw_root"])) != manifest["raw_sha256"]:
        raise ValueError("raw evidence changed since preparation")
    times = [int(row["timestamp_ns"]) for row in read_csv(Path(manifest["raw_root"]) / "frame_timestamps.csv")]
    events = read_csv(Path(manifest["raw_root"]) / "events.csv")
    mouse = read_csv(Path(manifest["raw_root"]) / "mouse_deltas.csv")
    rows = [json.loads(line) for line in (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    seen = set()
    reviewed = 0
    prompt_reviewed = prompt_positive = prompt_negative = prompt_unknown = 0
    decoding_positive = decoding_negative = decoding_unknown = 0
    for row in rows:
        if row["schema"] != SCHEMA or row["id"] in seen or row["session_id"] != manifest["session_id"]:
            raise ValueError("schema/duplicate/session mismatch")
        seen.add(row["id"])
        if not 0 <= row["frame"] < len(times) or row["observed_ns"] != times[row["frame"]]:
            raise ValueError("observation timestamp mismatch")
        history = row["history_frames"]
        if (not history or history != sorted(set(history)) or len(history) > 8 or
                history[0] < 0 or history[-1] != row["frame"] or
                row["history_timestamps_ns"] != [times[i] for i in history]):
            raise ValueError("history must end at the observation and cannot contain future frames")
        start, end = row["observed_ns"] + 200_000_000, row["observed_ns"] + 400_000_000
        if row["timing"]["action_start_ns"] != start or row["timing"]["action_end_ns"] != end:
            raise ValueError("action clock mismatch")
        replay = replay_window(events, mouse, start, end) if end <= times[-1] else None
        if row["replay"] != replay:
            raise ValueError("replay evidence was edited; corrections belong in action_review")
        after = bisect.bisect_left(times, end)
        if row["timing"]["after_frame"] != (after if after < len(times) else None):
            raise ValueError("after_frame does not follow action end")
        expected_issues = []
        if len(history) < 3:
            expected_issues.append("insufficient_history")
        if any(times[right] - times[left] > 350_000_000 for left, right in zip(history, history[1:])):
            expected_issues.append("visual_history_gap")
        if after >= len(times):
            expected_issues.append("incomplete_action_window")
        if end + 1_000_000_000 > times[-1]:
            expected_issues.append("insufficient_one_second_outcome_tail")
        if row["eligibility_issues"] != expected_issues:
            raise ValueError("eligibility flags were changed without changing source evidence")
        pressed = set()
        last_q = None
        for event in events:
            ts = int(event["timestamp_ns"])
            if ts > row["observed_ns"]:
                break
            if event["kind"] == "key_down":
                if event["code"] == "key:q" and event["code"] not in pressed:
                    last_q = ts
                pressed.add(event["code"])
            elif event["kind"] == "key_up":
                pressed.discard(event["code"])
        if row["executed_context"] != {"last_q_down_ns": last_q, "source": "human_events_not_model_prediction"}:
            raise ValueError("executed Q context does not match past events")
        q_label = prompt_q_target(row["labels"]["facts"]["interact_prompt"])
        if prompt_is_reviewed(row):
            prompt_reviewed += 1
            prompt_positive += q_label["q"] is True
            prompt_negative += q_label["q"] is False
            prompt_unknown += q_label["q"] is None
        if prompt_only:
            continue
        if row["review"]["status"] == "reviewed":
            if not row["review"]["annotator"] or not row["review"]["reviewer"]:
                raise ValueError("review provenance required")
            validate_labels(row)
            decoding_positive += row["labels"]["facts"]["decoding"] is True
            decoding_negative += row["labels"]["facts"]["decoding"] is False
            decoding_unknown += row["labels"]["facts"]["decoding"] is None
            outcome = row["labels"]["outcome"]
            if outcome["status"] == "confirmed_decode":
                first, last = outcome["first_decode_frame"], outcome["confirmed_through_frame"]
                if last >= len(times) or times[last] - times[first] < 1_000_000_000:
                    raise ValueError("confirmed decoding requires >= 1 second of saved outcome frames")
            if outcome["status"] == "no_entry" and "insufficient_one_second_outcome_tail" in expected_issues:
                raise ValueError("no_entry cannot be inferred from a truncated recording")
            reviewed += 1
        elif row["review"]["status"] != "pending":
            raise ValueError("invalid review status")
    required_count = prompt_reviewed if prompt_only else reviewed
    if require_reviewed and required_count != len(rows):
        raise ValueError(f"annotation incomplete: {required_count}/{len(rows)} reviewed")
    return {"rows": len(rows), "reviewed": reviewed, "pending": len(rows)-reviewed,
            "raw_hashes_match": True, "prompt_reviewed": prompt_reviewed,
            "q_positive": prompt_positive, "q_negative": prompt_negative,
            "q_unknown": prompt_unknown,
            "decoding_positive": decoding_positive, "decoding_negative": decoding_negative,
            "decoding_unknown_reviewed": decoding_unknown}


def export(workspace: Path, output: Path) -> dict:
    report = validate_workspace(workspace, require_reviewed=True)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["split"] not in {"train", "val", "test"} or not manifest["scenario_group"]:
        raise ValueError("split and scenario_group must be assigned at session-group level")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing export")
    rows = []
    for line in (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if is_deprecated_review(row):
            continue
        labels = row["labels"]
        q_label = prompt_q_target(labels["facts"]["interact_prompt"])
        in_scope = labels["facts"]["scope"] == "in_scope"
        source_phase = labels["decision"]["phase"]
        training_phase = TRAIN_PROTOCOL["phase_mapping"].get(source_phase, source_phase)
        training_decision = {**labels["decision"], "phase": training_phase}
        action_target = {**labels["action_review"], "q": q_label["q"],
                         "q_source": Q_CONTRACT["target_source"]}
        history_ready = len(row["history_frames"]) >= PROTOCOL["history_min"] and "visual_history_gap" not in row["eligibility_issues"]
        rows.append({"schema": TRAIN_SCHEMA, "id": row["id"], "protocol": TRAIN_PROTOCOL,
                     "split": manifest["split"], "scenario_group": manifest["scenario_group"],
                     "model_input": {"raw_root": manifest["raw_root"],
                                     "history_frames": row["history_frames"],
                                     "history_timestamps_ns": row["history_timestamps_ns"],
                                     "observed_ns": row["observed_ns"], "task": "decipher",
                                     # Top-level feedback only; M29's Q head
                                     # has no access to this input.
                                     "last_executed_q_ns": row["executed_context"]["last_q_down_ns"]},
                     "targets": {"facts": labels["facts"], "decision": training_decision,
                                 "action": action_target},
                     "masks": {"navigation": in_scope and history_ready and labels["action_review"]["source"] != "exclude",
                               "q": q_label["mask"],
                               "phase": in_scope and training_phase is not None,
                               "interact_prompt": q_label["mask"],
                               "decoding": in_scope and labels["facts"]["decoding"] is not None,
                               "visibility": in_scope and labels["facts"]["cipher_visibility"] != "unknown",
                               "bbox": in_scope and labels["facts"]["target_bbox"] is not None,
                               "prompt_bbox": labels["facts"].get("prompt_bbox") is not None,
                               "steering": in_scope and labels["decision"]["steering"] is not None,
                               "path": in_scope and labels["facts"]["path"] not in (None, "unknown"),
                               "top_intent_ce": False},
                     "audit_only": {"replay": row["replay"], "outcome": labels["outcome"],
                                    "executed_context": row["executed_context"],
                                    "review": row["review"], "raw_sha256": manifest["raw_sha256"],
                                    "action_label_delay_ms": TRAIN_PROTOCOL["action_delay_ms"],
                                    "source_phase": source_phase,
                                    "training_phase": training_phase,
                                    "counterfactual": labels["action_review"]["source"] == "correction"}})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {**report, "output": str(output), "schema": TRAIN_SCHEMA,
            "legacy_train_vla_compatible": False}


def export_q(workspace: Path, output: Path) -> dict:
    """Export current images + Q/NO_Q without requiring navigation or futures."""
    report = validate_workspace(workspace, prompt_only=True)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["split"] not in {"train", "val", "test"} or not manifest["scenario_group"]:
        raise ValueError("split and scenario_group must be assigned at session-group level")
    if output.exists():
        raise ValueError("Refusing to overwrite an existing export")
    rows = []
    for line in (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if is_deprecated_review(row):
            continue
        target = prompt_q_target(row["labels"]["facts"]["interact_prompt"])
        if not prompt_is_reviewed(row) or not target["mask"]:
            continue
        relative_image = f"frames/{row['frame']:08d}.jpg"
        prompt_bbox = row["labels"]["facts"].get("prompt_bbox")
        rows.append({"schema": "idv.prompt_q_training.v2", "id": row["id"],
                     "q_contract": Q_CONTRACT, "session_id": row["session_id"],
                     "split": manifest["split"], "scenario_group": manifest["scenario_group"],
                     "model_input": {"image_path": str(Path(manifest["raw_root"]) / relative_image),
                                     "observed_ns": row["observed_ns"]},
                     "targets": {"interact_prompt": target["q"], "prompt_bbox": prompt_bbox,
                                 "q": target["q"], "token": target["token"]},
                     "masks": {"q": True, "prompt_bbox": prompt_bbox is not None},
                     "audit_only": {"image_sha256": manifest["raw_sha256"][relative_image],
                                    "review": row["review"]}})
    if not rows:
        raise ValueError("no reviewed known interact_prompt labels; pending/unknown are not negatives")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return {**report, "exported_q_rows": len(rows), "output": str(output),
            "schema": "idv.prompt_q_training.v2", "legacy_train_vla_compatible": False}


def migrate_prompt_q(workspace: Path, output: Path) -> dict:
    """Copy draft1 annotations into the explicit prompt-Q supervision version."""
    if output.exists():
        raise ValueError("Refusing to overwrite an existing workspace")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["protocol"].get("id") != "m28_sparse_visual_v6_draft1":
        raise ValueError("migration only accepts the earlier draft1 protocol")
    if raw_manifest(Path(manifest["raw_root"])) != manifest["raw_sha256"]:
        raise ValueError("raw evidence changed since preparation")
    rows = []
    for line in (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        facts = row["labels"]["facts"]
        if "interact_prompt" in facts:
            raise ValueError("ambiguous mixed prompt field versions")
        facts["interact_prompt"] = facts.pop("prompt")
        prompt_q_target(facts["interact_prompt"])
        row["migration_audit"] = {"old_action_q": row["labels"]["action_review"]["q"],
                                  "old_review": row["review"]}
        row["labels"]["action_review"]["q"] = None
        # Preserve authored facts but re-review the changed target contract.
        row["review"] = {"status": "pending", "annotator": None, "reviewer": None,
                         "interact_prompt": {"status": "pending", "annotator": None, "reviewer": None}}
        rows.append(row)
    manifest.update(protocol=PROTOCOL, status="annotation_pending",
                    migrated_from=str(workspace.resolve()))
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decisions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return validate_workspace(output)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("prepare")
    make.add_argument("session", type=Path)
    make.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("validate")
    check.add_argument("workspace", type=Path)
    check.add_argument("--require-reviewed", action="store_true")
    check.add_argument("--prompt-only", action="store_true")
    write = commands.add_parser("export")
    write.add_argument("workspace", type=Path)
    write.add_argument("--output", type=Path, required=True)
    q_write = commands.add_parser("export-q")
    q_write.add_argument("workspace", type=Path)
    q_write.add_argument("--output", type=Path, required=True)
    migrate = commands.add_parser("migrate-prompt-q")
    migrate.add_argument("workspace", type=Path)
    migrate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (prepare(args.session, args.output) if args.command == "prepare" else
              validate_workspace(args.workspace, require_reviewed=args.require_reviewed,
                                 prompt_only=args.prompt_only) if args.command == "validate" else
              export_q(args.workspace, args.output) if args.command == "export-q" else
              migrate_prompt_q(args.workspace, args.output) if args.command == "migrate-prompt-q" else
              export(args.workspace, args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
