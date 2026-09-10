"""Import completed X-AnyLabeling frames into the v6 MVP annotation workflow.

All image sidecars must exist, including explicit empty annotations. No
missing-file-to-negative conversion. Raw recordings and old workspaces remain
untouched. Navigation decisions require a separate, reviewable CSV pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image

from idv_agent.scripts.prepare_mvp_v6 import (
    digest, export, export_q, prompt_q_target, Q_CONTRACT, read_csv,
    validate_workspace,
)

CLASSES = ("cipher_visible", "cipher_highlight", "interact_prompt")
REVIEW_FIELDS = (
    "id", "frame", "image_path", "interact_prompt", "q_target",
    "candidate_summary", "phase_proposal", "move_proposal", "camera_proposal",
    "eligibility_issues", "scope", "phase", "target_candidate", "target_track_id",
    "steering", "path", "action_source", "move", "camera_dx", "camera_dy",
    "reason", "reviewed",
    "decoding", "decoding_evidence_frames",
)
NAVIGATION_BLOCKING_ISSUES = frozenset(("insufficient_history", "incomplete_action_window", "visual_history_gap"))


def normalize_review_edit(row: dict, edit: dict) -> dict:
    """Separate state/intent review from unusable replay-action supervision."""
    normalized = dict(edit)
    facts = row.get("labels", {}).get("facts", {})
    decision = row.get("labels", {}).get("decision", {})
    # V6 is an MVP workflow: start from the in-scope, replay-accepted path
    # and only ask the annotator to override an actual exception.
    if not normalized.get("scope"):
        normalized["scope"] = "in_scope"
    if not normalized.get("phase") and decision.get("phase"):
        normalized["phase"] = decision["phase"]
    normalized["decoding"] = "true" if normalized.get("phase") == "maintain_decode" else "false"
    if not normalized.get("steering"):
        normalized["steering"] = "search_sweep" if normalized.get("phase") == "search" else "hold"
    if not normalized.get("path"):
        normalized["path"] = "direct"
    if not normalized.get("action_source"):
        normalized["action_source"] = "accept_replay"
    if not normalized.get("target_candidate"):
        candidates = [shape for shape in row.get("visual_import", {}).get("detections", [])
                      if shape.get("label") != "interact_prompt"]
        if len(candidates) == 1:
            normalized["target_candidate"] = candidates[0].get("candidate", "")
    # A visible interaction prompt means the agent is already close enough;
    # this endpoint supervises interact/Q and never requires navigation target
    # association. Candidate/track labels are for pre-prompt navigation only.
    if row["labels"]["facts"].get("interact_prompt") is True:
        normalized["action_source"] = "exclude"
    issues = NAVIGATION_BLOCKING_ISSUES.intersection(row.get("eligibility_issues", ()))
    if issues and normalized.get("action_source") in {"", "accept_replay", "correction"}:
        normalized["action_source"] = "exclude"
        if not normalized.get("reason"):
            normalized["reason"] = "自动排除导航监督：" + ", ".join(sorted(issues)) + "；保留 Q 与顶层状态"
    return normalized


def ensure_session_review_csv(workspace: Path) -> Path:
    """One editable CSV per raw session; imported workspaces keep snapshots."""
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    session = Path(manifest["raw_root"]).resolve()
    if manifest.get("session_id") != session.name:
        raise ValueError("review workspace/session identity mismatch")
    destination = session / "navigation_review_with_state.csv"
    if destination.exists():
        return destination
    source = workspace / "navigation_review_with_state.csv"
    if not source.is_file():
        source = workspace / "navigation_review.csv"
    if not source.is_file():
        raise FileNotFoundError("review template missing; prepare/import the session first")
    # Copy, never move or replace; old workspaces remain recoverable snapshots.
    try:
        with destination.open("xb") as handle:
            handle.write(source.read_bytes())
    except FileExistsError:
        pass
    return destination


def read_shapes(path: Path, image_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("shapes"), list):
        raise ValueError(f"{path}: expected shapes array")
    with Image.open(image_path) as image:
        width, height = image.size
        image.verify()
    if payload.get("imageWidth") != width or payload.get("imageHeight") != height:
        raise ValueError(f"{path}: image dimension mismatch")
    if payload.get("imagePath") and Path(payload["imagePath"]).name != image_path.name:
        raise ValueError(f"{path}: imagePath does not match frame")
    boxes = []
    for index, shape in enumerate(payload["shapes"]):
        if shape.get("label") not in CLASSES or shape.get("shape_type") != "rectangle":
            raise ValueError(f"{path}: unsupported label or shape at {index}")
        points = shape.get("points")
        if not isinstance(points, list) or len(points) not in (2, 4):
            raise ValueError(f"{path}: rectangle needs 2 or 4 points")
        for point in points:
            if not isinstance(point, list) or len(point) != 2 or any(
                    type(v) not in (int, float) or not math.isfinite(v) for v in point):
                raise ValueError(f"{path}: invalid point")
            if not 0 <= point[0] <= width or not 0 <= point[1] <= height:
                raise ValueError(f"{path}: rectangle outside image")
        x1, x2 = min(p[0] for p in points), max(p[0] for p in points)
        y1, y2 = min(p[1] for p in points), max(p[1] for p in points)
        if x1 >= x2 or y1 >= y2:
            raise ValueError(f"{path}: zero-area rectangle")
        boxes.append({"candidate": f"shape_{index}", "label": shape["label"],
                      "bbox": [x1/width, y1/height, x2/width, y2/height],
                      "group_id": shape.get("group_id"), "score": shape.get("score")})
    return payload, boxes


def import_completed(workspace: Path, output: Path, *, annotator: str,
                     completion_note: str, split: str, scenario_group: str) -> dict:
    validate_workspace(workspace)
    if output.exists():
        raise ValueError("output exists; choose a new version instead of overwriting annotations")
    if not annotator.strip() or not completion_note.strip() or not scenario_group.strip():
        raise ValueError("annotator, completion confirmation and scenario_group required")
    if split not in {"train", "val", "test"}:
        raise ValueError("invalid split")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    raw = Path(manifest["raw_root"])
    all_facts, snapshots = {}, {}
    shape_counts = Counter()
    q_rows = []
    provenance = {"status": "reviewed", "annotator": annotator,
                  "reviewer": "user_completion_confirmation", "completion_note": completion_note,
                  "validation": "schema_bounds_dimensions_hashes_only"}
    for frame_row in read_csv(raw / "frame_timestamps.csv"):
        frame = int(frame_row["frame_id"])
        image_path = raw / "frames" / f"{frame:08d}.jpg"
        sidecar = image_path.with_suffix(".json")
        if not sidecar.is_file():
            raise ValueError(f"{sidecar}: missing is unannotated, not NO_Q; save an explicit empty JSON")
        payload, shapes = read_shapes(sidecar, image_path)
        snapshots[sidecar.name] = sidecar.read_bytes()
        shape_counts.update(shape["label"] for shape in shapes)
        prompt = any(shape["label"] == "interact_prompt" for shape in shapes)
        target = prompt_q_target(prompt)
        prompt_boxes = [shape["bbox"] for shape in shapes if shape["label"] == "interact_prompt"]
        prompt_bbox = prompt_boxes[0] if len(prompt_boxes) == 1 else None
        all_facts[frame] = {"frame": frame, "image_path": str(image_path),
                            "observed_ns": int(frame_row["timestamp_ns"]),
                            "interact_prompt": prompt, "detections": shapes,
                            "annotation_sha256": digest(sidecar),
                            "source_checked": payload.get("checked"),
                            "annotation_source": str(sidecar), "review": provenance}
        q_rows.append({"schema": "idv.prompt_q_training.v2", "id": f"{manifest['session_id']}:{frame:08d}",
                       "q_contract": Q_CONTRACT, "session_id": manifest["session_id"],
                       "split": split, "scenario_group": scenario_group,
                       "model_input": {"image_path": str(image_path), "observed_ns": int(frame_row["timestamp_ns"])},
                       "targets": {"interact_prompt": prompt, "prompt_bbox": prompt_bbox,
                                   "q": prompt, "token": target["token"]},
                       "masks": {"q": True, "prompt_bbox": prompt_bbox is not None},
                       "audit_only": {"image_sha256": manifest["raw_sha256"][f"frames/{frame:08d}.jpg"],
                                      "annotation_sha256": digest(sidecar), "review": provenance}})
    rows, review_rows = [], []
    for line in (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        fact = all_facts[row["frame"]]
        candidates = [shape for shape in fact["detections"] if shape["label"] != "interact_prompt"]
        prompts = [shape for shape in fact["detections"] if shape["label"] == "interact_prompt"]
        row["visual_import"] = fact
        row["review"]["interact_prompt"] = provenance
        row["labels"]["facts"].update(interact_prompt=fact["interact_prompt"],
                                           prompt_bbox=prompts[0]["bbox"] if len(prompts) == 1 else None)
        # Preserve every candidate. Never choose the largest object as a
        # hidden goal-selection rule when several machines were annotated.
        row["labels"]["facts"].update(cipher_visibility="unknown", target_bbox=None)
        row["labels"]["action_review"]["q"] = None
        replay = row["replay"]
        review_row = dict.fromkeys(REVIEW_FIELDS, "")
        review_row.update(id=row["id"], frame=row["frame"], image_path=fact["image_path"],
                          interact_prompt=int(fact["interact_prompt"]), q_target=int(fact["interact_prompt"]),
                          candidate_summary=json.dumps(candidates, ensure_ascii=False),
                          phase_proposal="interact" if fact["interact_prompt"] else "",
                          move_proposal=replay["move_proposal"] if replay else "",
                          camera_proposal=json.dumps(replay["quantized_command_proposal"]) if replay else "",
                          eligibility_issues="|".join(row["eligibility_issues"]), reviewed="pending")
        rows.append(row)
        review_rows.append(review_row)
    manifest.update(split=split, scenario_group=scenario_group,
                    imported_from=str(workspace.resolve()), status="visual_complete_navigation_pending",
                    anylabeling={"annotator": annotator, "completion_note": completion_note,
                                 "json_hashes": {name: hashlib.sha256(data).hexdigest()
                                                 for name, data in snapshots.items()}})
    positives = sum(row["targets"]["q"] for row in q_rows)
    report = {"session": manifest["session_id"], "frames": len(q_rows), "json": len(snapshots),
              "shape_counts": dict(shape_counts), "q_positive": positives, "q_negative": len(q_rows)-positives,
              "decision_endpoints": len(rows), "navigation_review_pending": len(rows),
              "multi_candidate_frames": [frame for frame, fact in all_facts.items()
                                         if sum(shape["label"] != "interact_prompt" for shape in fact["detections"]) > 1],
              "split": split, "scenario_group": scenario_group,
              "validation": "source_schema_and_user_completion_confirmation; not independent visual accuracy audit"}
    output.mkdir(parents=True)
    (output / "source_annotations").mkdir()
    for name, data in snapshots.items():
        (output / "source_annotations" / name).write_bytes(data)
    for name, value in (("manifest.json", manifest), ("visual_import_report.json", report)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, values in (("decisions.jsonl", rows), ("visual_frames.jsonl", list(all_facts.values())),
                         ("prompt_q_all_frames.jsonl", q_rows)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")
    with (output / "navigation_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(review_rows)
    (output / "navigation_review_with_state.csv").write_bytes((output / "navigation_review.csv").read_bytes())
    validate_workspace(output)
    export_q(output, output / "prompt_q_endpoints.jsonl")
    ensure_session_review_csv(output)
    return report


def apply_review_row(row: dict, edit: dict, *, reviewer: str,
                     source_csv: str = "", source_csv_sha256: str = "") -> dict:
    """Return a validated copy; used by GUI completion and final export."""
    from copy import deepcopy
    from idv_agent.scripts.prepare_mvp_v6 import validate_labels
    if not reviewer.strip():
        raise ValueError("reviewer is required")
    edit = normalize_review_edit(row, edit)
    row = deepcopy(row)
    if int(edit["frame"]) != row["frame"] or int(edit["q_target"]) != int(row["labels"]["facts"]["interact_prompt"]):
        raise ValueError("read-only frame/Q columns were changed")
    labels = row["labels"]
    labels["facts"]["scope"] = edit["scope"]
    # Current decoding is an observed state, not an outcome inferred
    # from Q or from future frames. Old CSVs leave this field unknown.
    decoding_value = edit.get("decoding", "").strip().lower()
    if decoding_value not in {"", "unknown", "true", "false"}:
        raise ValueError("decoding must be true/false/unknown")
    evidence_text = edit.get("decoding_evidence_frames", "").strip()
    evidence_frames = json.loads(evidence_text) if evidence_text else []
    if not isinstance(evidence_frames, list) or any(
            type(frame) is not int or frame < 0 or frame > row["frame"]
            for frame in evidence_frames):
        raise ValueError("decoding evidence must reference current/past frame IDs, never future frames")
    if decoding_value in {"true", "false"}:
        labels["facts"].update(decoding=decoding_value == "true",
                               decoding_evidence=[{"frame": frame, "source": "human_current_state_review"}
                                                  for frame in evidence_frames])
    elif "decoding" in edit:
        labels["facts"].update(decoding=None, decoding_evidence=[])
    source = edit["action_source"]
    selected = edit["target_candidate"]
    if selected in {"none", "unknown", "occluded"}:
        labels["facts"].update(cipher_visibility={"none": "absent", "unknown": "unknown", "occluded": "occluded"}[selected],
                               target_bbox=None, target_evidence=selected)
    else:
        candidate = next((shape for shape in row["visual_import"]["detections"]
                          if shape["candidate"] == selected and shape["label"] != "interact_prompt"), None)
        if candidate is None:
            raise ValueError(f"{row['id']}: select a cipher candidate or none/unknown/occluded")
        labels["facts"].update(cipher_visibility="visible" if candidate["label"] == "cipher_visible" else "highlight_only",
                               target_bbox=candidate["bbox"], target_evidence=candidate["label"])
    labels["facts"].update(target_track_id=edit["target_track_id"] or None, path=edit["path"] or None,
                           prompt_target_link=None)
    labels["decision"].update(phase=edit["phase"] or None, steering=edit["steering"] or None,
                              reviewed_through_frame=row["frame"])
    if source == "accept_replay":
        if row["replay"] is None:
            raise ValueError("no complete replay action window; exclude navigation (Q stays supervised)")
        move, camera = row["replay"]["move_proposal"], row["replay"]["quantized_command_proposal"]
    elif source == "correction":
        move, camera = int(edit["move"]), [int(edit["camera_dx"]), int(edit["camera_dy"])]
    elif source == "exclude":
        move, camera = None, None
    else:
        raise ValueError("action_source must be accept_replay/correction/exclude")
    labels["action_review"].update(source=source, move=move, camera_command=camera, q=None, reason=edit["reason"] or None)
    # The CSV does not assert decoding outcomes. Existing optional audits
    # remain intact, and their labels still pass the usual validator.
    row["review"].update(status="reviewed", annotator=reviewer, reviewer=reviewer,
                         source_csv=source_csv, source_csv_sha256=source_csv_sha256)
    validate_labels(row)
    return row


def apply_navigation(workspace: Path, review_csv: Path, output: Path, *, reviewer: str) -> dict:
    """Apply explicit human navigation decisions into a new workspace copy."""
    validate_workspace(workspace)
    if output.exists() or not reviewer.strip():
        raise ValueError("choose a new output and provide reviewer identity")
    reviews = read_csv(review_csv)
    if len({row["id"] for row in reviews}) != len(reviews):
        raise ValueError("duplicate navigation review ID")
    by_id = {row["id"]: row for row in reviews}
    rows = [json.loads(line) for line in (workspace / "decisions.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if set(by_id) != {row["id"] for row in rows}:
        raise ValueError("review CSV must cover exactly the decision endpoint IDs")
    reviewed = 0
    for row in rows:
        edit = by_id[row["id"]]
        if edit["reviewed"] == "pending":
            continue
        if edit["reviewed"] != "reviewed":
            raise ValueError("reviewed must be pending/reviewed")
        updated = apply_review_row(row, edit, reviewer=reviewer,
                                   source_csv=str(review_csv.resolve()), source_csv_sha256=digest(review_csv))
        row.update(updated)
        reviewed += 1
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    manifest.update(navigation_review_source=str(review_csv.resolve()), navigation_review_sha256=digest(review_csv),
                    imported_from=str(workspace.resolve()))
    # Validate all edited labels before creating any output.
    from idv_agent.scripts.prepare_mvp_v6 import validate_labels
    for row in rows:
        if row["review"]["status"] == "reviewed":
            validate_labels(row)
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "decisions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    (output / "navigation_review.csv").write_bytes(review_csv.read_bytes())
    result = validate_workspace(output)
    export_q(output, output / "prompt_q_endpoints.jsonl")
    if reviewed == len(rows):
        export(output, output / "training_v6.jsonl")
    return {**result, "full_export_written": reviewed == len(rows)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("import")
    make.add_argument("workspace", type=Path)
    make.add_argument("--output", type=Path, required=True)
    make.add_argument("--annotator", required=True)
    make.add_argument("--completion-note", required=True)
    make.add_argument("--split", choices=("train", "val", "test"), required=True)
    make.add_argument("--scenario-group", required=True)
    review = sub.add_parser("review")
    review.add_argument("workspace", type=Path)
    review.add_argument("--csv", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--reviewer", required=True)
    args = parser.parse_args(argv)
    result = (import_completed(args.workspace, args.output, annotator=args.annotator,
                                completion_note=args.completion_note, split=args.split,
                                scenario_group=args.scenario_group) if args.command == "import" else
              apply_navigation(args.workspace, args.csv, args.output, reviewer=args.reviewer))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
