"""Fail-closed audit for the V6 workbench artifacts consumed by M29."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from idv_agent.scripts.prepare_mvp_v6 import validate_workspace
from idv_agent.scripts.train_m29 import label_distributions, validate_readiness
from idv_agent.training.m29_dataset import M29Dataset


def _review_counts(workspace: Path) -> dict:
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    path = Path(manifest["raw_root"]) / "navigation_review_with_state.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "path": str(path.resolve()), "rows": len(rows),
        "reviewed": sum(row.get("reviewed") == "reviewed" for row in rows),
        "pending": sum(row.get("reviewed") != "reviewed" for row in rows),
        "decoding_true": sum(row.get("reviewed") == "reviewed" and row.get("decoding") == "true" for row in rows),
        "decoding_false": sum(row.get("reviewed") == "reviewed" and row.get("decoding") == "false" for row in rows),
    }


def audit(workspaces: list[Path], q_data: Path, full_train: Path | None,
          full_val: Path | None) -> dict:
    errors = []
    workspace_reports = []
    for workspace in workspaces:
        try:
            validation = validate_workspace(workspace, prompt_only=True)
            review = _review_counts(workspace)
            workspace_reports.append({"workspace": str(workspace.resolve()),
                                      "validation": validation, "review": review})
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"{workspace}: {exc}")
    try:
        q_dataset = M29Dataset(q_data, expected_split="train")
        q_coverage, q_distribution = validate_readiness(q_dataset, "q")
        q_report = {"path": str(q_data.resolve()), "rows": len(q_dataset),
                    "coverage": q_coverage, "label_distributions": q_distribution,
                    "sessions": sorted(q_dataset.sessions), "groups": sorted(q_dataset.groups)}
    except (OSError, KeyError, ValueError) as exc:
        q_report = {"path": str(q_data), "error": str(exc)}
        errors.append(f"Q data: {exc}")
    full_report = None
    if full_train is None or full_val is None:
        errors.append("full M29 requires explicit independent train and val exports")
    else:
        try:
            train = M29Dataset(full_train, expected_split="train")
            val = M29Dataset(full_val, expected_split="val")
            if train.sessions & val.sessions or train.groups & val.groups:
                raise ValueError("full train/val session or scenario-group overlap")
            train_coverage, train_distribution = validate_readiness(train, "full")
            val_coverage, val_distribution = validate_readiness(val, "full")
            full_report = {
                "train": {"path": str(full_train.resolve()), "coverage": train_coverage,
                          "label_distributions": train_distribution},
                "val": {"path": str(full_val.resolve()), "coverage": val_coverage,
                        "label_distributions": val_distribution},
            }
        except (OSError, KeyError, ValueError) as exc:
            full_report = {"error": str(exc)}
            errors.append(f"full data: {exc}")
    total_rows = sum(report["review"]["rows"] for report in workspace_reports)
    reviewed = sum(report["review"]["reviewed"] for report in workspace_reports)
    if workspace_reports and reviewed != total_rows:
        errors.append(f"navigation/state review incomplete: {reviewed}/{total_rows}")
    return {
        "schema": "m29.v6_data_audit.v1", "gate_pass": not errors,
        "errors": errors, "workspaces": workspace_reports,
        "q_data": q_report, "full_data": full_report,
        "review_progress": {"reviewed": reviewed, "total": total_rows,
                            "pending": total_rows - reviewed},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspaces", type=Path, nargs="+", required=True)
    parser.add_argument("--q-data", type=Path, required=True)
    parser.add_argument("--full-train", type=Path)
    parser.add_argument("--full-val", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output exists; use a new audit report path")
    report = audit(args.workspaces, args.q_data, args.full_train, args.full_val)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "gate_pass": report["gate_pass"],
                      "errors": report["errors"]}, ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
