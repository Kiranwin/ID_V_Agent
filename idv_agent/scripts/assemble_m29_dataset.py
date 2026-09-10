"""Assemble versioned M29 exports without duplicating shared endpoint frames."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from idv_agent.scripts.train_m29 import label_distributions
from idv_agent.training.m29_dataset import M29Dataset


def canonicalize_mvp_row(row: dict) -> dict:
    """Apply the settled prompt/state contract when rebuilding a dataset."""
    row = json.loads(json.dumps(row, ensure_ascii=False))
    facts = row.get("targets", {}).get("facts", {})
    decision = row.get("targets", {}).get("decision", {})
    action = row.get("targets", {}).get("action", {})
    if facts.get("interact_prompt") is True:
        # A visible prompt means the endpoint is already close enough. It is
        # an interact/Q state, never a navigation endpoint.
        decision["phase"] = "interact"
        action["source"] = "exclude"
        row.setdefault("masks", {})["navigation"] = False
    return row


def assemble(inputs: list[Path], output: Path) -> dict:
    if not inputs:
        raise ValueError("at least one input export is required")
    if output.exists():
        raise ValueError("output exists; use a new versioned path")
    dataset = M29Dataset(",".join(str(path) for path in inputs))
    if len(dataset.splits) != 1:
        raise ValueError(f"one assembled file cannot mix dataset splits: {sorted(dataset.splits)}")
    content = "".join(json.dumps(canonicalize_mvp_row(row), ensure_ascii=False) + "\n"
                      for row in dataset.records)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    # Re-open the final artifact so its own hashes/schema are authoritative.
    final = M29Dataset(output, expected_split=next(iter(dataset.splits)))
    return {
        "schema": "m29.dataset_assembly.v1",
        "output": str(output.resolve()),
        "rows": len(final),
        "sessions": len(final.sessions),
        "scenario_groups": sorted(final.groups),
        "split": next(iter(final.splits)),
        "coverage": final.coverage(),
        "label_distributions": label_distributions(final),
        "sources": dataset.files,
        "duplicate_endpoint_policy": "prefer_full_when_q_target_and_latest_image_match",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(assemble(args.inputs, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
