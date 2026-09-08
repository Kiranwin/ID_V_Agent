"""Evaluate action-to-next-observation evidence for the learned MVP."""
from __future__ import annotations
import argparse, json
from pathlib import Path
from idv_agent.agent.mvp_closed_loop import ClosedLoopTransition, verify_transitions

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    rows = []
    for line_no, line in enumerate(args.trace.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(ClosedLoopTransition.from_dict(json.loads(line)))
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{args.trace}:{line_no}: {exc}") from exc
    report = verify_transitions(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["mvp_pass"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
