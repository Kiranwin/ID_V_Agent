"""Read-only camera quantization audit over exact 200 ms Raw Input windows."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from idv_agent.scripts.prepare_mvp_v6 import read_csv, replay_window, quantize

CANDIDATES = {
    "legacy5": (-110, -25, 0, 25, 110),
    "fine5_candidate": (-110, -12, 0, 12, 110),
    "micro7_candidate": (-110, -25, -8, 0, 8, 25, 110),
    "wide7_candidate": (-200, -110, -25, 0, 25, 110, 200),
}


def audit(session: Path) -> dict:
    ft = read_csv(session / "frame_timestamps.csv")
    times = [int(row["timestamp_ns"]) for row in ft]
    events = read_csv(session / "events.csv")
    mouse = read_csv(session / "mouse_deltas.csv")
    origin = min(times[0], *(int(row["timestamp_ns"]) for row in mouse)) if mouse else times[0]
    # Shift the origin by 0/50/100/150 ms: a bucket decision must not depend
    # on picking one convenient temporal phase. Windows overlap across phases.
    phases = []
    for offset in (0, 50, 100, 150):
        start = origin - 1 + offset * 1_000_000
        windows = []
        while start + 200_000_000 <= times[-1]:
            windows.append(replay_window(events, mouse, start, start + 200_000_000))
            start += 200_000_000
        values = [window["mouse_dx_counts"] for window in windows]
        nonzero = [value for value in values if value]
        metrics = {}
        for name, commands in CANDIDATES.items():
            errors = [abs(quantize(value, commands) - value) for value in values]
            metrics[name] = {
                "all_windows_mae": statistics.mean(errors) if errors else None,
                "nonzero_windows_mae": statistics.mean(abs(quantize(value, commands) - value)
                                                       for value in nonzero) if nonzero else None,
                "nonzero_to_zero": sum(quantize(value, commands) == 0 for value in nonzero),
                "commands": dict(Counter(str(quantize(value, commands)) for value in values)),
            }
        phases.append({"offset_ms": offset, "window_count": len(windows),
                       "nonzero_dx": nonzero, "candidate_metrics": metrics,
                       "windows": windows})
    return {"session": session.name, "unit": "Raw Input counts, not degrees or resized image pixels",
            "window_ms": 200, "total_mouse_dx": sum(int(r["dx"]) for r in mouse),
            "total_mouse_dy": sum(int(r["dy"]) for r in mouse), "phases": phases}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"schema": "idv.camera_quantization_audit.v1", "candidates": CANDIDATES,
              "sessions": [audit(session) for session in args.sessions],
              "decision": "retain_legacy5_provisionally; two mixed navigation sessions do not justify expansion",
              "limitations": ["no held-out calibration", "window phase samples are correlated",
                              "count reconstruction error is not control success",
                              "mouse gain and command injection response unmeasured"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for session in report["sessions"]:
        print(session["session"], session["total_mouse_dx"], session["total_mouse_dy"],
              session["phases"][0]["nonzero_dx"], session["phases"][0]["candidate_metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
