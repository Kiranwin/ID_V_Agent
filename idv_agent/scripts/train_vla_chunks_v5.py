"""Build, validate, and audit canonical VLA action-chunk v5 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idv_agent.scripts.train_vla_chunks_v4 import audit
from idv_agent.scripts.build_vla_chunks import build
from idv_agent.vla.action_chunk import (
    ACTION_CHUNK_HORIZON,
    MACRO_FRAMES,
    VLA_SCHEMA_VERSION_V5,
)


def build_train(session: Path, output: Path, *, stride: int = 3,
                anchor_stride: int = 12, history_stride: int = 3, history: int = 8,
                slow_period_s: float = 1.0) -> dict:
    if history != 8:
        raise ValueError("v5 history 必须固定为 8")
    build(session, output, stride=stride, anchor_stride=anchor_stride,
          history_stride=history_stride, history=history,
          schema_version=VLA_SCHEMA_VERSION_V5, slow_period_s=slow_period_s)
    report = audit(output, schema_version=VLA_SCHEMA_VERSION_V5)
    report["sampling"] = {
        "anchor_stride": anchor_stride,
        "history_stride": history_stride,
        "history": history,
        "horizon": ACTION_CHUNK_HORIZON,
        "macro_frames": MACRO_FRAMES,
    }
    report["session"] = session.name
    report["output"] = str(output)
    report_path = output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=3,
                        help="兼容旧调用；未显式覆盖时 v5 使用 anchor=12/history=3")
    parser.add_argument("--anchor-stride", type=int, default=12)
    parser.add_argument("--history-stride", type=int, default=3)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--slow-period-s", type=float, default=1.0)
    args = parser.parse_args(argv)
    report = build_train(args.session, args.output, stride=args.stride,
                         anchor_stride=args.anchor_stride, history_stride=args.history_stride,
                         history=args.history, slow_period_s=args.slow_period_s)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
