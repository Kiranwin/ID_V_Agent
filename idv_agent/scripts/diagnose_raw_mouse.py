"""Diagnose whether Windows Raw Input exposes relative mouse motion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from idv_agent.capture.raw_input_mouse import RawInputMouse, summarize_deltas, write_deltas_csv


def diagnose(*, duration: float, output: Path | None = None) -> dict[str, int | float]:
    if duration <= 0:
        raise ValueError("duration 必须 > 0")
    collector = RawInputMouse()
    collector.start()
    try:
        print(f"[raw-mouse] 采集 {duration:.1f}s；请在训练营中移动镜头")
        time.sleep(duration)
    finally:
        collector.stop()
    if output is not None:
        write_deltas_csv(output, collector.deltas)
        print(f"[raw-mouse] 已写入 {output}")
    summary = summarize_deltas(collector.deltas)
    print("[raw-mouse] " + json.dumps(summary, ensure_ascii=False))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="诊断 Windows Raw Input 鼠标相对位移")
    parser.add_argument("--duration", type=float, default=10.0, help="采集秒数，默认 10")
    parser.add_argument("--output", type=Path, default=None,
                        help="可选：输出 mouse_deltas.csv")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("duration 必须 > 0")
    try:
        diagnose(duration=args.duration, output=args.output)
    except (RuntimeError, OSError) as exc:
        parser.exit(1, f"[raw-mouse] 失败：{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
