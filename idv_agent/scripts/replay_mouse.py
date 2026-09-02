"""Replay keyboard and mouse input from a raw VLA session."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from idv_agent.agent.action_decoder import Command
from idv_agent.agent.action_executor import ActionExecutor


@dataclass(frozen=True)
class ReplayEvent:
    delay_s: float
    kind: str
    code: str | None = None
    dx_px: float = 0.0
    dy_px: float = 0.0


def _load_mouse_delta_events(path: Path, *, scale: float):
    df = pd.read_csv(path)
    missing = {"timestamp_ns", "dx", "dy"} - set(df.columns)
    if missing:
        raise ValueError(f"mouse_deltas.csv 缺少列: {', '.join(sorted(missing))}")
    return [(int(r.timestamp_ns), "mouse_move", None, float(r.dx) * scale, float(r.dy) * scale)
            for r in df.sort_values("timestamp_ns").itertuples(index=False) if float(r.dx) or float(r.dy)]


def load_replay_events(session: Path, *, mouse_deltas: Path | None = None,
                       scale: float = 1.0) -> list[ReplayEvent]:
    events_path = session / "events.csv"
    if not events_path.is_file():
        raise FileNotFoundError(f"找不到 {events_path}")
    events = pd.read_csv(events_path)
    missing = {"timestamp_ns", "kind", "code"} - set(events.columns)
    if missing:
        raise ValueError(f"events.csv 缺少列: {', '.join(sorted(missing))}")
    keyboard_raw: list[tuple[int, str, str | None, float, float]] = []
    pressed: set[str] = set()
    for r in events.sort_values("timestamp_ns").itertuples(index=False):
        ts, kind, code = int(r.timestamp_ns), str(r.kind), str(r.code)
        if kind == "key_down":
            if code in pressed or code in {"key:f9", "key:f10"}:
                continue
            pressed.add(code); keyboard_raw.append((ts, "press", code, 0.0, 0.0))
        elif kind == "key_up":
            was_pressed = code in pressed
            pressed.discard(code)
            if was_pressed and code not in {"key:f9", "key:f10"}:
                keyboard_raw.append((ts, "release", code, 0.0, 0.0))
        elif kind in {"mouse_down", "mouse_up"}:
            keyboard_raw.append((ts, "press" if kind == "mouse_down" else "release", code, 0.0, 0.0))
    # A separate diagnostic run has a different perf_counter epoch. Normalize
    # that external stream independently; same-session files keep their
    # original shared clock and therefore preserve keyboard/mouse alignment.
    raw = []
    if keyboard_raw:
        origin = keyboard_raw[0][0]
        raw.extend((ts - origin, kind, code, dx, dy) for ts, kind, code, dx, dy in keyboard_raw)
    if mouse_deltas is not None:
        mouse_raw = _load_mouse_delta_events(mouse_deltas, scale=scale)
        if mouse_raw:
            if mouse_deltas.resolve().parent == session.resolve():
                # Rebase to keyboard origin below after appending.
                raw.extend((ts, kind, code, dx, dy) for ts, kind, code, dx, dy in mouse_raw)
            else:
                origin = mouse_raw[0][0]
                raw.extend((ts - origin, kind, code, dx, dy) for ts, kind, code, dx, dy in mouse_raw)
    raw.sort(key=lambda item: item[0])
    if not raw:
        return []
    out: list[ReplayEvent] = []
    # Same-session mouse timestamps are absolute perf_counter values while
    # keyboard entries above are already keyboard-relative.
    if mouse_deltas is not None and mouse_deltas.resolve().parent == session.resolve() and keyboard_raw:
        keyboard_origin = keyboard_raw[0][0]
        raw = [(ts - keyboard_origin, kind, code, dx, dy) for ts, kind, code, dx, dy in raw]
        raw.sort(key=lambda item: item[0])
    prev = raw[0][0]
    for ts, kind, code, dx, dy in raw:
        out.append(ReplayEvent(max(0.0, (ts - prev) / 1e9), kind, code, dx, dy))
        prev = ts
    return out


def replay(session: Path, *, send_input: bool = False, scale: float = 1.0,
           start_delay: float = 5.0, mouse_deltas: Path | None = None) -> int:
    events = load_replay_events(session, mouse_deltas=mouse_deltas, scale=scale)
    keyboard_count = sum(e.kind in {"press", "release"} and e.code and e.code.startswith("key:") for e in events)
    mouse_count = sum(e.kind == "mouse_move" for e in events)
    print(f"[replay] events={len(events)} keyboard={keyboard_count} mouse_moves={mouse_count} duration={sum(e.delay_s for e in events):.3f}s scale={scale:g}")
    if not send_input:
        print("[replay] dry-run：未发送输入；使用 --send-input 才会实际回放")
        return 0
    if start_delay < 0:
        raise ValueError("start-delay 必须 >= 0")
    print("[replay] 即将发送键盘和鼠标输入；请确认仅在官方自定义剧本/训练营中运行")
    if start_delay:
        print(f"[replay] {start_delay:.1f}s 后开始，请切回游戏窗口")
        time.sleep(start_delay)
    executor = ActionExecutor(dry_run=False)
    try:
        for e in events:
            if e.delay_s:
                time.sleep(e.delay_s)
            if e.kind == "mouse_move":
                executor.execute([Command(kind="mouse_move", dx_px=e.dx_px, dy_px=e.dy_px)])
            else:
                executor.execute([Command(kind=e.kind, code=e.code)])
    finally:
        executor.shutdown()
    print("[replay] 回放完成")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="回放 VLA raw session 的键盘和鼠标输入")
    parser.add_argument("session", type=Path)
    parser.add_argument("--send-input", action="store_true", help="实际发送输入（默认 dry-run）")
    parser.add_argument("--scale", type=float, default=1.0, help="鼠标位移缩放倍数，默认 1.0")
    parser.add_argument("--start-delay", type=float, default=5.0, help="发送前等待秒数，默认 5")
    parser.add_argument("--mouse-deltas", type=Path, default=None, help="Raw Input 相对位移 CSV")
    args = parser.parse_args(argv)
    if args.scale < 0 or args.start_delay < 0:
        parser.error("scale/start-delay 必须 >= 0")
    mouse_deltas = args.mouse_deltas
    if mouse_deltas is None:
        candidate = args.session / "mouse_deltas.csv"
        mouse_deltas = candidate if candidate.is_file() else None
    return replay(args.session, send_input=args.send_input, scale=args.scale,
                  start_delay=args.start_delay, mouse_deltas=mouse_deltas)


if __name__ == "__main__":
    raise SystemExit(main())
