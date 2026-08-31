"""M0 DDA 捕获基准（只读屏幕，不发送输入、不写训练数据）。

示例：
    python -m idv_agent.scripts.benchmark_capture --title "第五人格" --seconds 60
"""

from __future__ import annotations

import argparse
import statistics
import time

from idv_agent.capture.input_logger import find_window_region
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture


def parse_region(s: str):
    try:
        vals = tuple(int(x.strip()) for x in s.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("region 必须是 left,top,width,height") from exc
    if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
        raise argparse.ArgumentTypeError("region 必须是有效的 left,top,width,height")
    return vals


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="DDA 屏幕捕获延迟基准（只读）")
    p.add_argument("--title", default="第五人格", help="窗口标题（模糊匹配）")
    p.add_argument("--region", type=parse_region, default=None)
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seconds", type=float, default=10.0)
    args = p.parse_args(argv)
    if args.fps <= 0 or args.seconds <= 0:
        p.error("--fps 和 --seconds 必须为正数")

    region = args.region
    if region is None and args.title:
        found = find_window_region(args.title)
        if found is not None:
            region = found
            print(f"[capture] window region={region} (left,top,width,height)")
        else:
            print(f"[capture] 未找到窗口 {args.title!r}，将尝试全屏")

    cfg = CaptureConfig(fps=args.fps, region=region, window_title=args.title)
    intervals_ms: list[float] = []
    frames = 0
    empty = 0
    deadline = time.perf_counter() + args.seconds
    next_tick = time.perf_counter()
    with ScreenCapture(cfg) as cap:
        while time.perf_counter() < deadline:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(next_tick - now)
            next_tick += 1.0 / args.fps
            t0 = time.perf_counter()
            frame = cap.grab()
            intervals_ms.append((time.perf_counter() - t0) * 1000.0)
            if frame is None:
                empty += 1
            else:
                frames += 1

    total = frames + empty
    elapsed = max(args.seconds, 1e-9)
    def pct(q):
        return statistics.quantiles(intervals_ms, n=100)[q - 1] if len(intervals_ms) >= 2 else intervals_ms[0]
    print("=== DDA capture benchmark ===")
    print(f"requested_fps={args.fps} duration_s={args.seconds:.2f}")
    print(f"grab_calls={total} frames={frames} empty={empty} effective_fps={frames/elapsed:.2f}")
    print(f"empty_rate={empty/max(total,1)*100:.2f}% grab_ms="
          f"mean={statistics.mean(intervals_ms):.2f} p50={pct(50):.2f} p95={pct(95):.2f} max={max(intervals_ms):.2f}")
    print("gate=" + ("PASS (capture <=33ms and empty_rate <5%)"
                     if pct(95) <= 33.0 and empty / max(total, 1) < 0.05
                     else "CHECK (未达到 M0 捕获门槛)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
