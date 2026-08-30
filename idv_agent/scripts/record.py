"""录制会话（管理员权限！NeAC 阻断非管理员 hook）。

用法：
    python -m idv_agent.scripts.record --list-windows
    python -m idv_agent.scripts.record --window-title "Identity V" --fps 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from idv_agent.capture.input_logger import InputRecorder, list_windows
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture


def _save_frame(cv2, frame, path):
    cv2.imwrite(str(path), frame)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="ID_V_Agent 录制")
    p.add_argument("--list-windows", action="store_true", help="列出窗口标题")
    p.add_argument("--window-title", type=str, default="Identity V")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--output", type=Path, default=Path("data/sessions"),
                   help="会话根目录，自动创建 <timestamp>_<rand> 子目录")
    p.add_argument("--max-seconds", type=float, default=0.0,
                   help=">0 时自动在 N 秒后结束（不指定则 Ctrl+C 停止）")
    p.add_argument("--max-frames", type=int, default=10000,
                   help="最多保存帧数（默认 10000，设 0 表示不限；避免产生缺失图片样本）")
    args = p.parse_args(argv)

    if args.list_windows:
        wins = list_windows()
        for w in wins:
            print(w)
        return 0

    from datetime import datetime
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}"
    session_dir = args.output / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    events_csv = session_dir / "events.csv"
    positions_csv = session_dir / "mouse_positions.csv"
    frame_ts_csv = session_dir / "frame_timestamps.csv"

    import cv2
    print(f"[record] 会话目录: {session_dir}  (管理员权限运行确保 hook 可用)")
    recorder = InputRecorder(events_csv=events_csv, positions_csv=positions_csv)
    recorder.start()

    cfg = CaptureConfig(fps=args.fps, window_title=args.window_title)
    start_ts = time.perf_counter_ns()
    start_wall = time.perf_counter()
    n_frames = 0
    frame_timestamps = []
    try:
        with ScreenCapture(cfg) as cap:
            deadline = None
            print("[record] 录制已开始；Ctrl+C 结束（或 --max-seconds 自动结束）")
            while True:
                if args.max_seconds > 0 and time.perf_counter() - start_wall >= args.max_seconds:
                    break
                if args.max_frames > 0 and n_frames >= args.max_frames:
                    print(f"[record] 达到 --max-frames={args.max_frames}，自动停止")
                    break
                # 录制控制：使用 --max-seconds 或 Ctrl+C，避免依赖全局热键权限。
                frame = cap.grab()
                if frame is None:
                    time.sleep(1 / args.fps)
                    continue
                capture_ts = time.perf_counter_ns()
                p = frames_dir / f"{n_frames:08d}.jpg"
                # 下采样到 ~1334x750 保持体积 / 帧率
                h, w = frame.shape[:2]
                if w > 1334:
                    scale = 1334 / w
                    frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                _save_frame(cv2, frame, p)
                frame_timestamps.append((n_frames, capture_ts))
                n_frames += 1
                time.sleep(1 / args.fps)
    except KeyboardInterrupt:
        print("\n[record] 中断")
    finally:
        recorder.stop()

    end_ts = time.perf_counter_ns()
    # 保留真实采样时钟，提取阶段优先使用它，避免 sleep/抓帧抖动造成对齐偏差。
    with frame_ts_csv.open("w", encoding="utf-8") as f:
        f.write("frame_id,timestamp_ns\n")
        for frame_id, ts in frame_timestamps:
            f.write(f"{frame_id},{ts}\n")
    duration_s = max((end_ts - start_ts) / 1e9, 1e-9)
    effective_fps = n_frames / duration_s
    # 帧时间戳：以录制起止为准做线性网格（简单近似）
    meta = {
        "session_id": session_id,
        "target_fps": args.fps,
        "num_frames": n_frames,
        "effective_fps": effective_fps,
        "start_ts_ns": start_ts,
        "end_ts_ns": end_ts,
        "window_title": args.window_title,
        "note": "P7: 录制时请记录游戏鼠标灵敏度并保持一致",
    }
    (session_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                           encoding="utf-8")
    print(f"[record] 完成：{n_frames} 帧 -> {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
