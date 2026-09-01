"""Record raw human demonstrations for the native VLA pipeline.

Unlike ``record.py`` this command writes only raw frames, a shared-clock frame
timestamp file, and input events. It deliberately does not create legacy
``per_frame_actions.csv``/``samples.jsonl`` labels; conversion and annotation
are explicit later steps.

Run only in the official custom scenario/training room. Input capture requires
an administrator terminal on Windows; this recorder never injects input.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from idv_agent.capture.input_logger import InputRecorder
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture
from idv_agent.configs.game_mode import DEFAULT_GAME_MODE, GAME_MODE_CHOICES
from idv_agent.vla.action_chunk import (
    DEFAULT_ACTION_DELAY_FRAMES,
    VLA_SCHEMA_VERSION,
)


def record(args) -> Path:
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}"
    session_dir = args.output / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    events_csv = session_dir / "events.csv"
    positions_csv = session_dir / "mouse_positions.csv"
    frame_ts_csv = session_dir / "frame_timestamps.csv"

    import cv2

    recorder = InputRecorder(events_csv=events_csv, positions_csv=positions_csv)
    frame_timestamps: list[tuple[int, int]] = []
    start_ts = time.perf_counter_ns()
    start_wall = time.perf_counter()
    frame_interval = 1.0 / max(args.fps, 1)
    next_deadline = time.perf_counter()
    n_frames = 0
    print(f"[record-vla] raw session={session_dir}")
    print("[record-vla] 仅限官方自定义剧本/训练营；本命令不注入输入")
    recorder.start()
    try:
        with ScreenCapture(CaptureConfig(fps=args.fps, window_title=args.window_title)) as cap:
            print("[record-vla] 录制开始；Ctrl+C 结束")
            while True:
                now = time.perf_counter()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                elif now - next_deadline > frame_interval:
                    next_deadline = now
                if args.max_seconds > 0 and time.perf_counter() - start_wall >= args.max_seconds:
                    break
                if args.max_frames > 0 and n_frames >= args.max_frames:
                    break
                frame = cap.grab()
                if frame is None:
                    continue
                capture_ts = time.perf_counter_ns()
                h, w = frame.shape[:2]
                if w > args.max_width:
                    scale = args.max_width / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                       interpolation=cv2.INTER_AREA)
                path = frames_dir / f"{n_frames:08d}.jpg"
                cv2.imwrite(str(path), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                frame_timestamps.append((n_frames, capture_ts))
                n_frames += 1
                next_deadline += frame_interval
    except KeyboardInterrupt:
        print("\n[record-vla] 中断")
    finally:
        recorder.stop()

    end_ts = time.perf_counter_ns()
    frame_ts_csv.write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{ts}\n" for i, ts in frame_timestamps),
        encoding="utf-8",
    )
    duration_s = max((end_ts - start_ts) / 1e9, 1e-9)
    meta = {
        "recording_type": "vla_raw",
        "vla_schema_version": VLA_SCHEMA_VERSION,
        "session_id": session_id,
        "source": "human",
        "mode": args.mode,
        "task_name": args.task_name,
        "task_instruction": args.task_instruction,
        "target_fps": args.fps,
        "num_frames": n_frames,
        "effective_fps": n_frames / duration_s,
        "start_ts_ns": start_ts,
        "end_ts_ns": end_ts,
        "window_title": args.window_title,
        "action_delay_frames": DEFAULT_ACTION_DELAY_FRAMES,
        "frame_timestamp_clock": "time.perf_counter_ns",
        "input_timestamp_clock": "time.perf_counter_ns",
        "max_width": args.max_width,
        "jpeg_quality": args.jpeg_quality,
        "note": args.note,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[record-vla] 完成：{n_frames} 帧 -> {session_dir}")
    return session_dir


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VLA 原始示范录制（不注入输入）")
    parser.add_argument("--window-title", default="第五人格")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("data/vla_raw_sessions"))
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-width", type=int, default=1334)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--task-name", default="find_cipher_and_decode")
    parser.add_argument("--task-instruction", default="找到密码机，靠近并进入破译")
    parser.add_argument("--mode", choices=GAME_MODE_CHOICES, default=DEFAULT_GAME_MODE,
                        help="游戏模式；写入 meta.json 并作为 VLA 条件 token")
    parser.add_argument("--note", default="只在官方自定义剧本/训练营录制")
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.max_width <= 0 or not 1 <= args.jpeg_quality <= 100:
        parser.error("fps/max-width 必须为正数，jpeg-quality 必须为 1..100")
    record(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
