"""Record raw human demonstrations for the native VLA pipeline.

This command writes only raw frames, a shared-clock frame
timestamp file, and input events. It deliberately does not create legacy
``per_frame_actions.csv``/``samples.jsonl`` labels; conversion and annotation
are explicit later steps.

Run only in the official custom scenario/training room. Input capture requires
an administrator terminal on Windows; this recorder never injects input.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
from threading import Lock
import time
from datetime import datetime
from pathlib import Path

from idv_agent.capture.input_logger import InputRecorder
from idv_agent.capture.raw_input_mouse import RawInputMouse, write_deltas_csv
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture
from idv_agent.configs.game_mode import DEFAULT_GAME_MODE, GAME_MODE_CHOICES
from idv_agent.vla.action_chunk import (
    DEFAULT_ACTION_DELAY_FRAMES,
    VLA_SCHEMA_VERSION,
)


class RecordingControls:
    """Thread-safe F9/F10 state shared by the hotkey listener and loop."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._toggle_pending = False
        self.exit_requested = False

    def on_key(self, key_name: str) -> None:
        key_name = str(key_name).lower()
        with self._lock:
            if key_name == "f9":
                self._toggle_pending = True
            elif key_name == "f10":
                self.exit_requested = True
                # F10 while recording also acts as an implicit stop.
                self._toggle_pending = True

    def consume_toggle(self) -> bool:
        with self._lock:
            pending = self._toggle_pending
            self._toggle_pending = False
            return pending


def _start_control_listener(controls: RecordingControls):
    """Start a small global listener for F9/F10 and return it for cleanup."""
    try:
        from pynput import keyboard
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pynput 未安装，无法监听 F9/F10") from exc

    pressed: set[str] = set()

    def on_press(key):
        try:
            name = key.name if isinstance(key, keyboard.Key) else None
        except Exception:
            name = None
        if name in {"f9", "f10"} and name not in pressed:
            pressed.add(name)
            controls.on_key(name)

    def on_release(key):
        try:
            name = key.name if isinstance(key, keyboard.Key) else None
        except Exception:
            name = None
        if name in pressed:
            pressed.remove(name)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    return listener


def _new_session(args):
    session_id = f"{datetime.now():%Y%m%d_%H%M%S_%f}"
    session_dir = args.output / session_id
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    return session_id, session_dir, frames_dir


def _trim_idle_boundaries(*, session_dir: Path,
                          frame_timestamps: list[tuple[int, int]],
                          raw_mouse: RawInputMouse | None,
                          context_frames: int = 0) -> tuple[list[tuple[int, int]], bool]:
    """Remove the leading/trailing frames with no recorded user input.

    Idle spans in the middle of a demonstration are meaningful and are kept.
    ``context_frames`` can retain visual context around the boundaries; the
    recorder uses zero by default so the F9-to-first-input idle span is removed
    exactly. The event and Raw Input files are clipped to the input interval.
    """
    if not frame_timestamps:
        return [], False
    context_frames = max(0, int(context_frames))
    activity_ts: list[int] = []
    held_codes: set[str] = set()
    events_path = session_dir / "events.csv"
    if events_path.is_file():
        with events_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                code = str(row.get("code") or "")
                if code in {"key:f9", "key:f10"}:
                    continue
                try:
                    timestamp = int(float(row["timestamp_ns"]))
                except (KeyError, TypeError, ValueError):
                    continue
                activity_ts.append(timestamp)
                kind = str(row.get("kind") or "")
                if kind in {"key_down", "mouse_down"}:
                    held_codes.add(code)
                elif kind in {"key_up", "mouse_up"}:
                    held_codes.discard(code)
    if raw_mouse is not None:
        activity_ts.extend(item.timestamp_ns for item in raw_mouse.deltas
                           if item.dx or item.dy)

    frame_times = [ts for _, ts in frame_timestamps]
    clip_start = clip_end = None
    if not activity_ts:
        # Keep the session structurally valid but discard every idle frame.
        selected_start, selected_end = 0, -1
    else:
        first_input = min(activity_ts)
        last_input = max(activity_ts)
        # A key/button that is still held when F9 stops remains active until
        # the final captured frame; otherwise trimming would erase the actual
        # action after its initial key_down edge.
        if held_codes:
            last_input = frame_times[-1]
        clip_start, clip_end = first_input, last_input
        selected_start = max(0, bisect.bisect_left(frame_times, first_input) - context_frames)
        selected_end = min(len(frame_timestamps) - 1,
                           bisect.bisect_left(frame_times, last_input) + context_frames)

    selected = frame_timestamps[selected_start:selected_end + 1]
    selected_ids = {frame_id for frame_id, _ in selected}
    frames_dir = session_dir / "frames"
    # Rename retained files through temporary names before deleting/renumbering
    # to avoid collisions when the kept range starts at a nonzero frame id.
    temp_files: list[tuple[Path, Path]] = []
    for old_id, _ in selected:
        source = frames_dir / f"{old_id:08d}.jpg"
        if source.is_file():
            temp = frames_dir / f".trim_{old_id:08d}.jpg"
            source.rename(temp)
            temp_files.append((temp, source))
    for source in frames_dir.glob("*.jpg"):
        try:
            if int(source.stem) not in selected_ids:
                source.unlink()
        except ValueError:
            continue
    renumbered: list[tuple[int, int]] = []
    for new_id, (temp, _old_path) in enumerate(temp_files):
        target = frames_dir / f"{new_id:08d}.jpg"
        temp.rename(target)
        old_id = int(_old_path.stem)
        timestamp = next(ts for frame_id, ts in selected if frame_id == old_id)
        renumbered.append((new_id, timestamp))

    if renumbered:
        start_ts, end_ts = renumbered[0][1], renumbered[-1][1]
    else:
        start_ts = end_ts = frame_times[0]
    (session_dir / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{ts}\n" for i, ts in renumbered),
        encoding="utf-8",
    )

    def _clip_csv(path: Path, fields: tuple[str, ...]) -> None:
        if not path.is_file():
            return
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        kept = []
        for row in rows:
            try:
                ts = int(float(row["timestamp_ns"]))
            except (KeyError, TypeError, ValueError):
                continue
            if clip_start is not None and clip_start <= ts <= clip_end:
                kept.append(row)
        kept.sort(key=lambda row: int(float(row["timestamp_ns"])))
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in fields} for row in kept)

    _clip_csv(events_path, ("timestamp_ns", "kind", "code", "value"))
    _clip_csv(session_dir / "mouse_deltas.csv", ("timestamp_ns", "dx", "dy"))
    return renumbered, bool(activity_ts)


def _finalize_session(*, args, session_id, session_dir, frame_timestamps,
                      n_frames, start_ts, end_ts) -> Path | None:
    frame_ts_csv = session_dir / "frame_timestamps.csv"
    frame_ts_csv.write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{ts}\n" for i, ts in frame_timestamps),
        encoding="utf-8",
    )
    mouse_deltas_path = session_dir / "mouse_deltas.csv"
    raw_mouse = getattr(args, "_raw_mouse", None)
    if raw_mouse is not None:
        write_deltas_csv(mouse_deltas_path, raw_mouse.deltas)
    trim_idle = bool(getattr(args, "trim_idle_boundaries", False))
    if trim_idle:
        frame_timestamps, had_input = _trim_idle_boundaries(
            session_dir=session_dir, frame_timestamps=frame_timestamps,
            raw_mouse=raw_mouse,
        )
    else:
        # Waiting before a decision and automatic decoding after Q are real
        # observations. Lack of keyboard input must not erase these frames.
        with (session_dir / "events.csv").open(encoding="utf-8", newline="") as handle:
            had_input = any(row.get("code") not in {"key:f9", "key:f10"}
                            for row in csv.DictReader(handle))
        had_input = had_input or bool(raw_mouse and any(
            item.dx or item.dy for item in raw_mouse.deltas))
    n_frames = len(frame_timestamps)
    if not frame_timestamps:
        shutil.rmtree(session_dir, ignore_errors=True)
        print(f"[record-vla] 丢弃空 session（未检测到输入）：{session_id}")
        return None
    # The first key edge may precede the first retained frame.  Keep that
    # event in the session clock so state replay can reconstruct a held key,
    # and make meta bounds cover both frames and retained input rows.
    frame_start, frame_end = frame_timestamps[0][1], frame_timestamps[-1][1]
    input_times: list[int] = []
    for path in (session_dir / "events.csv", session_dir / "mouse_deltas.csv"):
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8", newline="") as handle:
                input_times.extend(int(float(row["timestamp_ns"]))
                                   for row in csv.DictReader(handle)
                                   if row.get("timestamp_ns") not in (None, ""))
        except (OSError, TypeError, ValueError):
            pass
    start_ts = min([frame_start, *input_times])
    end_ts = max([frame_end, *input_times])
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
        # A one-frame segment has no measurable interval; report the target
        # rate instead of an artificial 1e9 FPS that raw-session validation
        # would reject.
        "effective_fps": ((n_frames - 1) * 1e9 / (frame_end - frame_start)
                          if n_frames > 1 and frame_end > frame_start else float(args.fps)),
        "start_ts_ns": start_ts,
        "end_ts_ns": end_ts,
        "window_title": args.window_title,
        "action_delay_frames": DEFAULT_ACTION_DELAY_FRAMES,
        "frame_timestamp_clock": "time.perf_counter_ns",
        "input_timestamp_clock": "time.perf_counter_ns",
        "max_width": args.max_width,
        "jpeg_quality": args.jpeg_quality,
        "note": args.note,
        "mouse_input_source": "raw_input",
        "camera_motion_valid": True,
        "raw_schema_version": "idv.raw_session.v2",
        "top_intent": "decipher",
        "idle_boundary_trimmed": trim_idle,
        "post_stop_seconds": float(getattr(args, "post_stop_seconds", 2.0)),
        "had_input": had_input,
    }
    (session_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[record-vla] 完成：{n_frames} 帧 -> {session_dir}")
    return session_dir


def record(args) -> Path | None:
    import cv2

    print("[record-vla] 仅限官方自定义剧本/训练营；本命令不注入输入")
    print("[record-vla] F9 开始/结束当前录制段，F10 退出 CLI")
    print("[record-vla] 开始后先静止约 2 秒；F9 停止时保留结果画面，F10 立即退出")
    controls = RecordingControls()
    hotkey_listener = _start_control_listener(controls)
    last_session = None
    try:
        with ScreenCapture(CaptureConfig(fps=args.fps, window_title=args.window_title)) as cap:
            while not controls.exit_requested:
                if not controls.consume_toggle():
                    time.sleep(0.05)
                    continue
                if controls.exit_requested:
                    break

                session_id, session_dir, frames_dir = _new_session(args)
                last_session = session_dir
                events_csv = session_dir / "events.csv"
                recorder = InputRecorder(
                    events_csv=events_csv,
                    ignored_codes={"key:f9", "key:f10"},
                )
                frame_timestamps: list[tuple[int, int]] = []
                start_ts = time.perf_counter_ns()
                start_wall = time.perf_counter()
                frame_interval = 1.0 / max(args.fps, 1)
                next_deadline = time.perf_counter()
                n_frames = 0
                stop_at = None
                print(f"[record-vla] 录制开始：{session_dir}")
                raw_mouse = RawInputMouse()
                try:
                    recorder.start()
                    raw_mouse.start()
                    args._raw_mouse = raw_mouse
                    while not controls.exit_requested:
                        now = time.perf_counter()
                        if now < next_deadline:
                            time.sleep(next_deadline - now)
                        elif now - next_deadline > frame_interval:
                            next_deadline = now
                        if controls.consume_toggle():
                            if stop_at is None:
                                stop_at = time.perf_counter() + float(getattr(args, "post_stop_seconds", 2.0))
                        if stop_at is not None and time.perf_counter() >= stop_at:
                            break
                        if args.max_seconds > 0 and time.perf_counter() - start_wall >= args.max_seconds:
                            break
                        if args.max_frames > 0 and n_frames >= args.max_frames:
                            break
                        # Anchor the frame to the capture request.  Measuring
                        # after grab/resize/JPEG write shifts the visual
                        # timestamp by the processing latency and can move an
                        # input event into the wrong frame at 20 FPS.
                        capture_ts = time.perf_counter_ns()
                        frame = cap.grab()
                        next_deadline += frame_interval
                        if frame is None:
                            continue
                        h, w = frame.shape[:2]
                        if w > args.max_width:
                            scale = args.max_width / w
                            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                               interpolation=cv2.INTER_AREA)
                        path = frames_dir / f"{n_frames:08d}.jpg"
                        if not cv2.imwrite(str(path), frame,
                                           [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                            raise RuntimeError(f"写入帧失败: {path}")
                        frame_timestamps.append((n_frames, capture_ts))
                        n_frames += 1
                finally:
                    raw_mouse.stop()
                    recorder.stop()
                    # Finalize only after both listeners have been stopped and
                    # their files are closed. Startup failures still cleanly
                    # release the recorder/raw-input resources.
                    if raw_mouse._thread is None:
                        finalized = _finalize_session(
                            args=args, session_id=session_id, session_dir=session_dir,
                            frame_timestamps=frame_timestamps, n_frames=n_frames,
                            start_ts=start_ts, end_ts=time.perf_counter_ns())
                        if finalized is None:
                            last_session = None
                if controls.exit_requested:
                    break
                print("[record-vla] 已停止；F9 开始下一段，F10 退出")
    except KeyboardInterrupt:
        print("\n[record-vla] 中断，退出")
    finally:
        try:
            hotkey_listener.stop()
        except Exception:
            pass
    return last_session


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="VLA 原始示范录制（不注入输入）")
    parser.add_argument("--window-title", default="第五人格")
    parser.add_argument("--fps", type=int, default=20,
                        help="采集 FPS；MVP v6 默认 20。Qwen 视觉解析不按此频率运行")
    parser.add_argument("--output", type=Path, default=Path("data/raw_sessions"))
    parser.add_argument("--post-stop-seconds", type=float, default=2.0,
                        help="F9 停止后继续采集结果画面的秒数；F10/时长上限仍立即结束")
    parser.add_argument("--trim-idle-boundaries", action="store_true",
                        help="仅历史兼容：删除边界空闲帧；新 MVP 录制应保留等待和破译结果")
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
    if not 0 <= args.post_stop_seconds <= 30:
        parser.error("post-stop-seconds 必须为 0..30")
    record(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
