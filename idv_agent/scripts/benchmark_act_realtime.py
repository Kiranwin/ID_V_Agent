"""ACT real-time dry-run benchmark.

Captures frames, encodes each frame once, keeps a rolling feature window and
runs slow/fast ACT consumers at independent fixed rates.  It never sends
keyboard or mouse input; output is an action trace plus latency statistics.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Event, Lock

import torch

from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture
from idv_agent.model.fast_slow_vla import SharedFastSlowVLA, SlowCondition
from idv_agent.scripts.train_vla import _load_act_backbone
from idv_agent.vla.action_chunk import BUTTON_NAMES, MOVE_DIRECTIONS, CAMERA_BUCKETS


def parse_output_size(value: str) -> tuple[int, int]:
    try:
        width, height = (int(part.strip()) for part in value.split(","))
    except (TypeError, ValueError) as exc:
        raise ValueError("output-size 必须是 width,height") from exc
    if width <= 0 or height <= 0:
        raise ValueError("output-size 必须为正数")
    return width, height


@dataclass(frozen=True)
class ActionChunkStep:
    move_dir: int
    camera_dx: int
    camera_dy: int
    buttons: tuple[int, ...]
    duration_frames: int


@dataclass(frozen=True)
class LatestFrame:
    payload: object
    captured_at: float
    frame_index: int


class LatestFrameSlot:
    """Single-slot producer/consumer buffer; stale frames are discarded."""

    def __init__(self):
        self._condition = Condition()
        self._item: LatestFrame | None = None

    def put(self, payload: object, *, captured_at: float, frame_index: int = 0) -> None:
        with self._condition:
            self._item = LatestFrame(payload, float(captured_at), int(frame_index))
            self._condition.notify()

    def take_latest(self) -> LatestFrame | None:
        with self._condition:
            item, self._item = self._item, None
            return item

    def wait_latest(self, stop: Event, timeout: float = 0.05) -> LatestFrame | None:
        with self._condition:
            while self._item is None and not stop.is_set():
                self._condition.wait(timeout)
            item, self._item = self._item, None
            return item


class FixedRateGate:
    """Absolute-time gate; producer cadence does not determine tick rate."""

    def __init__(self, hz: float, *, start: float = 0.0):
        if hz <= 0:
            raise ValueError("hz 必须为正数")
        self.period = 1.0 / float(hz)
        self.next_deadline = float(start)

    def due(self, now: float) -> bool:
        now = float(now)
        if now < self.next_deadline:
            return False
        skipped = int((now - self.next_deadline) // self.period)
        self.next_deadline += (skipped + 1) * self.period
        return True

def action_step_to_text(step: ActionChunkStep) -> str:
    pressed = [name for name, value in zip(BUTTON_NAMES, step.buttons) if value]
    button_text = ",".join(pressed) if pressed else "-"
    return (f"move={MOVE_DIRECTIONS[step.move_dir]} "
            f"camera=({step.camera_dx},{step.camera_dy}) "
            f"buttons={button_text} duration={step.duration_frames}")


class RealtimeWindow:
    """Small rolling feature window used by both ACT consumers."""

    def __init__(self, max_length: int = 3):
        if max_length < 1:
            raise ValueError("max_length 必须为正数")
        self.max_length = int(max_length)
        self._items: list[tuple[int, int, torch.Tensor]] = []
        self._lock = Lock()

    def append(self, feature: torch.Tensor, *, frame_index: int, timestamp_ns: int) -> None:
        if feature.ndim != 1:
            raise ValueError("feature 必须是 [D]")
        with self._lock:
            self._items.append((int(frame_index), int(timestamp_ns), feature))
            self._items = self._items[-self.max_length:]

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        with self._lock:
            items = list(self._items)
        if not items:
            raise RuntimeError("特征窗口为空")
        values = torch.stack([item[2] for item in items], dim=0).unsqueeze(0)
        valid = torch.ones((1, values.shape[1]), dtype=torch.bool, device=values.device)
        return values, valid, [item[0] for item in items]

    def latest_capture_time(self) -> float | None:
        with self._lock:
            return self._items[-1][1] / 1_000_000_000 if self._items else None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_model(args: argparse.Namespace, device: torch.device):
    amp_dtype = torch.float16 if device.type == "cuda" else torch.float32
    adapter, _, parent_manifest = _load_act_backbone(
        args.model_path, args.init_checkpoint, dtype=amp_dtype, device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "adapter" not in checkpoint or "core" not in checkpoint:
        raise ValueError("ACT checkpoint 缺少 adapter/core")
    adapter.visual_projection.load_state_dict(checkpoint["adapter"]["visual_projection"])
    adapter.condition_projection.load_state_dict(checkpoint["adapter"]["condition_projection"])
    temporal_dim = int(checkpoint.get("manifest", {}).get("training", {}).get("temporal_dim", args.temporal_dim))
    core = SharedFastSlowVLA(adapter.hidden_size, temporal_dim=temporal_dim,
                             history_action_dim=72).to(device=device, dtype=torch.float32)
    core.load_state_dict(checkpoint["core"])
    adapter.eval()
    core.eval()
    return adapter, core, parent_manifest


def _predict(core: SharedFastSlowVLA, window: RealtimeWindow, condition: SlowCondition,
             device: torch.device):
    features, valid, _ = window.snapshot()
    history = torch.zeros((1, 72), device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = core(features, condition, valid_mask=valid, history_actions=history,
                      run_slow=False)
    fast = output.fast
    move = fast.move_logits.argmax(-1)[0]
    dx = fast.camera_dx_logits.argmax(-1)[0]
    dy = fast.camera_dy_logits.argmax(-1)[0]
    buttons = (fast.button_logits.sigmoid() >= 0.5)[0]
    duration = fast.duration[0]
    steps = []
    for i in range(fast.move_logits.shape[1]):
        steps.append(ActionChunkStep(
            int(move[i]), CAMERA_BUCKETS[int(dx[i])], CAMERA_BUCKETS[int(dy[i])],
            tuple(int(v) for v in buttons[i].tolist()), int(round(float(duration[i])))))
    return output, steps


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    adapter, core, parent_manifest = _load_model(args, device)
    task_cache = adapter.encode_task_once(args.instruction, args.mode, task_id="realtime-benchmark")
    window = RealtimeWindow(args.history_frames)
    condition = core.initial_condition(1, device=device, mode_id=0)
    condition.mode_id = torch.tensor([0], dtype=torch.long, device=device)
    slow_period = 1.0 / args.slow_hz
    fast_period = 1.0 / args.fast_hz
    start_time = time.perf_counter()
    next_fast = next_slow = start_time
    captures = encoded = fast_ticks = slow_ticks = 0
    timings = {"capture": [], "encode": [], "fast": [], "slow": [], "total": [], "feature_age": []}
    latest_frame = LatestFrameSlot()
    stop = Event()
    capture_done = Event()
    worker_error: list[BaseException] = []
    print(f"[act-realtime] dry_run=True device={device} capture_fps={args.capture_fps} "
          f"fast_hz={args.fast_hz} slow_hz={args.slow_hz}")
    deadline = time.perf_counter() + args.seconds
    cfg = CaptureConfig(fps=args.capture_fps, region=args.region,
                        output_size=args.output_size, window_title=args.title)
    def capture_worker():
        nonlocal captures
        try:
            with ScreenCapture(cfg) as capture:
                next_capture = time.perf_counter()
                while not stop.is_set() and time.perf_counter() < deadline:
                    now = time.perf_counter()
                    if now < next_capture:
                        time.sleep(min(next_capture - now, 0.005))
                        continue
                    next_capture += 1.0 / args.capture_fps
                    t0 = time.perf_counter()
                    frame = capture.grab()
                    timings["capture"].append((time.perf_counter() - t0) * 1000)
                    captures += 1
                    if frame is not None:
                        latest_frame.put(frame, captured_at=time.perf_counter(), frame_index=captures - 1)
        except BaseException as exc:  # surface hardware/permission failures to caller
            worker_error.append(exc)
        finally:
            capture_done.set()

    def vision_worker():
        nonlocal encoded
        try:
            while not stop.is_set():
                item = latest_frame.wait_latest(stop, timeout=0.02)
                if item is None:
                    if capture_done.is_set():
                        break
                    continue
                import cv2
                from PIL import Image
                rgb = cv2.cvtColor(item.payload, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                t0 = time.perf_counter()
                feature = adapter.encode_frame(image, task_cache).to(torch.float32)
                _sync(device)
                timings["encode"].append((time.perf_counter() - t0) * 1000)
                window.append(feature, frame_index=item.frame_index,
                              timestamp_ns=int(item.captured_at * 1_000_000_000))
                encoded += 1
        except BaseException as exc:
            worker_error.append(exc)
        finally:
            capture_done.set()

    import threading
    capture_thread = threading.Thread(target=capture_worker, name="act-capture", daemon=True)
    vision_thread = threading.Thread(target=vision_worker, name="act-vision", daemon=True)
    capture_thread.start()
    vision_thread.start()
    next_fast = next_slow = time.perf_counter()
    while not stop.is_set() and time.perf_counter() < deadline:
        tick_start = time.perf_counter()
        now = time.perf_counter()
        latest_ts = window.latest_capture_time()
        if latest_ts is not None:
            timings["feature_age"].append(max(0.0, time.perf_counter() - latest_ts) * 1000)
        if latest_ts is not None and now >= next_slow:
            t0 = time.perf_counter()
            features, valid, _ = window.snapshot()
            history = torch.zeros((1, 72), device=device, dtype=torch.float32)
            with torch.inference_mode():
                slow_pass = core(features, condition, valid_mask=valid,
                                 history_actions=history, run_slow=True)
            _sync(device)
            timings["slow"].append((time.perf_counter() - t0) * 1000)
            condition = core.condition_from_slow_output(slow_pass.slow, condition.mode_id)
            slow_ticks += 1
            while next_slow <= now:
                next_slow += slow_period
        if latest_ts is not None and now >= next_fast:
            t0 = time.perf_counter()
            _, steps = _predict(core, window, condition, device)
            _sync(device)
            timings["fast"].append((time.perf_counter() - t0) * 1000)
            fast_ticks += 1
            print(f"[act:{fast_ticks}] frame={window.snapshot()[2][-1]} "
                  f"condition_intent={int(condition.intent_id[0])} "
                  f"step0={action_step_to_text(steps[0])}")
            while next_fast <= now:
                next_fast += fast_period
        timings["total"].append((time.perf_counter() - tick_start) * 1000)
        # When capture has not produced its first frame yet (or DXGI returns
        # a temporary empty frame), avoid a busy-spin that can starve the
        # capture thread and make the benchmark report zero captures.
        if latest_ts is None:
            time.sleep(0.001)
        remaining = min(next_fast, next_slow, deadline) - time.perf_counter()
        if remaining > 0:
            time.sleep(min(remaining, 0.005))
    stop.set()
    with latest_frame._condition:
        latest_frame._condition.notify_all()
    capture_thread.join(timeout=2)
    vision_thread.join(timeout=2)
    if worker_error:
        raise worker_error[0]
    def summarize(values):
        if not values:
            return {"n": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
        values = sorted(values)
        return {"n": len(values), "mean_ms": statistics.mean(values),
                "p95_ms": values[max(0, int(len(values) * .95) - 1)], "max_ms": max(values)}
    result = {"captures": captures, "encoded": encoded, "fast_ticks": fast_ticks,
              "slow_ticks": slow_ticks, "device": str(device),
              "checkpoint": str(Path(args.checkpoint).resolve()),
              "parent_stage": parent_manifest["stage"],
              "latency": {key: summarize(value) for key, value in timings.items()}}
    print(result)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--init-checkpoint", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--title", default="第五人格")
    p.add_argument("--region", default=None)
    p.add_argument("--output-size", type=parse_output_size, default=None,
                   help="可选下采样尺寸，例如 640,384")
    p.add_argument("--capture-fps", type=int, default=30)
    p.add_argument("--fast-hz", type=float, default=15)
    p.add_argument("--slow-hz", type=float, default=1)
    p.add_argument("--history-frames", type=int, default=3)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--mode", default="standard")
    p.add_argument("--instruction", default="找到密码机，靠近并进入破译")
    p.add_argument("--temporal-dim", type=int, default=256)
    args = p.parse_args(argv)
    if args.region:
        values = tuple(int(v) for v in args.region.split(","))
        if len(values) != 4:
            p.error("--region 需要 left,top,width,height")
        args.region = values
    if args.seconds <= 0 or args.capture_fps <= 0 or args.fast_hz <= 0 or args.slow_hz <= 0:
        p.error("时间和频率参数必须为正数")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
