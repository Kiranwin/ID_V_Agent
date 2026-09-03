"""Runtime ACT policy: cached visual features, slow condition, and chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from idv_agent.agent.act_action_executor import ACTActionChunkExecutor
from idv_agent.agent.act_scheduler import ACTChunkScheduler
from idv_agent.model.fast_slow_vla import SharedFastSlowVLA, SlowCondition
from idv_agent.model.temporal import FrameFeatureCache, TaskConditionCache
from idv_agent.vla.action_chunk import BUTTON_NAMES, CAMERA_BUCKETS, INTENTS, MOVE_DIRECTIONS
from idv_agent.configs.game_mode import GAME_MODE_CHOICES
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture


@dataclass(frozen=True)
class ACTActionStep:
    move_dir: int
    camera_dx: int
    camera_dy: int
    buttons: tuple[int, ...]
    duration_frames: int


class ACTPolicy:
    """A stateful inference policy for one ACT episode.

    The backbone is called once per observed frame.  Both heads consume the
    same rolling cache; fast predictions are fixed-rate and never interrupt an
    active chunk.  ``send`` is normally ``ActionExecutor.execute``.
    """

    def __init__(self, adapter: Any, core: SharedFastSlowVLA, *, instruction: str,
                 mode: str = "standard", device: torch.device | str = "cpu",
                 send: Callable[[Iterable[Any]], None] | None = None,
                 capture_fps: float = 30.0, fast_hz: float = 15.0,
                 slow_hz: float = 1.0, history_frames: int = 3,
                 max_feature_age_s: float = 0.5, cam_pixel_scale: float = 120.0):
        if mode not in GAME_MODE_CHOICES:
            raise ValueError(f"未知模式: {mode}")
        if not instruction.strip():
            raise ValueError("instruction 不能为空")
        if history_frames < 3:
            raise ValueError("history_frames 必须至少为 3")
        self.adapter = adapter
        self.core = core
        self.device = torch.device(device)
        self.mode = mode
        self.mode_id = GAME_MODE_CHOICES.index(mode)
        self.instruction = instruction
        self.history_frames = int(history_frames)
        self.task_cache: TaskConditionCache = adapter.encode_task_once(
            instruction, mode, task_id=f"act:{id(self)}")
        self.features = FrameFeatureCache(max_length=max(8, self.history_frames))
        # ACT cold start is travel/observe, unlike the generic model default.
        self.condition = core.initial_condition(1, device=self.device,
                                                mode_id=self.mode_id,
                                                intent_id=INTENTS.index("travel"))
        self.history_actions = torch.zeros((1, 72), dtype=torch.float32, device=self.device)
        self._last_slow = 0.0
        self._prediction_count = 0
        self.slow_period = 1.0 / float(slow_hz)
        self.executor = ACTActionChunkExecutor(
            send=send or (lambda _commands: None), capture_fps=capture_fps,
            tick_hz=fast_hz, cam_pixel_scale=cam_pixel_scale)
        self.scheduler = ACTChunkScheduler(
            predict=self._predict_chunk, executor=self.executor, fast_hz=fast_hz,
            max_feature_age_s=max_feature_age_s)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, model_path: str | Path | None,
                        init_checkpoint: str | Path, instruction: str,
                        mode: str = "standard", device: torch.device | str = "cuda",
                        **kwargs) -> "ACTPolicy":
        """Load an M3_ACT checkpoint with its M2_VG/M1_WK inheritance chain."""
        from idv_agent.scripts.benchmark_act_realtime import _load_model
        device_obj = torch.device(device)
        class Args:
            pass
        args = Args()
        args.model_path = str(model_path) if model_path else ""
        args.init_checkpoint = str(init_checkpoint)
        args.checkpoint = str(checkpoint)
        args.temporal_dim = int(kwargs.pop("temporal_dim", 256))
        adapter, core, _manifest = _load_model(args, device_obj)
        return cls(adapter, core, instruction=instruction, mode=mode,
                   device=device_obj, **kwargs)

    @property
    def ready(self) -> bool:
        return len(self.features) >= self.history_frames

    def observe(self, image: Any, *, frame_index: int, timestamp_ns: int) -> bool:
        """Encode and cache one frame; return whether the history is warm."""
        with torch.inference_mode():
            feature = self.adapter.encode_frame(image, self.task_cache)
        feature = feature.to(device=self.device, dtype=torch.float32).reshape(-1)
        self.features.append(frame_index, timestamp_ns, feature)
        return self.ready

    def _run_slow_if_due(self, now: float) -> None:
        if not self.ready or now - self._last_slow < self.slow_period:
            return
        values, timestamps, valid = self.features.window(self.history_frames)
        deltas = (timestamps - timestamps[-1]).to(values.dtype) / 1_000_000_000.0
        with torch.inference_mode():
            output = self.core(values.unsqueeze(0), self.condition,
                               valid_mask=valid.unsqueeze(0), time_deltas=deltas.unsqueeze(0),
                               history_actions=self.history_actions, run_slow=True)
        self.condition = self.core.condition_from_slow_output(output.slow,
                                                               self.condition.mode_id)
        self._last_slow = now
        print(f"[act:slow] frame={self.features.latest().frame_index} "
              f"intent={INTENTS[int(self.condition.intent_id[0])]} "
              f"subgoal={int(self.condition.subgoal_id[0])}")

    def _predict_chunk(self, _feature: object) -> list[ACTActionStep]:
        values, timestamps, valid = self.features.window(self.history_frames)
        deltas = (timestamps - timestamps[-1]).to(values.dtype) / 1_000_000_000.0
        with torch.inference_mode():
            output = self.core(values.unsqueeze(0), self.condition,
                               valid_mask=valid.unsqueeze(0), time_deltas=deltas.unsqueeze(0),
                               history_actions=self.history_actions, run_slow=False)
        fast = output.fast
        moves = fast.move_logits.argmax(-1)[0].tolist()
        dxs = fast.camera_dx_logits.argmax(-1)[0].tolist()
        dys = fast.camera_dy_logits.argmax(-1)[0].tolist()
        buttons = (fast.button_logits.sigmoid() >= 0.5)[0]
        durations = fast.duration[0].round().clamp(1, 30).to(torch.int64).tolist()
        steps = [ACTActionStep(int(move), CAMERA_BUCKETS[int(dxs[i])], CAMERA_BUCKETS[int(dys[i])],
                               tuple(int(v) for v in buttons[i].tolist()), int(durations[i]))
                 for i, move in enumerate(moves)]
        latest = self.features.latest()
        first = steps[0]
        pressed = ",".join(name for name, value in zip(BUTTON_NAMES, first.buttons) if value) or "-"
        age_ms = max(0.0, time.perf_counter() - latest.timestamp_ns / 1_000_000_000.0) * 1000
        self._prediction_count += 1
        fingerprint = float(values[-1].detach().float().mean().cpu())
        print(f"[act] pred={self._prediction_count} frame={latest.frame_index} "
              f"intent={INTENTS[int(self.condition.intent_id[0])]} "
              f"move={MOVE_DIRECTIONS[first.move_dir]}({first.move_dir}) "
              f"camera=({first.camera_dx},{first.camera_dy}) "
              f"buttons={pressed} duration={first.duration_frames} "
              f"feature_mean={fingerprint:.5f} feature_age_ms={age_ms:.1f}")
        return steps

    def tick(self, *, now: float | None = None) -> bool:
        """Run due slow/fast ticks. ``now`` is a monotonic seconds timestamp."""
        if not self.ready:
            return False
        current = time.perf_counter() if now is None else float(now)
        self._run_slow_if_due(current)
        latest = self.features.latest()
        if latest is None:
            return False
        self.scheduler.update_feature(latest, timestamp=latest.timestamp_ns / 1_000_000_000.0)
        active = self.scheduler.tick(now=current)
        self.history_actions = self.executor.history_tensor(device=self.device)
        return active

    def shutdown(self) -> None:
        self.scheduler.shutdown()

    def run_capture(self, capture_config: CaptureConfig, *, duration_s: float,
                    max_frames: int | None = None) -> int:
        """Run the ACT loop on live capture; input policy is defined by ``send``."""
        if duration_s <= 0:
            raise ValueError("duration_s 必须为正数")
        import cv2
        from PIL import Image
        frame_interval = 1.0 / self.executor.capture_fps
        deadline = time.perf_counter() + float(duration_s)
        frame_count = 0
        next_capture = time.perf_counter()
        try:
            with ScreenCapture(capture_config) as capture:
                while time.perf_counter() < deadline:
                    if max_frames is not None and frame_count >= max_frames:
                        break
                    now = time.perf_counter()
                    if now < next_capture:
                        time.sleep(min(next_capture - now, 0.005))
                        continue
                    next_capture += frame_interval
                    frame = capture.grab()
                    if frame is None:
                        continue
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp_ns = time.perf_counter_ns()
                    self.observe(Image.fromarray(rgb), frame_index=frame_count,
                                  timestamp_ns=timestamp_ns)
                    self.tick(now=timestamp_ns / 1_000_000_000.0)
                    frame_count += 1
        finally:
            self.shutdown()
        return frame_count
