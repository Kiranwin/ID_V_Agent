"""Runtime m28 ACT policy: cached visual state and causal h0 chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from idv_agent.agent.act_action_executor import ACTActionChunkExecutor
from idv_agent.agent.act_scheduler import ACTChunkScheduler
from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.temporal import FrameFeatureCache, TaskConditionCache
from idv_agent.vla.action_chunk import BUTTON_NAMES, CAMERA_BUCKETS, INTENTS, MACRO_FRAMES, MOVE_DIRECTIONS
from idv_agent.configs.game_mode import GAME_MODE_CHOICES
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture
from idv_agent.agent.mvp_closed_loop import ClosedLoopTraceWriter


@dataclass(frozen=True)
class ACTActionStep:
    move_dir: int
    camera_dx: int
    camera_dy: int
    buttons: tuple[int, ...]
    duration_frames: int


class ACTPolicy:
    """A stateful inference policy for one ACT episode.

    The backbone is called once per observed frame.  m28 derives state and the
    joint action from the same rolling visual cache; only h0 is submitted and
    never interrupts an active chunk.  ``send`` is normally
    ``ActionExecutor.execute``.
    """

    def __init__(self, adapter: Any, core: SharedFastSlowVLA, *, instruction: str,
                 mode: str = "standard", device: torch.device | str = "cpu",
                 send: Callable[[Iterable[Any]], None] | None = None,
                 capture_fps: float = 30.0, fast_hz: float = 15.0,
                 slow_hz: float = 1.0, history_frames: int = 8, history_stride: int = 3,
                 max_feature_age_s: float = 0.5,
                 use_time_deltas: bool = False, zero_history: bool = False,
                 closed_loop_trace: str | Path | None = None,
                 closed_loop_episode_id: str | None = None,
                 closed_loop_observer: Callable[[Any], dict[str, Any]] | None = None):
        if mode not in GAME_MODE_CHOICES:
            raise ValueError(f"未知模式: {mode}")
        if not instruction.strip():
            raise ValueError("instruction 不能为空")
        if history_frames != 8:
            raise ValueError("当前 v5 ACT 必须使用 history_frames=8")
        if history_stride != 3:
            raise ValueError("当前 v5 ACT 必须使用 history_stride=3")
        if float(capture_fps) != 30.0:
            raise ValueError("当前 v5 ACT 必须使用 capture_fps=30 以保持 duration_frames 的时间语义")
        if use_time_deltas:
            raise ValueError("当前 v5 ACT 未训练 time_deltas，必须保持关闭")
        self.adapter = adapter
        self.core = core
        self.device = torch.device(device)
        self.mode = mode
        self.mode_id = GAME_MODE_CHOICES.index(mode)
        self.instruction = instruction
        self.history_frames = int(history_frames)
        self.history_stride = int(history_stride)
        self.use_time_deltas = bool(use_time_deltas)
        self.zero_history = bool(zero_history)
        self.closed_loop_observer = closed_loop_observer
        self.closed_loop_trace = (ClosedLoopTraceWriter(closed_loop_trace,
            episode_id=closed_loop_episode_id or f"act-{id(self)}")
            if closed_loop_trace is not None else None)
        self._latest_external_observation = None
        self._pending_closed_loop = None
        self._pending_closed_loop_action = None
        self._last_observed_frame = -1
        self.interact_event_threshold = float(getattr(core, "interact_event_threshold", 0.0))
        if not torch.isfinite(torch.tensor(self.interact_event_threshold)):
            raise ValueError("core.interact_event_threshold 必须是有限数")
        self.task_cache: TaskConditionCache = adapter.encode_task_once(
            instruction, mode, task_id=f"act:{id(self)}")
        self.window_span = (self.history_frames - 1) * self.history_stride + 1
        self.features = FrameFeatureCache(max_length=self.window_span)
        self.visual_features = FrameFeatureCache(max_length=self.window_span)
        # Kept solely for the core's legacy temporal diagnostics.  m28's
        # deployed StateDecisionExpert receives task_features directly and
        # does not read or update this condition.
        self.condition = core.initial_condition(1, device=self.device, mode_id=self.mode_id,
                                                intent_id=INTENTS.index("travel"))
        self.history_actions = torch.zeros((1, 72), dtype=torch.float32, device=self.device)
        self._prediction_count = 0
        self.last_run_stats: dict[str, Any] = {}
        if slow_hz <= 0:
            raise ValueError("slow_hz 必须为正数")
        self.executor = ACTActionChunkExecutor(
            send=send or (lambda _commands: None), capture_fps=capture_fps,
            tick_hz=fast_hz, on_step_applied=self._on_step_applied)
        self.scheduler = ACTChunkScheduler(
            predict=self._predict_chunk, executor=self.executor, fast_hz=fast_hz,
            max_feature_age_s=max_feature_age_s)

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, *, model_path: str | Path | None,
                        init_checkpoint: str | Path | None = None, instruction: str = "",
                        mode: str = "standard", device: torch.device | str = "cuda",
                        **kwargs) -> "ACTPolicy":
        """Load an M3_ACT checkpoint from the current base-without-M2 protocol."""
        from idv_agent.scripts.benchmark_act_realtime import _load_model
        device_obj = torch.device(device)
        class Args:
            pass
        args = Args()
        args.model_path = str(model_path) if model_path else ""
        args.checkpoint = str(checkpoint)
        args.temporal_dim = int(kwargs.pop("temporal_dim", 256))
        adapter, core, _manifest = _load_model(args, device_obj)
        return cls(adapter, core, instruction=instruction, mode=mode,
                   device=device_obj, **kwargs)

    @property
    def ready(self) -> bool:
        return len(self.features) >= self.window_span

    def observe(self, image: Any, *, frame_index: int, timestamp_ns: int) -> bool:
        """Encode and cache one frame; return whether the history is warm."""
        if self.closed_loop_observer is not None:
            self._latest_external_observation = dict(self.closed_loop_observer(image))
        self._last_observed_frame = int(frame_index)
        self._flush_closed_loop(frame_index)
        with torch.inference_mode():
            encode_pair = getattr(self.adapter, "encode_frames_with_visual", None)
            if encode_pair is not None:
                feature, visual_feature = encode_pair([image], self.task_cache)
                feature, visual_feature = feature[0], visual_feature[0]
            else:
                feature = self.adapter.encode_frame(image, self.task_cache)
                visual_feature = feature
        feature = feature.to(device=self.device, dtype=torch.float32).reshape(-1)
        visual_feature = visual_feature.to(device=self.device, dtype=torch.float32).reshape(-1)
        self.features.append(frame_index, timestamp_ns, feature)
        self.visual_features.append(frame_index, timestamp_ns, visual_feature)
        return self.ready

    def _on_step_applied(self, step: object) -> None:
        if self.closed_loop_trace is None or self._latest_external_observation is None:
            return
        buttons = tuple(getattr(step, "buttons", (0,) * 6))
        self._pending_closed_loop = (self._last_observed_frame,
                                     dict(self._latest_external_observation))
        self._pending_closed_loop_action = {
            "move_dir": int(getattr(step, "move_dir", 0)),
            "camera_dx": int(getattr(step, "camera_dx", 0)),
            "camera_dy": int(getattr(step, "camera_dy", 0)),
            "buttons": [int(v) for v in buttons],
            "interact": int(buttons[0]) if buttons else 0,
        }

    def _flush_closed_loop(self, frame_index: int) -> None:
        if self.closed_loop_trace is None or self._pending_closed_loop is None:
            return
        if self.closed_loop_observer is None:
            return
        start_frame, before = self._pending_closed_loop
        if int(frame_index) < int(start_frame) + MACRO_FRAMES:
            return
        self.closed_loop_trace.append(before=before,
                                     action=self._pending_closed_loop_action or {},
                                     after=dict(self._latest_external_observation or {}))
        self._pending_closed_loop = None
        self._pending_closed_loop_action = None

    def _predict_chunk(self, _feature: object) -> list[ACTActionStep]:
        values, deltas, valid = self._temporal_inputs()
        visual_values, _, visual_valid = self._visual_temporal_inputs()
        history = torch.zeros_like(self.history_actions) if self.zero_history else self.history_actions
        with torch.inference_mode():
            output = self.core(values.unsqueeze(0), self.condition,
                               valid_mask=valid.unsqueeze(0), time_deltas=None if deltas is None else deltas.unsqueeze(0),
                               visual_frame_features=visual_values.unsqueeze(0),
                               history_actions=history, run_slow=False)
        fast = output.fast
        state_output = output.visual
        if state_output is None:
            raise RuntimeError("m28 ACT 需要 StateDecisionExpert 输出")
        moves = fast.move_logits.argmax(-1)[0].tolist()
        dxs = fast.camera_dx_logits.argmax(-1)[0].tolist()
        dys = fast.camera_dy_logits.argmax(-1)[0].tolist()
        buttons = fast.button_predictions(event_threshold=self.interact_event_threshold)[0]
        # v5 labels and executor history are fixed six-frame macro actions.
        # Do not expose the untrained variable-duration head to deployment.
        durations = [MACRO_FRAMES] * len(moves)
        steps = [ACTActionStep(int(move), CAMERA_BUCKETS[int(dxs[i])], CAMERA_BUCKETS[int(dys[i])],
                               tuple(int(v) for v in buttons[i].tolist()), int(durations[i]))
                 for i, move in enumerate(moves)]
        latest = self.features.latest()
        first = steps[0]
        pressed = ",".join(name for name, value in zip(BUTTON_NAMES, first.buttons) if value) or "-"
        age_ms = max(0.0, time.perf_counter() - latest.timestamp_ns / 1_000_000_000.0) * 1000
        self._prediction_count += 1
        fingerprint = float(values[-1].detach().float().mean().cpu())
        chunk_text = ";".join(f"{MOVE_DIRECTIONS[s.move_dir]}/{s.camera_dx},{s.camera_dy}/"
                              f"{','.join(n for n,v in zip(BUTTON_NAMES,s.buttons) if v) or '-'}@{s.duration_frames}"
                              for s in steps)
        intent = INTENTS[int(state_output.slow.intent_id[0])]
        control = state_output.camera_control
        phase = int(control.phase_logits.argmax(dim=-1)[0]) if control is not None else -1
        steering = int(control.steering_logits.argmax(dim=-1)[0]) if control is not None else -1
        print(f"[act] pred={self._prediction_count} frame={latest.frame_index} "
              f"intent={intent} phase={phase} steering={steering} "
              f"move={MOVE_DIRECTIONS[first.move_dir]}({first.move_dir}) "
              f"camera=({first.camera_dx},{first.camera_dy}) "
              f"buttons={pressed} duration={first.duration_frames} "
              f"feature_mean={fingerprint:.5f} feature_age_ms={age_ms:.1f} chunk={chunk_text}")
        # m25 re-observes after the causal first macro action.  The remaining
        # logits remain an internal prediction horizon only; submitting them
        # would execute actions conditioned on an obsolete image window.
        return steps[:1]

    def _temporal_inputs(self):
        values, timestamps, valid = self.features.window(self.history_frames, self.history_stride)
        deltas = None
        # v5 was trained with the temporal encoder's zero-delta default.
        # Keep this explicit so future protocol changes cannot silently alter
        # the deployed input distribution.
        return values, deltas, valid

    def _visual_temporal_inputs(self):
        values, timestamps, valid = self.visual_features.window(self.history_frames, self.history_stride)
        return values, None, valid

    def tick(self, *, now: float | None = None) -> bool:
        """Run the due m28 state/planner tick. ``now`` is monotonic seconds."""
        if not self.ready:
            return False
        current = time.perf_counter() if now is None else float(now)
        latest = self.features.latest()
        if latest is None:
            return False
        self.scheduler.update_feature(latest, timestamp=latest.timestamp_ns / 1_000_000_000.0)
        active = self.scheduler.tick(now=current)
        self.history_actions = (torch.zeros_like(self.history_actions)
                                if self.zero_history else self.executor.history_tensor(device=self.device))
        return active

    def shutdown(self) -> None:
        self.scheduler.shutdown()
        if self.closed_loop_trace is not None:
            self.closed_loop_trace.close()
            self.closed_loop_trace = None

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
        empty_grabs = 0
        reused_frames = 0
        started_at = time.perf_counter()
        last_frame = None
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
                        empty_grabs += 1
                        if last_frame is None:
                            continue
                        frame = last_frame
                        reused_frames += 1
                    else:
                        last_frame = frame
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    timestamp_ns = time.perf_counter_ns()
                    self.observe(Image.fromarray(rgb), frame_index=frame_count,
                                  timestamp_ns=timestamp_ns)
                    # The feature timestamp is the capture instant; tick at
                    # the post-encode monotonic time so freshness and action
                    # scheduling include inference latency.
                    self.tick(now=time.perf_counter())
                    frame_count += 1
        finally:
            self.shutdown()
            self.last_run_stats = {
                "requested_duration_s": float(duration_s),
                "elapsed_s": time.perf_counter() - started_at,
                "processed_frames": int(frame_count),
                "predictions": int(self._prediction_count),
                "empty_grabs": int(empty_grabs),
                "reused_frames": int(reused_frames),
                "device": str(self.device),
            }
        return frame_count
