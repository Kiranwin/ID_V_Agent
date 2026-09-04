"""Execute VLA-v5 macro action chunks with duration-aware key state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Iterable
from collections import deque

import torch

from idv_agent.agent.action_decoder import Command
from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP, SurvivorKeymap
from idv_agent.vla.action_chunk import (
    BUTTON_NAMES, CAMERA_BUCKETS, MACRO_FRAMES, MOVE_DIRECTIONS, camera_command_pixels,
)


_MOVE_KEYS = {
    0: (), 1: ("move_forward",), 2: ("move_forward", "move_right"),
    3: ("move_right",), 4: ("move_back", "move_right"),
    5: ("move_back",), 6: ("move_back", "move_left"),
    7: ("move_left",), 8: ("move_forward", "move_left"),
}
_BUTTON_ATTRS = ("interact", "vault", "item", "heal", "sprint", "crouch")


@dataclass
class _ActiveStep:
    step: object
    remaining_s: float


class ACTActionChunkExecutor:
    """Stateful executor for v5 chunks.

    ``send`` receives a list of :class:`Command` objects.  The default
    callback is intentionally absent: callers must wire an ActionExecutor and
    choose its dry-run/send-input policy explicitly.
    """

    def __init__(self, send: Callable[[Iterable[Command]], None], *,
                 keymap: SurvivorKeymap = DEFAULT_SURVIVOR_KEYMAP,
                 capture_fps: float = 30.0, tick_hz: float = 15.0,
                 dry_run: bool = True):
        if float(capture_fps) != 30.0:
            raise ValueError("v5 ACT 必须使用 capture_fps=30")
        if tick_hz <= 0:
            raise ValueError("tick_hz 必须为正数")
        self.send = send
        self.keymap = keymap
        self.capture_fps = float(capture_fps)
        self.tick_period = 1.0 / float(tick_hz)
        self.dry_run = bool(dry_run)
        self._queue: list[object] = []
        self._active: _ActiveStep | None = None
        self._held: set[str] = set()
        self._virtual_time = 0.0
        self._history: deque[object] = deque(maxlen=8)

    @property
    def active(self) -> bool:
        return self._active is not None or bool(self._queue)

    def set_send(self, send: Callable[[Iterable[Command]], None]) -> None:
        """Attach/replace the physical or dry-run command sink."""
        self.send = send

    def submit(self, steps: Iterable[object]) -> None:
        incoming = list(steps)
        for step in incoming:
            duration = int(getattr(step, "duration_frames", 0))
            move = int(getattr(step, "move_dir", -1))
            if duration != MACRO_FRAMES:
                raise ValueError(f"v5 duration_frames 必须固定为 {MACRO_FRAMES}")
            if not 0 <= move < len(MOVE_DIRECTIONS):
                raise ValueError("move_dir 超出 v5 范围")
            buttons = tuple(getattr(step, "buttons", ()))
            if len(buttons) != len(BUTTON_NAMES) or any(v not in (0, 1) for v in buttons):
                raise ValueError("buttons 必须是六个 0/1 值")
            if int(getattr(step, "camera_dx", 0)) not in CAMERA_BUCKETS or int(getattr(step, "camera_dy", 0)) not in CAMERA_BUCKETS:
                raise ValueError("camera 桶值无效")
        self._queue.extend(incoming)

    def _target(self, step: object) -> tuple[set[str], float, float]:
        keys = {getattr(self.keymap, attr) for attr in _MOVE_KEYS[int(step.move_dir)]}
        for pressed, attr in zip(step.buttons, _BUTTON_ATTRS):
            if pressed:
                value = getattr(self.keymap, attr, None)
                if value:
                    keys.add(value)
        dx = camera_command_pixels(int(step.camera_dx))
        dy = camera_command_pixels(int(step.camera_dy))
        return keys, dx, dy

    def _apply(self, step: object) -> None:
        target, dx, dy = self._target(step)
        commands = [Command("release", code=code) for code in sorted(self._held - target)]
        commands.extend(Command("press", code=code) for code in sorted(target - self._held))
        if dx or dy:
            commands.append(Command("mouse_move", dx_px=dx, dy_px=dy))
        if commands:
            self.send(commands)
        self._held = target
        self._history.append(step)

    @property
    def history_actions(self) -> tuple[object, ...]:
        """Recently applied macro steps, oldest first (up to eight)."""
        return tuple(self._history)

    def history_tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        """Encode executed history using the v5 9-field action representation."""
        values: list[float] = []
        for step in self._history:
            values.extend((float(step.move_dir), float(CAMERA_BUCKETS.index(step.camera_dx)),
                           float(CAMERA_BUCKETS.index(step.camera_dy))))
            values.extend(float(v) for v in step.buttons)
        values.extend([0.0] * (8 * 9 - len(values)))
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(1, -1)

    def tick(self, dt_s: float | None = None) -> bool:
        """Advance execution by one decision tick; return whether active."""
        dt = self.tick_period if dt_s is None else float(dt_s)
        if dt <= 0:
            raise ValueError("dt_s 必须为正数")
        self._virtual_time += dt
        if self._active is None and self._queue:
            step = self._queue.pop(0)
            self._active = _ActiveStep(step, int(step.duration_frames) / self.capture_fps)
            self._apply(step)
        if self._active is not None:
            self._active.remaining_s -= dt
            while self._active is not None and self._active.remaining_s <= 1e-9:
                if self._queue:
                    step = self._queue.pop(0)
                    self._active = _ActiveStep(step, int(step.duration_frames) / self.capture_fps)
                    self._apply(step)
                else:
                    self._active = None
                    self._apply(type("Stop", (), {"move_dir": 0, "camera_dx": 0,
                                                   "camera_dy": 0, "buttons": (0,) * 6})())
        return self.active

    def shutdown(self) -> None:
        self._queue.clear()
        self._active = None
        if self._held:
            self.send([Command("release", code=code) for code in sorted(self._held)])
            self._held.clear()
