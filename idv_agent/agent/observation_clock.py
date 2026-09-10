"""CPU-only timing contract for sparse visual decisions.

The capture producer overwrites a single slot. Model inference runs in one
worker; the executor runs on a separate clock. A result can cause at most one
macro, and the next result must include an observation after that macro ends.
This module does not send inputs or make task decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any


@dataclass(frozen=True)
class CapturedObservation:
    sequence: int
    captured_ns: int
    payload: Any


class LatestObservationSlot:
    def __init__(self) -> None:
        self._lock = Lock()
        self._item: CapturedObservation | None = None
        self._last_sequence = -1
        self._last_timestamp = -1
        self.overwritten = 0

    def put(self, item: CapturedObservation) -> None:
        with self._lock:
            if item.sequence <= self._last_sequence or item.captured_ns <= self._last_timestamp:
                raise ValueError("capture sequence/timestamp must increase")
            self.overwritten += int(self._item is not None)
            self._item = item
            self._last_sequence = item.sequence
            self._last_timestamp = item.captured_ns

    def take(self) -> CapturedObservation | None:
        with self._lock:
            item, self._item = self._item, None
            return item


@dataclass
class ObservationActionClock:
    """Owned exclusively by the executor thread; all times are monotonic ns."""
    action_ns: int = 200_000_000
    max_age_ns: int = 400_000_000
    consumed_sequence: int = -1
    action_end_ns: int = -1

    def __post_init__(self) -> None:
        if self.action_ns <= 0 or self.max_age_ns <= 0:
            raise ValueError("durations must be positive")

    def accept(self, observation: CapturedObservation, *, ready_ns: int, now_ns: int) -> bool:
        if not observation.captured_ns <= ready_ns <= now_ns:
            raise ValueError("capture <= inference ready <= executor now required")
        if observation.sequence <= self.consumed_sequence:
            return False
        if now_ns - observation.captured_ns > self.max_age_ns:
            return False
        if observation.captured_ns <= self.action_end_ns or now_ns < self.action_end_ns:
            return False
        self.consumed_sequence = observation.sequence
        self.action_end_ns = now_ns + self.action_ns
        return True

    def release_due(self, now_ns: int) -> bool:
        return self.action_end_ns >= 0 and now_ns >= self.action_end_ns
