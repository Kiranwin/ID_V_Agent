"""Fixed-rate ACT chunk scheduler with stale-feature safety stop."""

from __future__ import annotations

from typing import Callable, Iterable

from idv_agent.agent.act_action_executor import ACTActionChunkExecutor


class ACTChunkScheduler:
    def __init__(self, *, predict: Callable[[object], Iterable[object]],
                 executor: ACTActionChunkExecutor, fast_hz: float = 15.0,
                 max_feature_age_s: float = 0.5):
        if fast_hz <= 0 or max_feature_age_s <= 0:
            raise ValueError("fast_hz/max_feature_age_s 必须为正数")
        self.predict = predict
        self.executor = executor
        self.period = 1.0 / float(fast_hz)
        self.max_feature_age_s = float(max_feature_age_s)
        self.next_tick = 0.0
        self._feature = None
        self._feature_timestamp: float | None = None
        self._pending_steps: list[object] | None = None

    def update_feature(self, feature: object, *, timestamp: float) -> None:
        self._feature = feature
        self._feature_timestamp = float(timestamp)

    def tick(self, *, now: float) -> bool:
        now = float(now)
        self.executor.tick(dt_s=self.period)
        if self._feature_timestamp is None or now - self._feature_timestamp > self.max_feature_age_s:
            self._pending_steps = None
            self.executor.shutdown()
            self.next_tick = now + self.period
            return False
        if now >= self.next_tick:
            self.next_tick = now + self.period
            # Prediction cadence is independent from execution cadence.  Keep
            # only the newest not-yet-started block while the current block is
            # running; this avoids both mid-block discontinuities and an
            # unbounded queue when inference is faster than execution.
            self._pending_steps = list(self.predict(self._feature))
        if not self.executor.active and self._pending_steps is not None:
            self.executor.submit(self._pending_steps)
            self._pending_steps = None
        return self.executor.active

    def shutdown(self) -> None:
        self._pending_steps = None
        self.executor.shutdown()
