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

    def update_feature(self, feature: object, *, timestamp: float) -> None:
        self._feature = feature
        self._feature_timestamp = float(timestamp)

    def tick(self, *, now: float) -> bool:
        now = float(now)
        self.executor.tick(dt_s=self.period)
        if self._feature_timestamp is None or now - self._feature_timestamp > self.max_feature_age_s:
            self.executor.shutdown()
            self.next_tick = now + self.period
            return False
        if now < self.next_tick or self.executor.active:
            return self.executor.active
        self.next_tick = now + self.period
        self.executor.submit(self.predict(self._feature))
        return self.executor.active

    def shutdown(self) -> None:
        self.executor.shutdown()
