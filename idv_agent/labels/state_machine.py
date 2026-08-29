"""从原始事件流重建任意时刻的键鼠状态（状态重放）+ 第五人格破译隐式状态机。

两个视图：
1. 流式状态机 `StateReplay`：按时间推进得到每帧的 FrameState（当前按住键、持续时长、
   刚抬起键、鼠标增量），是 TAP vs HOLD 分类的关键。
2. 破译状态 `DecodeStateTracker`：跟踪「按 Q 自动破译」的隐式状态，并标记校准帧（P2 修复）。

关键（P2 修复）：破译中按 Space 不是翻窗而是**校准**。`DecodeStatus.calibration` 字段
记录该帧是否为校准，供后续视觉辅助标注补救——不重录也可能学到校准。
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from idv_agent.configs.schema import ExtractParams


@dataclass
class PressInterval:
    code: str
    press_ts: int
    release_ts: int  # 按键未抬起则 = end_ts_ns + 哨兵

    @property
    def duration_ms(self) -> float:
        return (self.release_ts - self.press_ts) / 1e6

    def contains(self, ts: int) -> bool:
        return self.press_ts <= ts <= self.release_ts


def build_press_intervals(events: pd.DataFrame, end_ts_ns: Optional[int] = None) -> List[PressInterval]:
    """把 down/up 事件配对成区间。

    - 同 code 重复 down 未 up（OS auto-repeat）：保留首次 press_ts。
    - up 找不到 down：跳过。
    - 录制结束仍按住的：release_ts 取 end_ts_ns + 1 哨兵。
    """
    intervals: List[PressInterval] = []
    pending: Dict[str, int] = {}

    sorted_ev = events.sort_values("timestamp_ns").itertuples(index=False)
    for ev in sorted_ev:
        kind = ev.kind
        code = ev.code
        ts = int(ev.timestamp_ns)
        if kind in ("key_down", "mouse_down"):
            if code not in pending:
                pending[code] = ts
        elif kind in ("key_up", "mouse_up"):
            press_ts = pending.pop(code, None)
            if press_ts is not None:
                intervals.append(PressInterval(code=code, press_ts=press_ts, release_ts=ts))

    if pending:
        if end_ts_ns is None:
            end_ts_ns = int(events["timestamp_ns"].max()) if len(events) else 0
        sentinel = end_ts_ns + 1
        for code, press_ts in pending.items():
            intervals.append(PressInterval(code=code, press_ts=press_ts, release_ts=sentinel))

    intervals.sort(key=lambda iv: iv.press_ts)
    return intervals


@dataclass
class FrameState:
    """单帧状态视图。"""
    timestamp_ns: int
    mouse_x: Optional[int]
    mouse_y: Optional[int]
    mouse_dx: float = 0.0
    mouse_dy: float = 0.0
    held_durations_ms: Dict[str, float] = field(default_factory=dict)   # 当前按住的键 → 总时长
    just_released_durations_ms: Dict[str, float] = field(default_factory=dict)  # 本帧刚抬起的键 → 总时长


class StateReplay:
    """按帧时间网格回放每帧 FrameState。"""

    def __init__(self, events: pd.DataFrame, positions: pd.DataFrame, end_ts_ns: Optional[int] = None):
        self.intervals = build_press_intervals(events, end_ts_ns=end_ts_ns)
        self._event_ts = list(map(int, events["timestamp_ns"])) if len(events) else []
        self._release_events = [iv for iv in self.intervals]
        self._positions = positions
        self._pos_ts = list(map(int, positions["timestamp_ns"])) if len(positions) else []
        self._pos_x = list(map(int, positions["x"])) if len(positions) else []
        self._pos_y = list(map(int, positions["y"])) if len(positions) else []
        # 预计算每个 interval 的 (press, release)；用于二分查询
        self._release_sorted = sorted(self.intervals, key=lambda iv: iv.release_ts)

    def frame_state(self, ts: int, prev_ts: Optional[int] = None) -> FrameState:
        held = {}
        for iv in self.intervals:
            if iv.contains(ts):
                held[iv.code] = iv.duration_ms
        # 刚抬起：release_ts 落在 (prev_ts, ts] 区间内
        released: Dict[str, float] = {}
        lo = 0 if prev_ts is None else prev_ts
        for iv in self._release_sorted:
            if lo < iv.release_ts <= ts:
                released[iv.code] = iv.duration_ms

        # 鼠标位置与增量
        if self._pos_ts:
            idx = max(0, bisect_right(self._pos_ts, ts) - 1)
            x, y = self._pos_x[idx], self._pos_y[idx]
            dx = dy = 0.0
            if prev_ts is not None:
                pidx = max(0, bisect_right(self._pos_ts, prev_ts) - 1)
                dx = float(x - self._pos_x[pidx])
                dy = float(y - self._pos_y[pidx])
        else:
            x = y = None
            dx = dy = 0.0

        return FrameState(
            timestamp_ns=ts,
            mouse_x=x, mouse_y=y,
            mouse_dx=dx, mouse_dy=dy,
            held_durations_ms=held,
            just_released_durations_ms=released,
        )


@dataclass
class DecodeStatus:
    active: bool = False
    start_ts: int = 0
    calibration: bool = False   # P2：本帧是否处于校准（破译中按 Space）


class DecodeStateTracker:
    """第五人格「按 Q 自动破译」隐式状态机。

    规则（docs/03）：
    1. Q press → 进入破译态，重置 start。
    2. WASD / SKILL / ITEM / MAP / EMOTE press → 退出。
    3. 破译中按 Space → 校准（保持破译态，标记 calibration，不视为翻窗，P2）。
    4. 破译时长 > max_decode_duration_s → 强制退出（Q 误触保护）。
    """

    _EXIT_KEYS = frozenset({"key:w", "key:a", "key:s", "key:d",
                            "key:f", "key:e", "key:1", "key:2", "key:3", "key:4",
                            "key:tab", "key:v"})

    def __init__(self, keymap_interact: str = "key:q",
                 keymap_vault: str = "key:space",
                 params: Optional[ExtractParams] = None):
        self.interact_key = keymap_interact
        self.vault_key = keymap_vault
        self.params = params or ExtractParams()
        self._status = DecodeStatus()

    def status(self) -> DecodeStatus:
        return self._status

    def on_frame(self, ts: int, frame: FrameState) -> DecodeStatus:
        """逐帧推进状态机。frame 携带本帧按键（按住/刚抬起）信息。"""
        st = self._status

        # 退出条件：按下 WASD/技能/道具/地图/表情
        if st.active and self._EXIT_KEYS.intersection(frame.held_durations_ms):
            st = DecodeStatus(active=False)
            self._status = st
            return st

        # 破译超时强制退出
        if st.active and (ts - st.start_ts) / 1e6 > self.params.max_decode_duration_s * 1000.0:
            st = DecodeStatus(active=False)
            self._status = st
            return st

        # 进入破译：Q 被按下（press 时 held 含 interact_key）
        if not st.active and self.interact_key in frame.held_durations_ms:
            st = DecodeStatus(active=True, start_ts=ts)
            self._status = st
            return st

        # 校准：破译中按 Space（保持破译态，标记校准）
        if st.active and self.vault_key in frame.held_durations_ms:
            st.calibration = True
            self._status = st
        elif st.active:
            st.calibration = False
            self._status = st
        return self._status
