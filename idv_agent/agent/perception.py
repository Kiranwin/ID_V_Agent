"""感知层：轻量结构化 + 事件检测（P8 修复入口）。

不做逐帧大模型。负责：
1. HUD 状态提取（占位）：修机进度、技能 CD、校准判定 → 裁剪区域将来接 OCR/颜色阈值。
2. 事件检测：受击、爆点、挂椅、破译完成、开门通电等关键事件触发慢层重规划。

P8 场景：监管者砍断破译是游戏侧状态变化（无按键事件）。`EventDetector` 提供
`on_frame(frame)` 接口，将来接入「受击红屏 / 破译进度条消失」视觉信号，把
「破译被打断」事件抛给慢层/状态机，避免破译状态残留到 90s 超时。

当前为**接口 + 占位实现**，真实视觉识别待 M1 感知层落地。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np


class PerceptionEvent:
    kind: str        # "hit" / "decode_interrupt" / "explosion" / "cipher_done" / "door_power"
    source: str
    ts: float


# 回调签名：(event_kind: str, detail: dict)
EventCallback = Callable[[str, dict], None]


class EventDetector:
    def __init__(self, on_event: Optional[EventCallback] = None):
        self._callbacks: list[EventCallback] = []
        if on_event is not None:
            self._callbacks.append(on_event)
        # 占位状态
        self._prev_hud = {}

    def on_event(self, cb: EventCallback) -> None:
        self._callbacks.append(cb)

    def emit(self, kind: str, detail: dict) -> None:
        for cb in self._callbacks:
            try:
                cb(kind, detail)
            except Exception as e:
                print(f"[perception] event callback error: {e}")

    def on_frame(self, frame: np.ndarray) -> dict:
        """处理一帧，返回本轮检测到的事件列表。占位实现：将来在这里接视觉。

        返回事件列表（可能是多个）。此占位不实际检测，仅保留接口供 M1 落地。
        """
        return []

    def on_decode_progress(self, progress: Optional[float]) -> None:
        """P8：破译进度消失（被打断）时触发 decode_interrupt 事件。"""
        if progress is None or progress > 0.0:
            return
        self.emit("decode_interrupt", {"progress": 0.0})
