"""对局记忆（P11 架构级修复）。

旧架构无记忆，策略上限锁死在「反射级」——学不出转点（不记得监管者在哪）、
学不出开门时机。本模块给慢层提供外部记忆：监管者最后位置、各机修机进度、
地窖位置、队友状态，作为慢层 VLM 规划的结构化上下文。

当前实现为启发式 + 时间衰减的事件驱动概率地图；M2 后可换轻量贝叶斯滤波。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MemoryEvent:
    kind: str          # "hunter_spot" / "cipher_decode" / "teammate_hooked" / "door_open" / "hunter_attack"
    position: Optional[str] = None   # 简化用编号/标签（"cipher_A" / "hunter_last_0" ...）
    detail: str = ""
    ts: float = field(default_factory=time.time)


class MatchMemory:
    """对局级记忆：事件日志 + 结构化状态缓存。跨局不保留。"""

    def __init__(self, ttl_s: float = 60.0):
        self.ttl_s = ttl_s
        self.events: list[MemoryEvent] = []
        self.hunter_last_position: Optional[str] = None   # 监管者最后目击位置
        self.hunter_last_ts: float = 0.0
        self.cipher_progress: dict[str, float] = {}       # 密码机名 -> 进度 0..1
        self.dungeon_position: Optional[str] = None       # 地窖位置
        self.teammate_hooked: list[str] = field(default_factory=list)
        self.door_open: bool = False

    # ---- 事件 ----
    def add_event(self, event: MemoryEvent) -> None:
        self.events.append(event)
        if len(self.events) > 200:
            self.events = self.events[-200:]
        # 派生状态
        if event.kind == "hunter_spot":
            self.hunter_last_position = event.position
            self.hunter_last_ts = event.ts
        elif event.kind == "cipher_decode":
            if event.position:
                self.cipher_progress[event.position] = float(event.detail or 0.0)
        elif event.kind == "teammate_hooked":
            if event.position:
                self.teammate_hooked.append(event.position)
        elif event.kind == "door_open":
            self.door_open = True
        elif event.kind == "dungeon_seen":
            self.dungeon_position = event.position

    # ---- 查询 ----
    def hunter_threat(self, threshold_s: float = 8.0) -> float:
        """监管者最近目击距今时间；超过 threshold_s 视为威胁衰减为低。返回 0..1 威胁度。"""
        if self.hunter_last_ts == 0:
            return 0.0
        age = time.time() - self.hunter_last_ts
        if age >= threshold_s:
            return 0.0
        return 1.0 - age / threshold_s

    def summary(self) -> dict:
        """结构化为慢层可消费的上下文。"""
        return {
            "hunter_last_position": self.hunter_last_position,
            "hunter_threat": round(self.hunter_threat(), 3),
            "cipher_progress": dict(self.cipher_progress),
            "dungeon_position": self.dungeon_position,
            "teammates_hooked": list(self.teammate_hooked),
            "door_open": self.door_open,
            "n_events": len(self.events),
        }
