"""部署策略抽象。

当前只保留规则策略作为 VLA 尚未接入时的安全兜底。VLA 原生策略将使用
独立的动作块接口，不再复用旧的逐帧 BC 输出。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PolicyOutput:
    category_id: int
    continuous: Tuple[float, float, float, float]   # (move_x, move_y, cam_dx, cam_dy)
    confidence: float = 0.0
    raw: object = None


class Policy:
    """策略接口。decide(frame, state) -> PolicyOutput。frame 为 BGR ndarray。"""

    def decide(self, frame, state) -> PolicyOutput:
        raise NotImplementedError


class RulePolicy(Policy):
    """规则安全兜底（律师版「找→走→破译」）。VLA 策略使用独立动作块接口。"""

    def __init__(self, rule_agent):
        self.rule_agent = rule_agent

    def decide(self, frame, state) -> PolicyOutput:
        # RealtimeAgent 的 state 包含 intent/skeleton/memory/spatial；规则器
        # 只消费结构化感知状态，避免把嵌套 spatial 误当成空输入。
        spatial = state.get("spatial", state) if isinstance(state, dict) else state
        cat, cont = self.rule_agent.decide(spatial)
        return PolicyOutput(category_id=int(cat), continuous=tuple(cont))
