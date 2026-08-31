"""规则决策器：求生者（律师）版「找密码机 → 过去 → 破译」闭环兜底（修复 P1）。

输入 spatial_state（来自感知层 SharedState.spatial）：
    {visible, position, distance, blocked, interacting, threat}

输出 (category_id: ActionCategory, continuous: (move_x, move_y, cam_dx, cam_dy))。

律师版与旧黑杰克版的关键差异：
- 类别用标准求生者 ActionCategory（20 类），非黑杰克 schema。
- 增加 threat 处理：感知到监管者/耳鸣时优先跑路（不硬顶破译），体现转点决策。

这是 M1 阶段「规则闭环」的核心：让整个环先转起来（不依赖训练好的模型），
为 M2 的 BC 训练提供游戏内可验证基线。
"""

from __future__ import annotations

import time
from typing import Tuple

from idv_agent.configs.schema import ActionCategory


SCAN_CAM_DX = 0.06        # 找密码机扫视幅度（≈12px/帧 @ 30Hz）
ALIGN_CAM_DX = 0.10       # 对齐目标转视角幅度
ALIGN_FAR_MULT = 1.6      # far_* 额外倍率
TURN_AROUND_DX = 0.15     # 撞墙转身
SCAN_PHASE_DURATION_S = 2.5
THREAT_RUN_DURATION_S = 2.0   # 感知到威胁时跑一段再重新评估


class RuleAgent:
    def __init__(self):
        self._scan_dir = +1
        self._scan_started_at = time.perf_counter()
        self._threat_until = 0.0

    def _scan_dx(self, base: float) -> float:
        if time.perf_counter() - self._scan_started_at >= SCAN_PHASE_DURATION_S:
            self._scan_dir *= -1
            self._scan_started_at = time.perf_counter()
        return base * self._scan_dir

    def decide(self, sp: dict) -> Tuple[int, Tuple[float, float, float, float]]:
        visible = sp.get("visible", "no")
        position = sp.get("position", "none")
        distance = sp.get("distance", "none")
        blocked = sp.get("blocked", "no")
        interacting = sp.get("interacting", "no")
        interact_prompt = sp.get("interact_prompt", "no")
        decoding_state = sp.get("decoding_state", "no")
        threat = sp.get("threat", "no")

        # 威胁（监管者近身/耳鸣）→ 跑路一段时间
        if threat == "yes":
            now = time.perf_counter()
            if now > self._threat_until:
                self._threat_until = now + THREAT_RUN_DURATION_S
            return int(ActionCategory.MOVE), (0, 1.0, 0, 0)

        # 已在破译
        if interacting == "yes" or decoding_state == "yes":
            return int(ActionCategory.INTERACT_HOLD), (0, 0, 0, 0)

        # 看到密码机
        if visible == "yes":
            if (interact_prompt == "yes" and position == "center"):
                return int(ActionCategory.INTERACT_TAP), (0, 0, 0, 0)
            if position in ("far_left", "left"):
                mult = ALIGN_FAR_MULT if position == "far_left" else 1.0
                return int(ActionCategory.LOOK), (0, 0, -ALIGN_CAM_DX * mult, 0)
            if position in ("far_right", "right"):
                mult = ALIGN_FAR_MULT if position == "far_right" else 1.0
                return int(ActionCategory.LOOK), (0, 0, ALIGN_CAM_DX * mult, 0)
            if position == "center":
                return int(ActionCategory.MOVE), (0, 1.0, 0, 0)

        # 撞墙
        if blocked == "yes":
            return int(ActionCategory.LOOK), (0, 0, self._scan_dx(TURN_AROUND_DX), 0)

        # 边移动边扫视
        return int(ActionCategory.MOVE_LOOK), (0, 1.0, self._scan_dx(SCAN_CAM_DX), 0)
