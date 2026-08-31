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


SCAN_CAM_DX = 0.025       # 慢速扫视（≈5px/帧 @ 30Hz）
ALIGN_CAM_DX = 0.045      # 慢速对齐目标
ALIGN_FAR_MULT = 1.6      # far_* 额外倍率
TURN_AROUND_DX = 0.15     # 撞墙转身
SCAN_PHASE_DURATION_S = 2.5
THREAT_RUN_DURATION_S = 2.0   # 感知到威胁时跑一段再重新评估


class CipherVisualServo:
    """Geometry-only controller for the M1 cipher approach.

    The controller treats the screen centre as the character's forward axis.
    It uses horizontal bbox error for camera alignment and bbox bottom/prompt
    as approach cues; area is retained only as a diagnostic value and never
    drives a near/far transition by itself.
    """

    def __init__(self, deadband: float = 0.05, align_band: float = 0.15,
                 cam_gain: float = 0.28, prompt_confirm_frames: int = 2):
        if not (0 <= deadband < align_band <= 0.5):
            raise ValueError("deadband/align_band 参数无效")
        if cam_gain <= 0 or prompt_confirm_frames < 1:
            raise ValueError("cam_gain/prompt_confirm_frames 参数无效")
        self.deadband = float(deadband)
        self.align_band = float(align_band)
        self.cam_gain = float(cam_gain)
        self.prompt_confirm_frames = int(prompt_confirm_frames)
        self._prompt_frames = 0
        self._q_sent = False

    def reset(self) -> None:
        self._prompt_frames = 0
        self._q_sent = False

    def decide(self, spatial: dict) -> Tuple[int, Tuple[float, float, float, float]]:
        """Return an ActionCategory/continuous tuple for one perception update."""
        visible = spatial.get("visible") == "yes"
        decoding = spatial.get("decoding_state") == "yes"
        prompt = spatial.get("interact_prompt") == "yes"
        if decoding:
            self._q_sent = True
            return int(ActionCategory.INTERACT_HOLD), (0, 0, 0, 0)
        if prompt:
            self._prompt_frames += 1
        else:
            self._prompt_frames = 0
            # A disappeared prompt after a failed/finished interaction allows
            # a later, newly confirmed prompt to trigger again.
            if not spatial.get("interacting") == "yes":
                self._q_sent = False
        if self._prompt_frames >= self.prompt_confirm_frames and not self._q_sent:
            self._q_sent = True
            return int(ActionCategory.INTERACT_TAP), (0, 0, 0, 0)
        if not visible:
            return int(ActionCategory.MOVE_LOOK), (0, 1.0, SCAN_CAM_DX, 0)

        center_x = spatial.get("center_x")
        if center_x is None:
            # Legacy detectors only expose a coarse position; preserve M1
            # behaviour until geometry becomes available.
            position = spatial.get("position", "none")
            if position in ("left", "far_left"):
                center_x = 0.25
            elif position in ("right", "far_right"):
                center_x = 0.75
            else:
                center_x = 0.5
        error = float(center_x) - 0.5
        cam = max(-1.0, min(1.0, error * self.cam_gain))
        if abs(error) > self.align_band:
            return int(ActionCategory.LOOK), (0, 0, cam, 0)
        if abs(error) > self.deadband:
            return int(ActionCategory.MOVE_LOOK), (0, 1.0, cam, 0)
        # In the centre corridor, move straight.  bottom_y and distance are
        # telemetry for logging/safety gates, not sole distance proxies.
        return int(ActionCategory.MOVE), (0, 1.0, cam, 0)


class RuleAgent:
    def __init__(self, visual_servo: CipherVisualServo | None = None):
        self._scan_dir = +1
        self._scan_started_at = time.perf_counter()
        self._threat_until = 0.0
        self.visual_servo = visual_servo

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

        # 撞墙优先于视觉伺服，确保脱困分支仍能打断“持续前进”。
        if blocked == "yes":
            return int(ActionCategory.LOOK), (0, 0, self._scan_dx(TURN_AROUND_DX), 0)

        # Optional geometry-based M1 controller.  Keep the original rule path
        # available for A/B tests and for sessions without bbox geometry.
        if self.visual_servo is not None:
            return self.visual_servo.decide(sp)

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

        # 边移动边扫视
        return int(ActionCategory.MOVE_LOOK), (0, 1.0, self._scan_dx(SCAN_CAM_DX), 0)
