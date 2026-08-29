"""动作空间 schema：20 个离散类别 + 4 个连续通道。

设计取舍：
- 类别按「逻辑动作」组织（INTERACT_HOLD / SKILL_1_TAP），不直接绑定按键；
  键位映射由 configs/keymap.py 提供，可换角色 / 自定义键位。
- TAP vs HOLD 通过持续时间阈值区分，便于学到「快按」与「持续按住」的语义差异。
- DROP_BOARD 暂为占位类：当前规则下与 VAULT 等同发射，需视觉判定区分翻窗/翻板（P10）。

维护约定（docs/03）：
- 此文件是动作空间唯一来源，改动必须同步 docs/03 与 tests。
- 连续通道顺序（CONT_MOVE_X/Y, CONT_CAM_DX/DY）全管线复用，禁止改序。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ActionCategory(IntEnum):
    NOOP = 0
    MOVE = 1
    LOOK = 2
    MOVE_LOOK = 3
    INTERACT_TAP = 4
    INTERACT_HOLD = 5
    INTERACT_RELEASE = 6
    VAULT = 7
    DROP_BOARD = 8       # 占位：与 VAULT 等同，需视觉判定才能区分
    SKILL_1_TAP = 9
    SKILL_1_HOLD = 10
    SKILL_2_TAP = 11
    SKILL_2_HOLD = 12
    ITEM_1 = 13
    ITEM_2 = 14
    ITEM_3 = 15
    ITEM_4 = 16
    MAP = 17
    EMOTE = 18
    OTHER = 19


NUM_CATEGORIES = len(ActionCategory)
NUM_CONTINUOUS = 4

# 连续通道顺序约定（全管线复用）
CONT_MOVE_X = 0      # A = -1, D = +1
CONT_MOVE_Y = 1      # S = -1, W = +1
CONT_CAM_DX = 2      # 鼠标 X 增量 / cam_pixel_scale，截断到 [-1, 1]
CONT_CAM_DY = 3      # 鼠标 Y 增量 / cam_pixel_scale，截断到 [-1, 1]


@dataclass
class ExtractParams:
    """从原始 events 推导动作类别 / 连续值时的可调参数。"""

    # 短按 vs 长按时间阈值（ms）。按下后到 frame 时刻的持续时间 < tap_threshold 视为 TAP。
    tap_threshold_ms: float = 150.0

    # 抬起事件的「余热窗口」：抬起后多少 ms 内仍标记为 RELEASE 而非 NOOP。
    release_window_ms: float = 50.0

    # 鼠标移动的「显著」阈值（像素/帧）。低于此视为静止，避免微抖被算作 LOOK。
    cam_motion_dead_zone_px: float = 2.0

    # 鼠标增量归一化物理尺度：cam_pixel_scale 像素 → 1.0。
    # P7 灵敏度域：录数据与部署必须用同一尺度，且记录游戏内灵敏度保持一致。
    cam_pixel_scale: float = 200.0

    # WASD 离散方向归一化是否保持单位向量；False 时对角线移动幅度 = √2 会被 clip。
    normalize_diagonal: bool = True

    # ---- 第五人格特有的隐式状态：破译 ----
    # 求生者按一次 Q 即进入「自动破译」，此后无按键事件但语义是 INTERACT_HOLD。
    enable_decoding_state: bool = True

    # 破译状态最长持续时间（秒）。超过则强制退出，防止 Q 误触导致后续被错标。
    # IDV 单次破译最长约 80s；90s 是「真破译」与「Q 误触 + 长静止」的平衡点。
    max_decode_duration_s: float = 90.0
