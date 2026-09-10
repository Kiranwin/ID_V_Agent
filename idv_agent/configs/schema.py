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

