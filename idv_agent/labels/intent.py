"""启发式意图推断 + 键位友好显示 + 动作骨架。

意图推断：ActionCategory → IntentCategory 映射 → 滑窗多数投票平滑。
注意（P5）：HEAL/RESCUE 依赖视觉，启发式推断不含，属弱监督。
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from idv_agent.configs.intent import (
    INTENT_VECTOR_DIM,
    IntentCategory,
    get_prototype_matrix,
)
from idv_agent.configs.schema import ActionCategory, NUM_CONTINUOUS

# ActionCategory -> 候选 IntentCategory（无视觉判定时的最佳猜测）
_CATEGORY_TO_INTENT: dict[ActionCategory, IntentCategory] = {
    ActionCategory.NOOP: IntentCategory.IDLE,
    ActionCategory.MOVE: IntentCategory.MOVE,
    ActionCategory.LOOK: IntentCategory.IDLE,
    ActionCategory.MOVE_LOOK: IntentCategory.MOVE,
    ActionCategory.INTERACT_HOLD: IntentCategory.DECODE,
    ActionCategory.INTERACT_TAP: IntentCategory.DECODE,
    ActionCategory.INTERACT_RELEASE: IntentCategory.DECODE,
    ActionCategory.VAULT: IntentCategory.VAULT,
    ActionCategory.DROP_BOARD: IntentCategory.VAULT,
    ActionCategory.SKILL_1_TAP: IntentCategory.USE_SKILL,
    ActionCategory.SKILL_1_HOLD: IntentCategory.USE_SKILL,
    ActionCategory.SKILL_2_TAP: IntentCategory.USE_SKILL,
    ActionCategory.SKILL_2_HOLD: IntentCategory.USE_SKILL,
    ActionCategory.ITEM_1: IntentCategory.USE_ITEM,
    ActionCategory.ITEM_2: IntentCategory.USE_ITEM,
    ActionCategory.ITEM_3: IntentCategory.USE_ITEM,
    ActionCategory.ITEM_4: IntentCategory.USE_ITEM,
    ActionCategory.MAP: IntentCategory.OBSERVE,
    ActionCategory.EMOTE: IntentCategory.EMOTE,
    ActionCategory.OTHER: IntentCategory.IDLE,
}


def category_to_intent(cat_id: int) -> IntentCategory:
    return _CATEGORY_TO_INTENT.get(ActionCategory(cat_id), IntentCategory.IDLE)


def smooth_intents(intents: Sequence[IntentCategory], window_size: int) -> list[IntentCategory]:
    """中心对称窗口多数投票平滑，减少帧间抖动。"""
    if window_size <= 1:
        return list(intents)
    n = len(intents)
    out: list[IntentCategory] = [IntentCategory.IDLE] * n
    half = window_size // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        c = Counter(intents[lo:hi])
        out[i] = c.most_common(1)[0][0]
    return out


def infer_intents_for_frames(actions_df: pd.DataFrame, fps: float, smooth_window_s: float = 1.0):
    """从 per_frame_actions DataFrame 推断每帧 IntentCategory + 16 维向量。

    返回 (intents_list, vectors[N, 16])。
    """
    raw = [category_to_intent(int(c)) for c in actions_df["category_id"]]
    window = max(1, int(round(smooth_window_s * fps)))
    smoothed = smooth_intents(raw, window)

    proto = get_prototype_matrix()
    vectors = np.stack([proto[int(it)] for it in smoothed], axis=0).astype(np.float32)
    return smoothed, vectors


# ---------- 键位友好显示 ----------

_KEY_DISPLAY_OVERRIDES = {
    "key:space": "SPACE",
    "key:tab": "TAB",
    "key:shift": "SHIFT",
    "key:ctrl": "CTRL",
    "key:alt": "ALT",
    "key:enter": "ENTER",
    "key:esc": "ESC",
    "btn:left": "MOUSE_L",
    "btn:right": "MOUSE_R",
    "btn:middle": "MOUSE_M",
    "key:scroll": "SCROLL",
}


def display_key(code: str) -> str:
    if code in _KEY_DISPLAY_OVERRIDES:
        return _KEY_DISPLAY_OVERRIDES[code]
    if code.startswith("key:"):
        return code[4:].upper()
    if code.startswith("btn:"):
        return code[4:].upper()
    return code.upper()


def _cont(c, idx: int) -> float:
    return c[idx] if isinstance(c, (list, tuple)) else 0.0


def build_action_skeleton(cat_id: int, cont) -> str:
    """把 category + continuous 组合成语义动作骨架文本。"""
    cat = ActionCategory(cat_id)
    move_x = _cont(cont, 0)
    move_y = _cont(cont, 1)
    base = cat.name
    if cat in (ActionCategory.MOVE, ActionCategory.MOVE_LOOK):
        parts = []
        if move_y > 0:
            parts.append("W")
        if move_y < 0:
            parts.append("S")
        if move_x < 0:
            parts.append("A")
        if move_x > 0:
            parts.append("D")
        if parts:
            base += "(" + "+".join(parts) + ")"
    return base


def build_full_action(cat_id: int, cont, held_keys: str = "") -> str:
    """完整操作字符串：text_action（键）+ 连续值 + CAM。"""
    from idv_agent.configs.schema import CONT_CAM_DX, CONT_CAM_DY
    cat = ActionCategory(cat_id)
    skel = build_action_skeleton(cat_id, cont)
    move_x = _cont(cont, 0)
    move_y = _cont(cont, 1)
    cam_dx = _cont(cont, CONT_CAM_DX)
    cam_dy = _cont(cont, CONT_CAM_DY)
    s = skel + f"(MOVE {move_x:.2f},{move_y:.2f})"
    if abs(cam_dx) + abs(cam_dy) > 1e-3:
        s += f"+CAM({cam_dx:.2f},{cam_dy:.2f})"
    if held_keys:
        s += f"[{held_keys}]"
    return s


def build_numeric_action(cont) -> list[float]:
    """连续动作归一化到 [-1,1]。4 通道（见 schema）。"""
    return [float(np.clip(_cont(cont, i), -1, 1)) for i in range(NUM_CONTINUOUS)]


def display_held_keys(codes: Sequence[str]) -> str:
    return "+".join(sorted(display_key(c) for c in codes))
