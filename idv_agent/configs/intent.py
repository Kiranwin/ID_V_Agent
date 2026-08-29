"""求生者战术意图类别 + 16 维严格正交原型向量。

设计：
- 类别用 IntEnum，便于 (id, name) 双向查询。
- 原型向量用 QR 分解保证严格正交（np.random.randn 后 normalize 不严格正交，
  会让两个意图向量内积接近但不等于 0，干扰快系统条件输入语义）。
- 固定 seed 确保跨会话可复现：所有 session 的 intent_vector 来自同一组原型。

注意（P5）：HEAL / RESCUE 依赖视觉判定，按键推断的启发式映射不产生这两个标签，
属弱监督。后续用视觉辅助标注补。
"""

from __future__ import annotations

from enum import IntEnum
from functools import lru_cache

import numpy as np


INTENT_VECTOR_DIM = 16


class IntentCategory(IntEnum):
    IDLE = 0          # 待机观察
    MOVE = 1          # 移动巡逻
    DECODE = 2        # 破译密码机
    VAULT = 3         # 翻窗 / 翻板
    HEAL = 4          # 自我治疗 / 队友治疗（依赖视觉，弱监督）
    RESCUE = 5        # 救援队友（依赖视觉，弱监督）
    USE_SKILL = 6     # 使用技能
    USE_ITEM = 7      # 使用道具
    EMOTE = 8         # 表情 / 呼救
    OBSERVE = 9       # 查看地图


NUM_INTENT_CATEGORIES = len(IntentCategory)


@lru_cache(maxsize=1)
def get_prototype_matrix(seed: int = 20260516) -> np.ndarray:
    """[NUM_INTENT_CATEGORIES, INTENT_VECTOR_DIM] 严格正交单位行向量矩阵。"""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((INTENT_VECTOR_DIM, INTENT_VECTOR_DIM))
    q, _ = np.linalg.qr(raw)
    # q 的列两两正交单位长；取前 NUM_INTENT_CATEGORIES 列转置成行向量
    proto = q[:, :NUM_INTENT_CATEGORIES].T.astype(np.float32)
    return proto


def intent_to_vector(intent: IntentCategory) -> np.ndarray:
    return get_prototype_matrix()[int(intent)].copy()
