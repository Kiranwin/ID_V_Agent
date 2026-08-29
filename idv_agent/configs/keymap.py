"""第五人格官方 PC 端：求生者键位 → 逻辑动作 映射表。

约定：键值字符串与 capture/input_logger.py 中 `_key_to_str` 的输出对齐。
- 普通字母数字键: "key:w"  "key:1"
- 特殊键(pynput 命名): "key:space"  "key:tab"  "key:f1"
- 鼠标按键: "btn:left"  "btn:right"  "btn:middle"

改了游戏内键位直接编辑本文件，无需改其它代码。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SurvivorKeymap:
    # 移动（连续值聚合）
    move_forward: str = "key:w"
    move_back: str = "key:s"
    move_left: str = "key:a"
    move_right: str = "key:d"

    # 离散动作
    interact: str = "key:q"        # 交互（破译/治疗/救援/校准）；长按/短按靠持续时长区分
    vault: str = "key:space"       # 翻窗/翻板；破译中做校准时也是 Space
    skill_1: str = "key:f"         # 一技能（部分角色 F）
    skill_2: str = "key:e"         # 二技能（部分角色 E）
    map_view: str = "key:tab"
    emote: str = "key:v"

    # 道具栏（可空表示该位无绑定）
    item_keys: tuple[str, ...] = field(default_factory=lambda: (
        "key:1", "key:2", "key:3", "key:4",
    ))


DEFAULT_SURVIVOR_KEYMAP = SurvivorKeymap()
