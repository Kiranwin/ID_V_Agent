"""把快控制器 (category_id, cont[4]) 解码成键鼠差分命令。

关键设计：**有状态差分**。维护当前已按物理键集合，每帧算：
- to_press = 目标 - 已按下
- to_release = 已按下 - 目标

TAP 类别（模型只在 press 帧输出 TAP，下帧切 NOOP）会自然变成 press/next release；
HOLD 类别（连续多帧输出）保持按键按下。

连续值约定（与 configs/schema.py 一致）：cont[0]=move_x, cont[1]=move_y,
cont[2]=cam_dx, cont[3]=cam_dy。WASD 阈值 >0.5 才算按下；鼠标增量反归一化 cam_pixel_scale。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP, SurvivorKeymap
from idv_agent.configs.schema import ActionCategory


_FREEZE_CONTINUOUS_CATEGORIES = {
    ActionCategory.NOOP,
    ActionCategory.INTERACT_HOLD,
    ActionCategory.INTERACT_TAP,
    ActionCategory.INTERACT_RELEASE,
    ActionCategory.VAULT,
    ActionCategory.DROP_BOARD,
    ActionCategory.MAP,
    ActionCategory.EMOTE,
    ActionCategory.OTHER,
}

_NO_KEY_CATEGORIES = {
    ActionCategory.NOOP,
    ActionCategory.LOOK,
    ActionCategory.INTERACT_RELEASE,
    ActionCategory.OTHER,
}


@dataclass
class Command:
    kind: str        # "press" / "release" / "mouse_move"
    code: Optional[str] = None     # 物理键码（"key:w"）/ None for mouse
    dx_px: float = 0.0
    dy_px: float = 0.0

    def __repr__(self) -> str:
        if self.kind == "mouse_move":
            return f"mouse_move(dx={self.dx_px:+.0f},dy={self.dy_px:+.0f})"
        return f"{self.kind}({self.code})"


@dataclass
class DecoderState:
    held_keys: set[str] = field(default_factory=set)


class ActionDecoder:
    def __init__(
        self,
        keymap: SurvivorKeymap = DEFAULT_SURVIVOR_KEYMAP,
        wasd_threshold: float = 0.5,
        cam_pixel_scale: float = 200.0,
        cam_motion_dead_zone_px: float = 1.0,
    ):
        self.keymap = keymap
        self.wasd_threshold = wasd_threshold
        self.cam_pixel_scale = cam_pixel_scale
        self.cam_dead_zone = cam_motion_dead_zone_px
        self.state = DecoderState()

    def reset(self) -> None:
        self.state = DecoderState()

    def _wasd_target(self, move_x: float, move_y: float) -> set[str]:
        keys: set[str] = set()
        if move_x > self.wasd_threshold:
            keys.add(self.keymap.move_right)
        elif move_x < -self.wasd_threshold:
            keys.add(self.keymap.move_left)
        if move_y > self.wasd_threshold:
            keys.add(self.keymap.move_forward)
        elif move_y < -self.wasd_threshold:
            keys.add(self.keymap.move_back)
        return keys

    def _target_keys(self, category: ActionCategory, move_x: float, move_y: float) -> set[str]:
        if category in _NO_KEY_CATEGORIES:
            return set()
        keys: set[str] = set()
        if category in (ActionCategory.MOVE, ActionCategory.MOVE_LOOK):
            keys |= self._wasd_target(move_x, move_y)
        elif category in (ActionCategory.INTERACT_TAP, ActionCategory.INTERACT_HOLD):
            keys.add(self.keymap.interact)
        elif category in (ActionCategory.VAULT, ActionCategory.DROP_BOARD):
            v = getattr(self.keymap, "vault", None)
            if v is not None:
                keys.add(v)
        elif category in (ActionCategory.SKILL_1_TAP, ActionCategory.SKILL_1_HOLD):
            v = getattr(self.keymap, "skill_1", None)
            if v is not None:
                keys.add(v)
        elif category in (ActionCategory.SKILL_2_TAP, ActionCategory.SKILL_2_HOLD):
            v = getattr(self.keymap, "skill_2", None)
            if v is not None:
                keys.add(v)
        elif category in (ActionCategory.ITEM_1, ActionCategory.ITEM_2,
                          ActionCategory.ITEM_3, ActionCategory.ITEM_4):
            items = getattr(self.keymap, "item_keys", ())
            idx = int(category) - int(ActionCategory.ITEM_1)
            if idx < len(items):
                keys.add(items[idx])
        elif category == ActionCategory.MAP:
            v = getattr(self.keymap, "map_view", None)
            if v is not None:
                keys.add(v)
        elif category == ActionCategory.EMOTE:
            v = getattr(self.keymap, "emote", None)
            if v is not None:
                keys.add(v)
        return keys

    def decode(self, category_id: int, continuous: Iterable[float]) -> list[Command]:
        category = ActionCategory(int(category_id))
        cont = list(continuous)
        if len(cont) != 4:
            raise ValueError(f"continuous 必须 4 维，收到 {len(cont)}")
        move_x, move_y, cam_dx, cam_dy = cont

        if category in _FREEZE_CONTINUOUS_CATEGORIES:
            move_x = move_y = cam_dx = cam_dy = 0.0

        target = self._target_keys(category, move_x, move_y)
        held = self.state.held_keys
        to_release = held - target
        to_press = target - held

        cmds: list[Command] = []
        for code in sorted(to_release):
            cmds.append(Command(kind="release", code=code))
        for code in sorted(to_press):
            cmds.append(Command(kind="press", code=code))

        cam_dx_px = cam_dx * self.cam_pixel_scale
        cam_dy_px = cam_dy * self.cam_pixel_scale
        if abs(cam_dx_px) > self.cam_dead_zone or abs(cam_dy_px) > self.cam_dead_zone:
            cmds.append(Command(kind="mouse_move", dx_px=cam_dx_px, dy_px=cam_dy_px))

        self.state.held_keys = target
        return cmds

    def release_all(self) -> list[Command]:
        cmds = [Command(kind="release", code=k) for k in sorted(self.state.held_keys)]
        self.state.held_keys = set()
        return cmds
