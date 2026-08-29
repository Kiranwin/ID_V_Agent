"""把 ActionDecoder 输出的 Command 发给系统（或 dry-run 打印）。

物理键码（input_logger 格式）→ pydirectinput 名字映射。设计：
- dry_run=True 只记录到内部 deque，不调用 pydirectinput（默认安全）。
- 真发送按 release-before-press 顺序，避免按键冲突。
- shutdown() 释放所有按下键，避免脚本异常退出后键卡住。
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

from idv_agent.agent.action_decoder import Command


# input_logger 格式 -> pydirectinput 格式
_SPECIAL_KEY_MAP = {
    "space": "space",
    "tab": "tab",
    "shift": "shift",
    "ctrl": "ctrl",
    "alt": "alt",
    "enter": "enter",
    "esc": "escape",
}


def _to_pydirectinput_key(code: str) -> Optional[tuple[str, str]]:
    """返回 (kind, name) 或 None（无法识别）。kind: "key" / "mouse"。"""
    if code.startswith("key:"):
        sub = code[4:]
        if sub in _SPECIAL_KEY_MAP:
            return ("key", _SPECIAL_KEY_MAP[sub])
        if len(sub) == 1:
            return ("key", sub.lower())
        if sub.startswith("f") and sub[1:].isdigit():
            return ("key", sub)
        return ("key", sub)
    if code.startswith("btn:"):
        sub = code[4:]
        if sub in ("left", "right", "middle"):
            return ("mouse", sub)
    return None


class ActionExecutor:
    def __init__(self, dry_run: bool = True, log_history: int = 200):
        self.dry_run = dry_run
        self.history: deque[str] = deque(maxlen=log_history)
        self.counter = {"press": 0, "release": 0, "mouse_move": 0, "ignored": 0}
        self._held: set[str] = set()
        self._pdi = None
        if not dry_run:
            try:
                import pydirectinput  # noqa: F401
                pydirectinput.FAILSAFE = False
                pydirectinput.PAUSE = 0
                self._pdi = pydirectinput
            except ImportError as e:
                raise RuntimeError(
                    "需要 pydirectinput 才能真发送键鼠：pip install pydirectinput\n"
                    f"原始错误: {e}"
                )

    def execute(self, commands: Iterable[Command]) -> None:
        for cmd in commands:
            if cmd.kind == "press":
                self._do_press(cmd.code)
            elif cmd.kind == "release":
                self._do_release(cmd.code)
            elif cmd.kind == "mouse_move":
                self._do_mouse_move(cmd.dx_px, cmd.dy_px)

    def _do_press(self, code: str) -> None:
        target = _to_pydirectinput_key(code)
        self.history.append(f"press {code}")
        if target is None:
            self.counter["ignored"] += 1
            return
        kind, name = target
        if not self.dry_run:
            try:
                if kind == "key":
                    self._pdi.keyDown(name)
                else:
                    self._pdi.mouseDown(button=name)
            except Exception as e:
                self.history.append(f"  [ERR] press {code}: {e}")
                self.counter["ignored"] += 1
                return
        self._held.add(code)
        self.counter["press"] += 1

    def _do_release(self, code: str) -> None:
        target = _to_pydirectinput_key(code)
        self.history.append(f"release {code}")
        if target is None:
            self.counter["ignored"] += 1
            return
        kind, name = target
        if not self.dry_run:
            try:
                if kind == "key":
                    self._pdi.keyUp(name)
                else:
                    self._pdi.mouseUp(button=name)
            except Exception as e:
                self.history.append(f"  [ERR] release {code}: {e}")
                self.counter["ignored"] += 1
                return
        self._held.discard(code)
        self.counter["release"] += 1

    def _do_mouse_move(self, dx: float, dy: float) -> None:
        self.history.append(f"mouse_move({dx:+.0f},{dy:+.0f})")
        if not self.dry_run:
            try:
                self._pdi.moveRel(int(round(dx)), int(round(dy)),
                                  relative=True, disable_mouse_acceleration=True)
            except TypeError:
                self._pdi.moveRel(int(round(dx)), int(round(dy)), relative=True)
            except Exception as e:
                self.history.append(f"  [ERR] mouse_move: {e}")
                self.counter["ignored"] += 1
                return
        self.counter["mouse_move"] += 1

    def shutdown(self) -> None:
        for code in list(self._held):
            self._do_release(code)
