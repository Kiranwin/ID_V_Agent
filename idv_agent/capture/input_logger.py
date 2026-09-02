"""输入日志（pynput 键鼠 hook）与窗口检测。

关键点（NeAC，见 docs/04）：
- 录制必须**管理员权限**运行，否则 pynput 用户态 hook 被 NeAC/系统隔离，
  进游戏内的 W/A/S/D 事件收不到。
- 所有事件与帧统一 `time.perf_counter_ns()` 时间戳，供 labels/state_machine 对齐。

事件 kind：
    key_down / key_up     key 列给 "key:w" 形式
    mouse_down / mouse_up 给 "btn:left" 形式
    scroll / mouse_move    记录在 mouse_positions.csv（鼠标移动）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread
from typing import Optional

from .screen_capture import CaptureConfig


def _key_to_str(key) -> str:
    """把 pynput 的 key 对象统一转成 "key:w"/"key:space"/"btn:left" 字符串。"""
    try:
        from pynput import keyboard, mouse
        if isinstance(key, keyboard.KeyCode):
            return "key:" + (key.char if key.char is not None else f"<{key.vk}>")
        if isinstance(key, keyboard.Key):
            return "key:" + key.name
        if isinstance(key, mouse.Button):  # pragma: no cover
            return f"btn:{key.name}"
    except Exception:  # pragma: no cover
        pass
    return f"key:{key}"


@dataclass
class EventLogger:
    """把键鼠事件追加写入 CSV。可被独立启动 / 停止。"""
    events_csv: Path
    positions_csv: Path
    _events_fh = None
    _positions_fh = None

    def __post_init__(self) -> None:
        self.events_csv.parent.mkdir(parents=True, exist_ok=True)
        self._events_fh = self.events_csv.open("w", encoding="utf-8")
        self._events_fh.write("timestamp_ns,kind,code,value\n")
        self._positions_fh = self.positions_csv.open("w", encoding="utf-8")
        self._positions_fh.write("timestamp_ns,x,y\n")
        self._lock = Lock()
        self._last_pos = None

    # ---- 写入 ----
    def log_event(self, kind: str, code: str, value: float = 1.0) -> None:
        ts = time.perf_counter_ns()
        with self._lock:
            if self._events_fh is None:
                return
            self._events_fh.write(f"{ts},{kind},{code},{value}\n")
            self._events_fh.flush()

    def log_mouse_position(self, x: int, y: int) -> None:
        ts = time.perf_counter_ns()
        with self._lock:
            if self._positions_fh is None:
                return
            self._positions_fh.write(f"{ts},{x},{y}\n")
            self._positions_fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._events_fh is not None:
                self._events_fh.close()
                self._events_fh = None
            if self._positions_fh is not None:
                self._positions_fh.close()
                self._positions_fh = None


class InputRecorder:
    """pynput 后台监听，事件转发到 EventLogger。`start()` / `stop()` 生命周期。"""

    def __init__(self, events_csv: Path, positions_csv: Path, poll_position_hz: float = 100.0,
                 ignored_codes: set[str] | None = None):
        self.logger = EventLogger(events_csv=events_csv, positions_csv=positions_csv)
        self._poll_position_hz = poll_position_hz
        self._ignored_codes = set(ignored_codes or ())
        self._stop = False
        self._threads: list[Thread] = []
        self._listeners = []

    def start(self) -> None:
        try:
            from pynput import keyboard, mouse
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pynput 未安装") from e

        def on_key(kind: str):
            def _cb(key):
                code = _key_to_str(key)
                if code not in self._ignored_codes:
                    self.logger.log_event(kind, code)
            return _cb

        kb_listener = keyboard.Listener(
            on_press=on_key("key_down"),
            on_release=on_key("key_up"),
        )
        ms_listener = mouse.Listener(
            on_click=lambda x, y, button, pressed: self.logger.log_event(
                "mouse_down" if pressed else "mouse_up", _key_to_str(button)
            ),
            on_scroll=lambda x, y, dx, dy: self.logger.log_event("scroll", f"key:scroll"),
        )
        self._listeners = [kb_listener, ms_listener]
        kb_listener.start()
        ms_listener.start()
        # 后台高频轮询鼠标位置（pynput 的 on_move 会刷屏，用 on_move 记录稀疏轮询）
        def _poll_position():
            from pynput import mouse as _mouse
            controller = _mouse.Controller()
            while not self._stop:
                pos = controller.position
                self.logger.log_mouse_position(*pos)
                time.sleep(1.0 / self._poll_position_hz)
        t = Thread(target=_poll_position, name="mouse-poll", daemon=True)
        self._threads.append(t)
        t.start()

    def stop(self) -> None:
        self._stop = True
        for listener in self._listeners:
            try:
                listener.stop()
            except Exception:
                pass
        for t in self._threads:
            t.join(timeout=1.0)
        self.logger.close()


def list_windows(title_substr: Optional[str] = None) -> list[str]:
    """列出前台/可见窗口标题。用于确认游戏窗口名（pygetwindow）。"""
    try:
        import pygetwindow as gw
    except ImportError:  # pragma: no cover
        return []
    try:
        wins = gw.getAllTitles()
    except Exception:
        return []
    wins = [w for w in wins if w.strip() and (title_substr is None or title_substr.lower() in w.lower())]
    return wins


def find_window_region(window_title: str) -> tuple[int, int, int, int] | None:
    """按标题模糊匹配窗口，返回 (left, top, width, height)。找不到返回 None。"""
    try:
        import pygetwindow as gw
    except ImportError:
        return None
    try:
        win = gw.getWindowsWithTitle(window_title)
        if not win:
            return None
        w = win[0]
        if not w.isActive:
            try:
                w.activate()
                time.sleep(0.1)
            except Exception:
                pass
        return (w.left, w.top, w.width, w.height)
    except Exception:
        return None
