"""Windows Raw Input relative mouse-motion capture.

This module uses only the user-mode Raw Input API.  It does not inspect a game
process or inject input.  The collector is intentionally separate from the
normal recorder until it has been validated in the training room.
"""

from __future__ import annotations

import csv
import ctypes
import ctypes.wintypes
import os
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class MouseDelta:
    timestamp_ns: int
    dx: int
    dy: int


def summarize_deltas(deltas: list[MouseDelta]) -> dict[str, int | float]:
    nonzero = [item for item in deltas if item.dx or item.dy]
    return {
        "samples": len(deltas),
        "nonzero_samples": len(nonzero),
        "nonzero_rate": len(nonzero) / len(deltas) if deltas else 0.0,
        "total_dx": sum(item.dx for item in deltas),
        "total_dy": sum(item.dy for item in deltas),
        "max_abs_dx": max((abs(item.dx) for item in deltas), default=0),
        "max_abs_dy": max((abs(item.dy) for item in deltas), default=0),
    }


def write_deltas_csv(path: Path, deltas: list[MouseDelta]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(("timestamp_ns", "dx", "dy"))
        writer.writerows((item.timestamp_ns, item.dx, item.dy) for item in deltas)


def _parse_raw_mouse_delta(raw: bytes, header_size: int) -> tuple[int, int] | None:
    """Decode one RAWINPUT buffer; return None for non-relative reports."""
    if len(raw) < header_size + 16:
        return None
    if struct.unpack_from("<I", raw, 0)[0] != 0:  # RIM_TYPEMOUSE
        return None
    mouse = raw[header_size:]
    flags = struct.unpack_from("<H", mouse, 0)[0]
    if flags & 0x01:
        return None  # absolute tablet/touch report
    # RAWMOUSE: flags (0), buttons (4), ulRawButtons (8), lLastX (12),
    # lLastY (16), extra information (20).
    return struct.unpack_from("<ii", mouse, 12)


class RawInputMouse:
    """Collect relative mouse reports through a hidden Win32 message window."""

    _WM_INPUT = 0x00FF
    _WM_QUIT = 0x0012
    _RID_INPUT = 0x10000003
    _RIM_TYPEMOUSE = 0
    _RIDEV_INPUTSINK = 0x00000100

    def __init__(self, callback: Callable[[MouseDelta], None] | None = None):
        if os.name != "nt":
            raise RuntimeError("Windows Raw Input 仅支持 Windows")
        self.callback = callback
        self.deltas: list[MouseDelta] = []
        self._thread: threading.Thread | None = None
        self._thread_id = 0
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._wndproc = None
        self._class_name = f"IDVAgentRawMouse_{id(self):x}"

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RawInputMouse 已启动")
        self._thread = threading.Thread(target=self._run, name="raw-input-mouse", daemon=True)
        self._thread.start()
        if not self._ready.wait(3.0):
            raise RuntimeError("Raw Input 消息窗口启动超时")
        if self._error is not None:
            raise RuntimeError(f"Raw Input 启动失败: {self._error}") from self._error

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        if self._thread_id:
            ctypes.windll.user32.PostThreadMessageW(self._thread_id, self._WM_QUIT, 0, 0)
        self._thread.join(timeout=3.0)
        self._thread = None

    def _run(self) -> None:  # pragma: no cover - requires Windows desktop
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            self._thread_id = kernel32.GetCurrentThreadId()

            WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_longlong, ctypes.c_void_p,
                                         ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t)

            class WNDCLASSW(ctypes.Structure):
                _fields_ = [("style", ctypes.c_uint), ("lpfnWndProc", WNDPROC),
                            ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
                            ("hInstance", ctypes.c_void_p), ("hIcon", ctypes.c_void_p),
                            ("hCursor", ctypes.c_void_p), ("hbrBackground", ctypes.c_void_p),
                            ("lpszMenuName", ctypes.c_wchar_p), ("lpszClassName", ctypes.c_wchar_p)]

            def wndproc(hwnd, msg, wparam, lparam):
                if msg == self._WM_INPUT:
                    self._read_raw_input(ctypes.c_void_p(lparam))
                return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

            self._wndproc = WNDPROC(wndproc)
            instance = kernel32.GetModuleHandleW(None)
            wc = WNDCLASSW(lpfnWndProc=self._wndproc, hInstance=instance,
                           lpszClassName=self._class_name)
            atom = user32.RegisterClassW(ctypes.byref(wc))
            if not atom and ctypes.get_last_error() != 1410:  # class already exists
                raise ctypes.WinError()
            hwnd = user32.CreateWindowExW(0, self._class_name, self._class_name, 0,
                                          0, 0, 0, 0, 0, 0, instance, None)
            if not hwnd:
                raise ctypes.WinError()
            class RAWINPUTDEVICE(ctypes.Structure):
                _fields_ = [("usUsagePage", ctypes.c_ushort), ("usUsage", ctypes.c_ushort),
                            ("dwFlags", ctypes.c_uint), ("hwndTarget", ctypes.c_void_p)]
            rid = RAWINPUTDEVICE(0x01, 0x02, self._RIDEV_INPUTSINK, hwnd)
            if not user32.RegisterRawInputDevices(ctypes.byref(rid), 1, ctypes.sizeof(rid)):
                raise ctypes.WinError()
            self._ready.set()
            msg = ctypes.wintypes.MSG()
            while not self._stop.is_set() and user32.GetMessageW(ctypes.byref(msg), 0, 0, 0) > 0:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            user32.DestroyWindow(hwnd)
            user32.UnregisterClassW(self._class_name, instance)
        except BaseException as exc:
            self._error = exc
            self._ready.set()

    def _read_raw_input(self, handle) -> None:  # pragma: no cover - requires Windows desktop
        user32 = ctypes.windll.user32
        size = ctypes.c_uint(0)
        header_size = 8 + ctypes.sizeof(ctypes.c_void_p) * 2
        if user32.GetRawInputData(handle, self._RID_INPUT, None, ctypes.byref(size), header_size) == -1:
            return
        buf = ctypes.create_string_buffer(size.value)
        if user32.GetRawInputData(handle, self._RID_INPUT, buf, ctypes.byref(size), header_size) == -1:
            return
        raw = buf.raw[:size.value]
        delta = _parse_raw_mouse_delta(raw, header_size)
        if delta is None:
            return
        dx, dy = delta
        item = MouseDelta(time.perf_counter_ns(), dx, dy)
        self.deltas.append(item)
        if self.callback is not None:
            self.callback(item)

    def __enter__(self) -> "RawInputMouse":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
