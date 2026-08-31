"""屏幕捕获封装（dxcam / DDA）。

- dxcam 默认返回 BGR ndarray；静态画面返回 None（需调用方用 last_frame 兜底）。
- 统一 `perf_counter_ns` 时钟：录制时与 input_logger 的事件流对齐。
- 录制参数集中在此，便于复用（RealtimeAgent 也用）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CaptureConfig:
    fps: int = 30                 # 目标帧率
    region: tuple[int, int, int, int] | None = None  # (left, top, w, h)；None=全屏
    output_size: tuple[int, int] | None = None       # 下采样尺寸；None=原始
    window_title: str | None = "Identity V"          # 目标窗口标题（模糊匹配）


class ScreenCapture:
    """dxcam 屏幕捕获。仅在真实游戏环境可用，统一异常处理。"""

    def __init__(self, config: CaptureConfig):
        self.config = config
        self._camera = None
        self._cv2 = None
        self._region = None

    def _lazy_init(self) -> None:
        if self._camera is not None:
            return
        try:
            import dxcam
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("dxcam 未安装（pip install dxcam）") from e
        try:
            import cv2
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("opencv-python 未安装") from e
        self._cv2 = cv2
        try:
            self._camera = dxcam.create(output_idx=0, output_color="BGR")
        except Exception as exc:
            # DXGI DuplicateOutput 常以 COMError/Access Denied 形式出现；
            # 给出针对 Windows 会话、管理员权限和 RDP/锁屏的可执行提示。
            self._camera = None
            msg = str(exc)
            raise RuntimeError(
                "DXGI 屏幕复制初始化失败（DuplicateOutput 拒绝访问）。"
                "请确认当前 Python 进程与游戏处于同一交互式桌面、以管理员身份运行，"
                "且未处于锁屏/RDP 断开会话；然后重试。原始错误: " + msg
            ) from exc
        # dxcam 使用 (left, top, right, bottom)；对外配置保持更直观的
        # (left, top, width, height)。未显式指定时按窗口标题解析一次区域。
        if self.config.region is not None:
            left, top, width, height = self.config.region
            self._region = (left, top, left + width, top + height)
        elif self.config.window_title:
            try:
                from idv_agent.capture.input_logger import find_window_region
                found = find_window_region(self.config.window_title)
                if found is not None:
                    left, top, width, height = found
                    self._region = (left, top, left + width, top + height)
            except Exception:
                # 找不到窗口时安全回退全屏，便于 headless/多显示器调试。
                self._region = None

    def grab(self) -> Optional[np.ndarray]:
        """抓一帧。返回 BGR ndarray（下采样后），静态画面可能为 None。"""
        self._lazy_init()
        frame = self._camera.grab(region=self._region)
        if frame is None:
            return None
        if self.config.output_size is not None:
            frame = self._cv2.resize(
                frame, self.config.output_size, interpolation=self._cv2.INTER_AREA
            )
        return frame

    def __enter__(self) -> "ScreenCapture":
        self._lazy_init()
        return self

    def __exit__(self, *exc) -> None:
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            self._camera = None
            self._region = None
