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
        self._camera = dxcam.create(output_idx=0, output_color="BGR")

    def grab(self) -> Optional[np.ndarray]:
        """抓一帧。返回 BGR ndarray（下采样后），静态画面可能为 None。"""
        self._lazy_init()
        frame = self._camera.grab(region=self.config.region)
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
