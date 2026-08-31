"""感知层：轻量结构化 + 事件检测（P8 修复入口）。

不做逐帧大模型。负责：
1. HUD 状态提取（占位）：修机进度、技能 CD、校准判定 → 裁剪区域将来接 OCR/颜色阈值。
2. 事件检测：受击、爆点、挂椅、破译完成、开门通电等关键事件触发慢层重规划。

P8 场景：监管者砍断破译是游戏侧状态变化（无按键事件）。`EventDetector` 提供
`on_frame(frame)` 接口，将来接入「受击红屏 / 破译进度条消失」视觉信号，把
「破译被打断」事件抛给慢层/状态机，避免破译状态残留到 90s 超时。

当前为**接口 + 占位实现**，真实视觉识别待 M1 感知层落地。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


class PerceptionEvent:
    kind: str        # "hit" / "decode_interrupt" / "explosion" / "cipher_done" / "door_power"
    source: str
    ts: float


# 回调签名：(event_kind: str, detail: dict)
EventCallback = Callable[[str, dict], None]


@dataclass
class CipherDetection:
    """密码机候选的结构化结果，供 RuleAgent 消费。"""
    visible: str = "no"
    position: str = "none"
    distance: str = "none"
    confidence: float = 0.0
    bbox: Optional[tuple[int, int, int, int]] = None
    source: str = "none"

    def as_spatial(self) -> dict:
        return {
            "visible": self.visible,
            "position": self.position,
            "distance": self.distance,
            "confidence": round(float(self.confidence), 3),
        }


class CipherMachineDetector:
    """轻量密码机候选检测器。

    优先使用同地图模板（避免把场景物体误当密码机）；无模板时使用保守的
    HSV 高亮候选启发式，仅输出低置信度结果，便于离线回放和后续校准。
    """

    def __init__(self, template_path: Optional[str | Path] = None,
                 template_threshold: float = 0.72,
                 heuristic_threshold: float = 0.82,
                 highlight_threshold: float = 0.78):
        self.template_threshold = template_threshold
        self.heuristic_threshold = heuristic_threshold
        self.highlight_threshold = highlight_threshold
        self.template = None
        if template_path:
            try:
                import cv2
                self.template = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
                if self.template is None:
                    raise ValueError("模板图片无法读取")
            except Exception as exc:
                raise ValueError(f"密码机模板加载失败: {template_path}: {exc}") from exc

    @staticmethod
    def _location(cx: float, width: int) -> str:
        ratio = cx / max(width, 1)
        if ratio < 0.35: return "far_left" if ratio < 0.18 else "left"
        if ratio > 0.65: return "far_right" if ratio > 0.82 else "right"
        return "center"

    @staticmethod
    def _distance(area: float, frame_area: float) -> str:
        ratio = area / max(frame_area, 1.0)
        if ratio >= 0.035: return "near"
        if ratio >= 0.008: return "mid"
        return "far"

    def _detect_highlight(self, frame: np.ndarray) -> Optional[CipherDetection]:
        """Detect the yellow through-wall cipher marker.

        Identity V renders a cipher as a thin yellow beacon plus a small base
        whenever geometry is occluded.  This is deliberately kept separate
        from the generic bright-colour heuristic: the tall, narrow silhouette
        is a much stronger cue and can safely drive the M1 search policy.
        """
        import cv2
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Yellow beacon (OpenCV hue 15..45), high saturation/value.  Ignore
        # HUD text at the top and the player's action bar at the bottom.
        mask = cv2.inRange(hsv, np.array([15, 120, 150]), np.array([45, 255, 255]))
        mask[: int(h * 0.16), :] = 0
        mask[int(h * 0.92):, :] = 0
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)),
        )
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for i in range(1, n):
            x, y, bw, bh, area = [int(v) for v in stats[i]]
            aspect = bh / max(float(bw), 1.0)
            # The marker is tall and narrow; reject scenery/UI blobs.
            if area < 100 or bh < 24 or bw < 4 or bw > max(80, int(w * .12)):
                continue
            if aspect < 2.5 or area > w * h * 0.03:
                continue
            shape_score = min(1.0, (aspect - 2.5) / 5.0)
            size_score = min(1.0, area / 900.0)
            confidence = min(0.99, 0.78 + 0.12 * shape_score + 0.08 * size_score)
            candidates.append((confidence, x, y, bw, bh, area))
        if not candidates:
            return None
        confidence, x, y, bw, bh, area = max(candidates)
        visible = "yes" if confidence >= self.highlight_threshold else "no"
        return CipherDetection(
            visible, self._location(x + bw / 2, w),
            self._distance(area, w * h), float(confidence),
            (x, y, bw, bh), "highlight",
        )

    def detect(self, frame: np.ndarray) -> CipherDetection:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return CipherDetection()
        import cv2
        h, w = frame.shape[:2]
        if self.template is not None:
            th, tw = self.template.shape[:2]
            if th <= h and tw <= w:
                # CCOEFF_NORMED is undefined for uniform templates (e.g. a
                # tightly cropped icon/patch), which can make minMaxLoc pick
                # the top-left corner regardless of the actual match.  Use
                # normalized squared difference in that case and convert it
                # to the same higher-is-better score convention.
                method = cv2.TM_CCOEFF_NORMED
                # JPEG/template patches may be constant per channel while
                # channel means differ (overall std is then non-zero).
                channel_std = np.std(self.template.reshape(-1, self.template.shape[-1]), axis=0)
                if float(np.max(channel_std)) < 1e-3:
                    method = cv2.TM_SQDIFF_NORMED
                result = cv2.matchTemplate(frame, self.template, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                if method == cv2.TM_SQDIFF_NORMED:
                    score, loc = 1.0 - float(min_val), min_loc
                else:
                    score, loc = float(max_val), max_loc
                if score >= self.template_threshold:
                    x, y = int(loc[0]), int(loc[1])
                    return CipherDetection("yes", self._location(x + tw/2, w),
                        self._distance(tw*th, w*h), float(score), (x,y,tw,th), "template")

        # Through-wall yellow beacon.  This cue is available even when the
        # 3D machine mesh itself is completely hidden.
        highlight = self._detect_highlight(frame)
        if highlight is not None:
            return highlight

        # 保守启发式：密码机常见的高亮黄/青色局部块，排除顶部 HUD。
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 90, 100]), np.array([100, 255, 255]))
        mask[: int(h * 0.16), :] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        candidates = []
        for i in range(1, n):
            x, y, bw, bh, area = [int(v) for v in stats[i]]
            if area < 80 or bw < 6 or bh < 6 or area > w*h*0.08:
                continue
            fill = area / max(bw*bh, 1)
            if fill < 0.12:
                continue
            candidates.append((area * min(fill, 1.0), x, y, bw, bh))
        if not candidates:
            return CipherDetection()
        _, x, y, bw, bh = max(candidates)
        # 启发式只作为候选，置信度低于模板结果，避免盲目驱动注入。
        conf = min(0.9, 0.45 + (bw*bh)/(w*h)*4.0)
        visible = "yes" if conf >= self.heuristic_threshold else "no"
        return CipherDetection(visible, self._location(x+bw/2, w),
            self._distance(bw*bh, w*h), conf, (x,y,bw,bh), "heuristic")


class EventDetector:
    def __init__(self, on_event: Optional[EventCallback] = None,
                 cipher_detector: Optional[CipherMachineDetector] = None):
        self._callbacks: list[EventCallback] = []
        if on_event is not None:
            self._callbacks.append(on_event)
        # 占位状态
        self._prev_hud = {}
        self.cipher_detector = cipher_detector or CipherMachineDetector()

    def on_event(self, cb: EventCallback) -> None:
        self._callbacks.append(cb)

    def emit(self, kind: str, detail: dict) -> None:
        for cb in self._callbacks:
            try:
                cb(kind, detail)
            except Exception as e:
                print(f"[perception] event callback error: {e}")

    def on_frame(self, frame: np.ndarray) -> dict:
        """处理一帧，返回本轮检测到的事件列表。占位实现：将来在这里接视觉。

        返回事件列表（可能是多个）。此占位不实际检测，仅保留接口供 M1 落地。
        """
        detection = self.cipher_detector.detect(frame)
        return detection.as_spatial()

    def on_decode_progress(self, progress: Optional[float]) -> None:
        """P8：破译进度消失（被打断）时触发 decode_interrupt 事件。"""
        if progress is None or progress > 0.0:
            return
        self.emit("decode_interrupt", {"progress": 0.0})
