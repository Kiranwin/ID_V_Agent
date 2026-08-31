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
    interact_prompt: str = "no"
    decoding_state: str = "no"
    # Normalized geometry.  These are deliberately separate from `distance`:
    # bbox area is not a reliable range estimate when the antenna/HUD changes
    # which part of a 3-D cipher is visible.
    frame_width: int = 0
    frame_height: int = 0

    def as_spatial(self) -> dict:
        x, y, bw, bh = self.bbox or (0, 0, 0, 0)
        width = max(int(self.frame_width), 1)
        height = max(int(self.frame_height), 1)
        center_x = (x + bw / 2) / width if self.bbox else None
        bottom_y = (y + bh) / height if self.bbox else None
        area_ratio = (bw * bh) / (width * height) if self.bbox else 0.0
        return {
            "visible": self.visible,
            "position": self.position,
            "distance": self.distance,
            "confidence": round(float(self.confidence), 3),
            "source": self.source,
            "bbox": list(self.bbox) if self.bbox else None,
            "interact_prompt": self.interact_prompt,
            "decoding_state": self.decoding_state,
            "center_x": round(center_x, 4) if center_x is not None else None,
            "bottom_y": round(bottom_y, 4) if bottom_y is not None else None,
            "bbox_area_ratio": round(area_ratio, 6),
        }


class CipherMachineDetector:
    """轻量密码机候选检测器。

    优先使用同地图模板（避免把场景物体误当密码机）；无模板时使用保守的
    HSV 高亮候选启发式，仅输出低置信度结果，便于离线回放和后续校准。
    """

    def __init__(self, template_path: Optional[str | Path] = None,
                 yolo_model_path: Optional[str | Path] = None,
                 debug: bool = False,
                 template_threshold: float = 0.72,
                 heuristic_threshold: float = 0.82,
                 highlight_threshold: float = 0.78,
                 template_scales: tuple[float, ...] =
                 (0.60, 0.75, 0.90, 1.00, 1.15, 1.35, 1.60)):
        self.template_threshold = template_threshold
        self.debug = bool(debug)
        self.heuristic_threshold = heuristic_threshold
        self.highlight_threshold = highlight_threshold
        self.template_scales = tuple(float(s) for s in template_scales if s > 0)
        self.yolo_model = None
        if yolo_model_path:
            model_path = Path(yolo_model_path)
            if not model_path.is_file():
                raise ValueError(f"YOLO 权重不存在: {model_path}")
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(str(model_path))
            except Exception as exc:
                raise ValueError(f"YOLO 模型加载失败: {model_path}: {exc}") from exc
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
            (x, y, bw, bh), "highlight", frame_width=w, frame_height=h,
        )

    def detect(self, frame: np.ndarray) -> CipherDetection:
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return CipherDetection()
        import cv2
        h, w = frame.shape[:2]
        if self.yolo_model is not None:
            result = self._detect_yolo(frame)
            if result is not None:
                return result
        if self.template is not None:
            # A 3-D machine changes apparent size with distance.  Try a small
            # set of scales; this remains cheap for a single crop and avoids
            # requiring a separate template for every camera distance.
            best = None
            for scale in self.template_scales:
                tw = max(2, int(round(self.template.shape[1] * scale)))
                th = max(2, int(round(self.template.shape[0] * scale)))
                if th > h or tw > w:
                    continue
                templ = cv2.resize(self.template, (tw, th), interpolation=cv2.INTER_AREA)
                # CCOEFF_NORMED is undefined for uniform templates (e.g. a
                # tightly cropped icon/patch), so use squared difference in
                # that case and convert to a higher-is-better score.
                channel_std = np.std(templ.reshape(-1, templ.shape[-1]), axis=0)
                method = (cv2.TM_SQDIFF_NORMED
                          if float(np.max(channel_std)) < 1e-3
                          else cv2.TM_CCOEFF_NORMED)
                result = cv2.matchTemplate(frame, templ, method)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                score, loc = ((1.0 - float(min_val), min_loc)
                              if method == cv2.TM_SQDIFF_NORMED
                              else (float(max_val), max_loc))
                if best is None or score > best[0]:
                    best = (score, loc, tw, th)
            if best is not None and best[0] >= self.template_threshold:
                score, loc, tw, th = best
                x, y = int(loc[0]), int(loc[1])
                return CipherDetection("yes", self._location(x + tw/2, w),
                    self._distance(tw*th, w*h), float(score), (x,y,tw,th), "template",
                    frame_width=w, frame_height=h)

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
            self._distance(bw*bh, w*h), conf, (x,y,bw,bh), "heuristic",
            frame_width=w, frame_height=h)

    def _detect_yolo(self, frame: np.ndarray) -> Optional[CipherDetection]:
        """Run the trained four-class detector and merge its UI cues."""
        h, w = frame.shape[:2]
        try:
            outputs = self.yolo_model.predict(frame, verbose=False, conf=0.25, device="0")
        except Exception as exc:
            print(f"[perception] YOLO inference error: {exc}")
            return None
        boxes = getattr(outputs[0], "boxes", None) if outputs else None
        if boxes is None or len(boxes) == 0:
            return None
        candidates = []
        prompt = decode = None
        debug_boxes = []
        names = getattr(self.yolo_model, "names", {}) or {}
        for xyxy, cls_t, conf_t in zip(boxes.xyxy.cpu().tolist(), boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist()):
            cls = int(cls_t); conf = float(conf_t)
            name = str(names.get(cls, cls)) if isinstance(names, dict) else str(cls)
            x1, y1, x2, y2 = [max(0, int(v)) for v in xyxy]
            bw, bh = max(1, x2-x1), max(1, y2-y1)
            debug_boxes.append({"class_id": cls, "name": name, "confidence": round(conf, 3), "bbox": (x1, y1, bw, bh)})
            if cls == 2 or name in {"interact_prompt", "interact_decode"}:
                prompt = (x1, y1, bw, bh, conf)
            elif cls == 3 or name == "decoding_state":
                decode = (x1, y1, bw, bh, conf)
            elif cls in (0, 1) or name in {"cipher_visible", "cipher_highlight"}:
                candidates.append((conf, x1, y1, bw, bh))
        if not candidates and prompt is None and decode is None:
            if self.debug: print(f"[yolo-debug] boxes={debug_boxes} result=none")
            return None
        if candidates:
            conf, x, y, bw, bh = max(candidates)
        else:
            item = prompt or decode
            conf, x, y, bw, bh = item[4], item[0], item[1], item[2], item[3]
        if prompt is not None:
            conf = max(conf, prompt[4])
        if decode is not None:
            conf = max(conf, decode[4])
        detection = CipherDetection(
            visible="yes" if candidates else "no",
            position=self._location(x + bw/2, w),
            distance=self._distance(bw*bh, w*h),
            confidence=conf,
            bbox=(x, y, bw, bh),
            source="yolo",
            interact_prompt="yes" if prompt is not None else "no",
            decoding_state="yes" if decode is not None else "no",
            frame_width=w,
            frame_height=h,
        )
        if self.debug: print(f"[yolo-debug] boxes={debug_boxes} result={detection.as_spatial()} bbox={detection.bbox}")
        return detection


class EventDetector:
    def __init__(self, on_event: Optional[EventCallback] = None,
                 cipher_detector: Optional[CipherMachineDetector] = None,
                 yolo_model_path: Optional[str | Path] = None,
                 yolo_debug: bool = False):
        self._callbacks: list[EventCallback] = []
        if on_event is not None:
            self._callbacks.append(on_event)
        # 占位状态
        self._prev_hud = {}
        self.cipher_detector = cipher_detector or CipherMachineDetector(
            yolo_model_path=yolo_model_path, debug=yolo_debug
        )

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
