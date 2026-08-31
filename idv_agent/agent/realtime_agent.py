"""快慢双系统实时部署调度器。

主线程（目标 30Hz）：
    DXGI grab -> Policy.decide -> ActionDecoder -> ActionExecutor

慢线程（异步，秒级）：慢层规划（VLM 或 SlowPlanner）更新 intent + skeleton 到 SharedState。

设计：
- RealtimeAgent 用 Policy 抽象（可 M1 规则 / M2 学习快层 / M3 VLA，P9 可插拔）。
- 不依赖训练好的 ckpt 也能跑：RulePolicy 兜底让闭环先转起来（P1）。
- 接入 MatchMemory + EventDetector：关键事件触发慢层重规划/转点。
- F12 紧急退出 + 退出时 release_all 防止按键卡住。
- 默认 dry-run（只打印命令），--send-input 仅在自定义剧本/训练模式启用（合规红线）。

注：P6 清醒约束——慢层不宣称高频；用事件触发 + 低频轮询，避免慢层占主循环。
"""

from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from idv_agent.agent.action_decoder import ActionDecoder, Command
from idv_agent.agent.action_executor import ActionExecutor
from idv_agent.agent.memory import MatchMemory
from idv_agent.agent.perception import EventDetector
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture
from idv_agent.configs.intent import INTENT_VECTOR_DIM
from idv_agent.model.policy import Policy


@dataclass
class SharedState:
    intent_vector: torch.Tensor = field(
        default_factory=lambda: torch.zeros(INTENT_VECTOR_DIM)
    )
    skeleton_idx: int = 0
    last_plan_time: float = 0.0
    last_frame: Optional[np.ndarray] = None
    spatial: dict = field(default_factory=dict)   # 感知结构化状态（规则 agent 用）
    lock: threading.Lock = field(default_factory=threading.Lock)


class LatencyTracker:
    def __init__(self, window: int = 100):
        self._stages: dict[str, deque[float]] = {}
        self._window = window

    def add(self, stage: str, ms: float) -> None:
        if stage not in self._stages:
            self._stages[stage] = deque(maxlen=self._window)
        self._stages[stage].append(ms)

    def summary(self) -> dict:
        out = {}
        for stage, dq in self._stages.items():
            arr = np.array(dq)
            out[stage] = {
                "mean": float(arr.mean()),
                "p50": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "max": float(arr.max()),
            }
        return out


class RealtimeAgent:
    def __init__(
        self,
        policy: Policy,                  # Policy 实现（Rule / Learned）
        executor: ActionExecutor,
        capture_config: CaptureConfig,
        target_fps: int = 30,
        memory: Optional[MatchMemory] = None,
        slow_planner=None,               # Optional[SlowPlanner]
        cipher_template: Optional[str] = None,
        yolo_model_path: Optional[str] = None,
        cipher_detection_interval_s: float = 4.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.policy = policy
        self.executor = executor
        self.capture_config = capture_config
        self.target_fps = target_fps
        self.memory = memory or MatchMemory()
        self.slow_planner = slow_planner
        self.device = device
        if cipher_detection_interval_s < 0.5:
            raise ValueError("cipher_detection_interval_s 必须 >= 0.5s")
        self.cipher_detection_interval_s = float(cipher_detection_interval_s)
        # 需要 .fast 字段的 Policy（LearnedPolicy）需要 decoder 和 skeleton 注入——
        # 由调用方在构造 LearnedPolicy 时已持有模型。decoder 在此统一构造。
        from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP
        self.decoder = ActionDecoder(DEFAULT_SURVIVOR_KEYMAP)

        self.shared = SharedState()
        from idv_agent.agent.perception import CipherMachineDetector
        # One detector can combine YOLO with the legacy template/HSV fallback.
        self.perception = EventDetector(
            on_event=self._on_event,
            cipher_detector=CipherMachineDetector(
                template_path=cipher_template,
                yolo_model_path=yolo_model_path,
            ) if (cipher_template or yolo_model_path) else None,
        )
        self.stop_event = threading.Event()
        self.latency = LatencyTracker()
        self.frame_count = 0
        self.slow_count = 0
        self._slow_thread: Optional[threading.Thread] = None
        self._last_cipher_detection_at: float = 0.0

    def _on_event(self, kind: str, detail: dict) -> None:
        """感知事件 → 写记忆 → 触发(慢层重规划由调用方决定)。"""
        if self.memory is not None:
            from idv_agent.agent.memory import MemoryEvent
            self.memory.add_event(MemoryEvent(kind=kind, detail=str(detail)))

    def _slow_loop(self, period_s: float) -> None:
        if self.slow_planner is None:
            return
        import cv2
        from PIL import Image
        while not self.stop_event.is_set():
            time.sleep(0.05)
            now = time.perf_counter()
            if now - self.shared.last_plan_time < period_s:
                continue
            with self.shared.lock:
                frame = self.shared.last_frame
            if frame is None:
                continue
            try:
                t0 = time.perf_counter()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                intent_vec, sk_idx, intent_name, raw = self.slow_planner.plan(pil)
                with self.shared.lock:
                    self.shared.intent_vector = torch.tensor(intent_vec, dtype=torch.float32)
                    self.shared.skeleton_idx = int(sk_idx)
                self.latency.add("slow_forward", (time.perf_counter() - t0) * 1000)
                self.shared.last_plan_time = now
                self.slow_count += 1
                print(f"[slow] {intent_name} {raw[:40]}")
            except Exception as e:
                print(f"[slow] error: {e}")
                self.shared.last_plan_time = now

    def run(self, max_frames: Optional[int] = None, slow_period_s: float = 2.0) -> None:
        """运行主循环。真实注入与否由构造时 executor 的 dry_run 决定（合规：仅沙盒）。"""
        # F12 紧急退出
        hotkeys = None
        try:
            from pynput import keyboard
            def _on_quit():
                print("\n[F12] 紧急退出")
                self.stop_event.set()
            hotkeys = keyboard.GlobalHotKeys({"<f12>": _on_quit})
            hotkeys.start()
        except Exception:
            print("[warn] 全局热键不可用；用 Ctrl+C 退出")

        # 慢线程
        if self.slow_planner is not None:
            self._slow_thread = threading.Thread(
                target=self._slow_loop, args=(slow_period_s,), name="slow-planner", daemon=True
            )
            self._slow_thread.start()

        frame_interval = 1.0 / self.target_fps
        next_deadline = time.perf_counter()

        with ScreenCapture(self.capture_config) as cap:
            try:
                while not self.stop_event.is_set():
                    if max_frames is not None and self.frame_count >= max_frames:
                        break
                    now = time.perf_counter()
                    if now < next_deadline:
                        time.sleep(max(0, next_deadline - now))
                    next_deadline += frame_interval

                    t_start = time.perf_counter()
                    t0 = time.perf_counter()
                    frame = cap.grab()
                    self.latency.add("grab", (time.perf_counter() - t0) * 1000)
                    if frame is None:
                        with self.shared.lock:
                            frame = self.shared.last_frame
                        if frame is None:
                            continue
                    else:
                        with self.shared.lock:
                            self.shared.last_frame = frame

                    # 密码机感知由慢层低频执行；快层复用最近结果，避免每帧
                    # 模板匹配。首次帧立即检测，默认间隔 4 秒（可调 3~5s）。
                    detect_now = (
                        self._last_cipher_detection_at <= 0.0
                        or time.perf_counter() - self._last_cipher_detection_at
                        >= self.cipher_detection_interval_s
                    )
                    if detect_now:
                        spatial = self.perception.on_frame(frame)
                        self._last_cipher_detection_at = time.perf_counter()
                        with self.shared.lock:
                            self.shared.spatial = dict(spatial)
                        print(
                            f"[cipher] detect visible={spatial.get('visible')} "
                            f"position={spatial.get('position')} "
                            f"distance={spatial.get('distance')} "
                            f"confidence={spatial.get('confidence', 0.0):.3f} "
                            f"source={spatial.get('source', 'none')}"
                        )
                        from idv_agent.agent.memory import MemoryEvent
                        import json
                        self.memory.add_event(MemoryEvent(
                            kind="cipher_detection",
                            position=spatial.get("position"),
                            detail=json.dumps(spatial, ensure_ascii=False),
                        ))

                    # 决策
                    t0 = time.perf_counter()
                    with self.shared.lock:
                        state = {
                            "intent_vector": self.shared.intent_vector.tolist(),
                            "skeleton_idx": int(self.shared.skeleton_idx),
                            "spatial": dict(self.shared.spatial),
                            "memory": self.memory.summary(),
                        }
                    out = self.policy.decide(frame, state)
                    self.latency.add("decision", (time.perf_counter() - t0) * 1000)

                    # 解码 + 执行
                    cmds = self.decoder.decode(out.category_id, out.continuous)
                    self.executor.execute(cmds)
                    self.latency.add("total", (time.perf_counter() - t_start) * 1000)
                    self.frame_count += 1
            finally:
                cleanup = self.decoder.release_all()
                self.executor.execute(cleanup)
                self.executor.shutdown()
                self.stop_event.set()
                if self._slow_thread is not None:
                    self._slow_thread.join(timeout=2.0)
                if hotkeys is not None:
                    hotkeys.stop()
