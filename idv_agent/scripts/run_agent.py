"""SGan（State-Guided Action Network，原 M29）实时运行入口。

单一文件包含运行引擎与 CLI：20 FPS 采集线程 + 单槽覆盖 + ≤5 Hz Qwen 视觉推理
线程 + 独立 50 Hz 命令线程。默认 dry-run，只把决策写进 trace，不发送键鼠。

用法：

    python -m idv_agent.scripts.run_agent --mode sgan ^
      --checkpoint checkpoints/m29_q_interact_fix300_20260910/m29.pt ^
      --model-path <Qwen3-VL-4B 权重目录> ^
      --trace reports/sgan_dryrun.jsonl --duration 30

真发送（仅官方自定义剧本/训练模式，且必须显式授权）：

    python -m idv_agent.scripts.run_agent --mode sgan ... ^
      --send-input --allow-undeployed-send-input

字段、依赖脚本、trace 结构与使用样例见 docs/agent.md。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from threading import Event, Lock, Thread

from idv_agent.agent.m29_policy import ExecutionFeedback, M29Policy, M29CommandMerger
from idv_agent.agent.observation_clock import CapturedObservation, LatestObservationSlot
from idv_agent.capture.screen_capture import CaptureConfig, ScreenCapture
from idv_agent.model.m29_checkpoint import load_m29_checkpoint
from idv_agent.training.m29_features import load_m29_encoder


def run(args) -> dict:
    """执行一次运行：采集线程、推理线程与命令时钟，返回 trace header。

    ``args`` 需要 checkpoint / model_path / title / device / duration / trace /
    send_input / enable_navigation / allow_undeployed_send_input。
    """
    if args.trace.exists():
        raise ValueError("trace exists; use a new trace path")
    core, manifest = load_m29_checkpoint(args.checkpoint, device=args.device)
    if args.send_input and not manifest.get("deployable", False) and not getattr(args, "allow_undeployed_send_input", False):
        raise ValueError("SGan checkpoint has not passed readiness gates; add --allow-undeployed-send-input only for an explicitly authorized standard-mode sandbox run")
    encoder = load_m29_encoder(args.model_path or manifest["base_model"], args.device)
    if encoder.adapter.act_feature_dim != core.feature_dim:
        raise ValueError("SGan checkpoint/Qwen feature dimension mismatch")
    stop, capture_done = Event(), Event()
    frame_slot = LatestObservationSlot()
    lock = Lock()
    result = [None]
    errors = []
    counts = {"captures": 0, "encoded": 0, "empty_grabs": 0, "superseded_results": 0}
    feedback = ExecutionFeedback()
    policy = M29Policy(core, encoder, feedback,
                       navigation_enabled=(manifest["task"] == "full" or
                                            bool(getattr(args, "enable_navigation", False))))
    physical = None
    if args.send_input:
        import pydirectinput
        physical = pydirectinput
        physical.PAUSE = 0

    def send(commands):
        for cmd in commands:
            if physical is None:
                continue  # explicit simulation, labelled in trace
            if cmd.kind == "press":
                physical.keyDown(cmd.code.removeprefix("key:"))
            elif cmd.kind == "release":
                physical.keyUp(cmd.code.removeprefix("key:"))
            elif cmd.kind == "mouse_move":
                physical.moveRel(int(cmd.dx_px), int(cmd.dy_px), relative=True)
        return True

    merger = M29CommandMerger(send, feedback)
    started = time.perf_counter()
    deadline = started + args.duration

    def capture_worker():
        try:
            import cv2
            from PIL import Image
            with ScreenCapture(CaptureConfig(fps=20, window_title=args.title)) as capture:
                next_capture = time.perf_counter()
                sequence = 0
                while not stop.is_set() and time.perf_counter() < deadline:
                    remaining = next_capture-time.perf_counter()
                    if remaining > 0:
                        stop.wait(min(remaining, .01)); continue
                    next_capture = max(next_capture+.05, time.perf_counter())
                    frame = capture.grab()
                    captured = time.perf_counter_ns()
                    sequence += 1
                    if frame is None:
                        counts["empty_grabs"] += 1; continue
                    if frame.shape[1] > 1334:
                        frame = cv2.resize(frame, (1334, round(frame.shape[0]*1334/frame.shape[1])), interpolation=cv2.INTER_AREA)
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_slot.put(CapturedObservation(sequence, captured, image))
                    counts["captures"] += 1
        except BaseException as exc:
            errors.append(exc); stop.set()
        finally:
            capture_done.set()

    def inference_worker():
        next_inference = 0.
        try:
            while not stop.is_set():
                remaining = next_inference-time.perf_counter()
                if remaining > 0:
                    stop.wait(min(remaining, .01)); continue
                observation = frame_slot.take()
                if observation is None:
                    if capture_done.is_set(): break
                    stop.wait(.005); continue
                next_inference = time.perf_counter()+.2
                decision = policy.observe(observation)
                counts["encoded"] += 1
                if stop.is_set(): break
                with lock:
                    counts["superseded_results"] += int(result[0] is not None)
                    result[0] = decision
        except BaseException as exc:
            errors.append(exc); stop.set()

    workers = [Thread(target=capture_worker, daemon=True), Thread(target=inference_worker, daemon=True)]
    try:
        for worker in workers: worker.start()
        while not stop.is_set() and time.perf_counter() < deadline:
            with lock:
                decision, result[0] = result[0], None
            merger.tick(time.perf_counter_ns(), decision)
            stop.wait(.02)
    finally:
        stop.set()
        try:
            merger.shutdown()
        finally:
            for worker in workers: worker.join(timeout=2)
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            header = {"schema": "m29.runtime_trace.v1", "source": "live_input" if args.send_input else "dry_run_simulation",
                      "counts": counts, "overwritten_captures": frame_slot.overwritten,
                      "elapsed_s": time.perf_counter()-started,
                      "inference_worker_alive": any(w.is_alive() for w in workers),
                      "errors": [str(error) for error in errors]}
            args.trace.write_text("\n".join(json.dumps(row) for row in [header, *merger.trace])+"\n", encoding="utf-8")
    if errors:
        raise errors[0]
    return header


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m idv_agent.scripts.run_agent",
        description="SGan（原 M29 状态引导策略）实时运行入口；默认 dry-run。",
    )
    p.add_argument("--mode", choices=["sgan"], default="sgan",
                   help="运行架构；只保留 sgan（原 m29），旧的 rule/act 已移除")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="SGan checkpoint 文件（例如 checkpoints/<run>/m29.pt）")
    p.add_argument("--model-path", default=None,
                   help="Qwen3-VL 基础模型目录；省略时读取 checkpoint manifest 的 base_model")
    p.add_argument("--title", default="第五人格",
                   help="被捕获的窗口标题")
    p.add_argument("--device", default="cuda",
                   help="推理设备，例如 cuda")
    p.add_argument("--duration", type=float, default=30.0,
                   help="运行时长（秒）")
    p.add_argument("--trace", type=Path, required=True,
                   help="运行 trace JSONL 输出路径；必须是不存在的路径")
    p.add_argument("--send-input", action="store_true",
                   help="真实发送键鼠；仅官方自定义剧本/训练模式，默认关闭")
    p.add_argument("--allow-undeployed-send-input", action="store_true",
                   help="显式允许 deployable=false 的 checkpoint 真发送；必须与 --send-input 同时使用")
    p.add_argument("--enable-navigation", action="store_true",
                   help="task=q_interact 的 checkpoint 也输出导航动作（dry-run/诊断用）")
    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.duration <= 0:
        p.error("--duration 必须为正数")
    if args.allow_undeployed_send_input and not args.send_input:
        p.error("--allow-undeployed-send-input 必须与 --send-input 同时使用")
    if not args.checkpoint.is_file():
        p.error(f"checkpoint 不存在: {args.checkpoint}")
    if args.trace.exists():
        p.error(f"trace 已存在，请换一个新路径: {args.trace}")

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "SGan 需要 CUDA；当前 torch.cuda.is_available()=False。"
            "请在装有 CUDA 版 PyTorch 的 idv312 环境运行。"
        )

    print(f"[run_agent] mode=sgan, dry_run={not args.send_input}, duration={args.duration}s")
    result = run(args)
    counts = result["counts"]
    print(f"[run_agent] 结束，captures={counts['captures']} encoded={counts['encoded']}")
    print(f"[sgan:trace] {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
