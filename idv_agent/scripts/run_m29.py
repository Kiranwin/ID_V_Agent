"""20 Hz capture + <=5 Hz M29 inference + independent 50 Hz command clock."""
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


def run(args):
    if args.trace.exists():
        raise ValueError("trace exists; use a new trace path")
    core, manifest = load_m29_checkpoint(args.checkpoint, device=args.device)
    if args.send_input and not manifest.get("deployable", False) and not getattr(args, "allow_undeployed_send_input", False):
        raise ValueError("M29 checkpoint has not passed readiness gates; add --allow-undeployed-send-input only for an explicitly authorized standard-mode sandbox run")
    encoder = load_m29_encoder(args.model_path or manifest["base_model"], args.device)
    if encoder.adapter.act_feature_dim != core.feature_dim:
        raise ValueError("M29 checkpoint/Qwen feature dimension mismatch")
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--title", default="第五人格")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--send-input", action="store_true")
    parser.add_argument("--allow-undeployed-send-input", action="store_true",
                        help="显式允许当前未标记 deployable 的 checkpoint 进入标准模式真实输入；必须与 --send-input 同时使用")
    parser.add_argument("--enable-navigation", action="store_true",
                        help="允许 q_interact checkpoint 在 dry-run 中输出导航；不会绕过 send-input deployable 门禁")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("duration must be positive")
    if args.allow_undeployed_send_input and not args.send_input:
        parser.error("--allow-undeployed-send-input 必须与 --send-input 同时使用")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
