"""M29 inference and one-owner command merger with executed-Q feedback."""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import Lock

import torch

from idv_agent.agent.action_decoder import Command
from idv_agent.agent.observation_clock import CapturedObservation, ObservationActionClock
from idv_agent.model.state_guided_action import CAMERA_COMMANDS, PHASES
from idv_agent.training.m29_dataset import execution_feedback

MOVE_KEYS = ((), ("key:w",), ("key:w", "key:d"), ("key:d",), ("key:s", "key:d"),
             ("key:s",), ("key:s", "key:a"), ("key:a",), ("key:w", "key:a"))


class ExecutionFeedback:
    def __init__(self):
        self._lock = Lock()
        self.last_q_sent_ns = None
        self.q_sent_history = deque(maxlen=128)
        self.last_status = "none"

    def acknowledge(self, *, at_ns: int, q_sent: bool, status: str):
        with self._lock:
            self.last_status = status
            if q_sent:
                self.last_q_sent_ns = at_ns
                self.q_sent_history.append(at_ns)

    def snapshot(self, observed_ns: int):
        with self._lock:
            # Feedback after capture is not observable in this history.
            sent = next((ts for ts in reversed(self.q_sent_history) if ts <= observed_ns), None)
            return execution_feedback(observed_ns, sent)


@dataclass(frozen=True)
class M29Decision:
    sequence: int
    captured_ns: int
    ready_ns: int
    q: bool
    move: int
    camera_dx: int
    camera_dy: int
    phase: str
    prompt_probability: float
    decoding_probability: float
    navigation_ready: bool


class M29Policy:
    """Single inference worker owns model and complete feature history."""
    def __init__(self, core, encoder, feedback: ExecutionFeedback, *, navigation_enabled=True):
        self.core = core.eval()
        self.encoder = encoder
        self.feedback = feedback
        self.device = next(core.parameters()).device
        self.history = deque(maxlen=8)
        self.navigation_enabled = navigation_enabled

    def observe(self, observation: CapturedObservation, *, now_ns=None) -> M29Decision:
        if self.history and observation.captured_ns <= self.history[-1][0]:
            raise ValueError("M29 observed capture timestamps must strictly increase")
        if self.history and observation.captured_ns-self.history[-1][0] > 350_000_000:
            self.history.clear()
        feature = self.encoder.image(observation.payload)[0].to(self.device).float()
        self.history.append((observation.captured_ns, feature))
        current = time.perf_counter_ns() if now_ns is None else now_ns
        times = torch.tensor([[(ts-observation.captured_ns)/1e9 for ts, _ in self.history]], device=self.device)
        inputs = {"features": torch.stack([f for _,f in self.history])[None],
                  "valid_mask": torch.ones((1,len(self.history)), device=self.device, dtype=torch.bool),
                  "relative_times_s": times,
                  "execution_feedback": torch.tensor([self.feedback.snapshot(observation.captured_ns)], device=self.device)}
        with torch.inference_mode():
            output = self.core(**inputs)
        # scalar copies complete CUDA work before ready timestamp is published
        q = float(output.q_logits.sigmoid()[0])
        phase = PHASES[int(output.decision.phase_logits.argmax(-1)[0])]
        decoding = float(output.facts.decoding_logits.sigmoid()[0])
        move = int(output.navigation.move_logits.argmax(-1)[0])
        dx = CAMERA_COMMANDS[int(output.navigation.camera_dx_logits.argmax(-1)[0])]
        dy = CAMERA_COMMANDS[int(output.navigation.camera_dy_logits.argmax(-1)[0])]
        ready = time.perf_counter_ns() if now_ns is None else now_ns
        return M29Decision(observation.sequence, observation.captured_ns, ready, q >= .5,
                           move, dx, dy, phase, q, decoding,
                           self.navigation_enabled and len(self.history) >= 3)


class M29CommandMerger:
    """CPU executor owner. Q is never gated by top-level phase/decoding."""
    def __init__(self, send, feedback: ExecutionFeedback, *, max_age_ns=400_000_000):
        # send(commands)->True only when all commands were actually submitted
        # successfully (or explicitly simulated by a dry-run sink).
        self.send = send
        self.feedback = feedback
        self.clock = ObservationActionClock(max_age_ns=max_age_ns)
        self.consumed = -1
        self.held = set()
        self.trace = []

    def tick(self, now_ns: int, decision: M29Decision | None = None):
        if self.held and self.clock.release_due(now_ns):
            commands = [Command("release", code=key) for key in sorted(self.held)]
            if self.send(commands) is not True:
                raise RuntimeError("M29 key release failed")
            self.held.clear()
            self.trace.append({"event": "action_end", "at_ns": now_ns})
        if decision is None or decision.sequence <= self.consumed:
            return
        self.consumed = decision.sequence
        if not decision.captured_ns <= decision.ready_ns <= now_ns:
            raise ValueError("invalid M29 capture/ready/execute ordering")
        if now_ns-decision.captured_ns > self.clock.max_age_ns:
            self.trace.append({"event": "stale_result", "sequence": decision.sequence})
            return
        commands = []
        navigation = decision.navigation_ready and self.clock.accept(
            CapturedObservation(decision.sequence, decision.captured_ns, None),
            ready_ns=decision.ready_ns, now_ns=now_ns)
        target = self.held
        if navigation:
            if not 0 <= decision.move < len(MOVE_KEYS) or decision.camera_dx not in CAMERA_COMMANDS or decision.camera_dy not in CAMERA_COMMANDS:
                raise ValueError("invalid M29 navigation command")
            target = set(MOVE_KEYS[decision.move])
            commands.extend(Command("release", code=k) for k in sorted(self.held-target))
            commands.extend(Command("press", code=k) for k in sorted(target-self.held))
            if decision.camera_dx or decision.camera_dy:
                commands.append(Command("mouse_move", dx_px=decision.camera_dx, dy_px=decision.camera_dy))
        if decision.q:
            commands.extend((Command("press", code="key:q"), Command("release", code="key:q")))
        if commands:
            try:
                accepted = self.send(commands) is True
            except Exception:
                self.feedback.acknowledge(at_ns=now_ns, q_sent=False, status="send_failed")
                raise
            if not accepted:
                self.feedback.acknowledge(at_ns=now_ns, q_sent=False, status="send_failed")
                raise RuntimeError("M29 command send failed")
        self.held = target
        self.feedback.acknowledge(at_ns=now_ns, q_sent=decision.q, status="submitted")
        self.trace.append({"event": "decision", "sequence": decision.sequence,
                           "captured_ns": decision.captured_ns, "ready_ns": decision.ready_ns,
                           "submitted_ns": now_ns, "phase": decision.phase,
                           "decoding_probability": decision.decoding_probability,
                           "prompt_probability": decision.prompt_probability,
                           "q": decision.q, "navigation_applied": navigation,
                           "move": decision.move, "camera": [decision.camera_dx, decision.camera_dy],
                           "feedback": "Q submitted is an attempt, not decoding success"})

    def shutdown(self):
        # Release all supported movement/Q keys even if a partial send failed
        # before internal held state could be updated.
        ok = self.send([Command("release", code=k) for k in ("key:w", "key:a", "key:s", "key:d", "key:q")])
        self.held.clear()
        if ok is not True:
            raise RuntimeError("M29 shutdown release failed")
