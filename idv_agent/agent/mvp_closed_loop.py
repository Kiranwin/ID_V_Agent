"""Evidence format and verifier for the learned MVP closed loop."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

STAGES = ("find_cipher", "approach_cipher", "q_prompt", "decode_entry")

@dataclass(frozen=True)
class ClosedLoopTransition:
    episode_id: str
    step: int
    before: dict[str, Any]
    action: dict[str, Any]
    after: dict[str, Any]
    source: str = "unknown"
    causal: bool = False

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ClosedLoopTransition":
        required = ("episode_id", "step", "before", "action", "after")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"闭环记录缺少字段: {','.join(missing)}")
        if not all(isinstance(row[key], dict) for key in ("before", "action", "after")):
            raise ValueError("before/action/after 必须是对象")
        return cls(str(row["episode_id"]), int(row["step"]), row["before"], row["action"],
                   row["after"], str(row.get("source", "unknown")), bool(row.get("causal", False)))


class ClosedLoopTraceWriter:
    """Append-only writer used by a live sandbox runner.

    The caller must append only after the post-action observation has been
    captured.  This prevents an action log without a resulting observation
    from being mistaken for causal evidence.
    """

    def __init__(self, path: str | Path, *, episode_id: str, source: str = "live_sandbox", causal: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.episode_id = str(episode_id)
        self.source = str(source)
        self.causal = bool(causal)
        self._fh = self.path.open("a", encoding="utf-8")
        self._step = 0

    def append(self, *, before: dict[str, Any], action: dict[str, Any], after: dict[str, Any]) -> None:
        row = {"episode_id": self.episode_id, "step": self._step,
               "before": before, "action": action, "after": after,
               "source": self.source, "causal": self.causal}
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._step += 1

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ClosedLoopTraceWriter":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

def _yes(value: Any) -> bool:
    return value in (True, 1, "yes", "visible", "near", "reachable")

def _distance_rank(value: Any) -> int:
    if isinstance(value, (int, float)):
        return 3 if float(value) <= 0.25 else 2 if float(value) <= 0.55 else 1
    return {"near": 3, "reachable": 3, "medium": 2, "far": 1, "unknown": 0, None: 0}.get(value, 0)

def transition_stages(item: ClosedLoopTransition) -> tuple[str, ...]:
    before, action, after = item.before, item.action, item.after
    stages: list[str] = []
    if not _yes(before.get("visible")) and _yes(after.get("visible")):
        stages.append("find_cipher")
    if _yes(before.get("visible")) and _distance_rank(after.get("distance")) > _distance_rank(before.get("distance")):
        stages.append("approach_cipher")
    buttons = action.get("buttons")
    q_pressed = bool(action.get("interact", action.get("q", buttons[0] if buttons else 0)))
    if _yes(after.get("interact_prompt")):
        stages.append("q_prompt")
    if q_pressed and (after.get("frame_state") == "decoding" or _yes(after.get("decoding"))):
        stages.append("decode_entry")
    return tuple(stages)

def verify_transitions(transitions: Iterable[ClosedLoopTransition]) -> dict[str, Any]:
    rows = list(transitions)
    stage_rows = {stage: [] for stage in STAGES}
    for row in rows:
        for stage in transition_stages(row):
            stage_rows[stage].append(row)
    causal_live = bool(rows) and all(row.causal and row.source == "live_sandbox" for row in rows)
    stage_pass = {stage: bool(stage_rows[stage]) and causal_live for stage in STAGES}
    return {"schema": "mvp.closed_loop_eval.v1", "transitions": len(rows),
            "episodes": sorted({row.episode_id for row in rows}),
            "source": sorted({row.source for row in rows}),
            "causal_live_sandbox": causal_live,
            "stage_counts": {stage: len(stage_rows[stage]) for stage in STAGES},
            "stage_pass": stage_pass,
            "mvp_pass": causal_live and all(stage_pass.values()),
            "failure_reasons": [] if causal_live else ["trace_not_proven_causal_live_sandbox"]}
