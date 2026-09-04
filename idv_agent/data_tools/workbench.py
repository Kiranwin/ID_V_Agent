"""Application operations used by the local VLA data workbench."""

from __future__ import annotations

from idv_agent.labels.extract import extract_session
from idv_agent.scripts.init_intent_segments import init as init_intent_segments
from idv_agent.scripts.train_vla_chunks_v5 import build_train
from idv_agent.scripts.validate_intent_segments import validate as validate_intents
from idv_agent.scripts.validate_vla_raw import validate as validate_raw

from .session_store import SessionStore


class DataWorkbench:
    CHUNKS_FILENAME = "vla_chunks_v5.jsonl"

    def __init__(self, store: SessionStore):
        self.store = store

    def sessions(self) -> list[dict]:
        return self.store.list_sessions()

    def session_summary(self, name: str) -> dict:
        return self.store.summary(name)

    def extract_actions(self, name: str) -> int:
        session = self.store.resolve_session(name)
        errors = validate_raw(session)
        if errors:
            raise ValueError("raw session invalid: " + "; ".join(errors))
        return len(extract_session(session))

    def init_intents(self, name: str, *, overwrite: bool = False) -> int:
        session = self.store.resolve_session(name)
        output = session / "intent_segments.csv"
        existing = output if output.exists() else session / "intent_segments.jsonl"
        if existing.exists() and not overwrite:
            raise FileExistsError(str(existing))
        return init_intent_segments(session, output, overwrite=overwrite)

    def build_vla_chunks_v5(self, name: str) -> dict:
        """Build and audit the canonical v5 training file."""
        session = self.store.resolve_session(name)
        errors = validate_raw(session)
        if errors:
            raise ValueError("raw session invalid: " + "; ".join(errors))

        intent_path = session / "intent_segments.csv"
        if not intent_path.is_file():
            intent_path = session / "intent_segments.jsonl"
        if not intent_path.is_file():
            raise ValueError("缺少 intent_segments.csv/jsonl，请先初始化并保存意图片段")

        frame_ids = self.store.summary(name)["frame_ids"]
        validate_intents(
            intent_path,
            frame_start=frame_ids[0] if frame_ids else None,
            frame_end=frame_ids[-1] if frame_ids else None,
            require_full_coverage=True,
        )

        output = session / self.CHUNKS_FILENAME
        report = build_train(
            session,
            output,
            history=8,
            slow_period_s=1.0,
        )
        return {
            "chunk_count": report["chunks"],
            "output": output.name,
            "audit_output": output.with_suffix(".audit.json").name,
            "audit": report,
            "gate_pass": report["gate_pass"],
        }

    build_vla_chunks_v4 = build_vla_chunks_v5
    build_train_vla_chunks = build_vla_chunks_v5
