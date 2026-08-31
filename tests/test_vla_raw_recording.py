import json
from pathlib import Path

from idv_agent.scripts.validate_vla_raw import validate


def test_validate_vla_raw_session(tmp_path: Path):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "mouse_positions.csv").write_text("timestamp_ns,x,y\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text("frame_id,timestamp_ns\n0,100\n", encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({"recording_type": "vla_raw"}), encoding="utf-8")
    assert validate(session) == []


def test_old_session_is_rejected(tmp_path: Path):
    session = tmp_path / "old"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"recording_type": "legacy"}), encoding="utf-8")
    errors = validate(session)
    assert any("vla_raw" in error for error in errors)
