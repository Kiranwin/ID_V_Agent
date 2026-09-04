import json
from pathlib import Path

from idv_agent.scripts.validate_vla_raw import validate


def test_validate_vla_raw_session(tmp_path: Path):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "mouse_deltas.csv").write_text("timestamp_ns,dx,dy\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text("frame_id,timestamp_ns\n0,100\n", encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard",
    }), encoding="utf-8")
    assert validate(session) == []


def test_old_session_is_rejected(tmp_path: Path):
    session = tmp_path / "old"
    session.mkdir()
    (session / "meta.json").write_text(json.dumps({"recording_type": "legacy"}), encoding="utf-8")
    errors = validate(session)
    assert any("vla_raw" in error for error in errors)


def test_validate_accepts_mouse_deltas_file(tmp_path: Path):
    session = tmp_path / "raw"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    for name, content in {
        "events.csv": "timestamp_ns,kind,code,value\n",
        "mouse_deltas.csv": "timestamp_ns,dx,dy\n100,1,-2\n",
        "frame_timestamps.csv": "frame_id,timestamp_ns\n0,100\n",
    }.items():
        (session / name).write_text(content, encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard", "mouse_input_source": "raw_input",
    }), encoding="utf-8")
    assert validate(session) == []


def test_validate_rejects_malformed_mouse_deltas(tmp_path: Path):
    session = tmp_path / "raw"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    for name, content in {
        "events.csv": "timestamp_ns,kind,code,value\n",
        "mouse_deltas.csv": "timestamp_ns,dx,dy\n",
        "mouse_deltas.csv": "timestamp_ns,dx\n100,1\n",
        "frame_timestamps.csv": "frame_id,timestamp_ns\n0,100\n",
    }.items():
        (session / name).write_text(content, encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard",
    }), encoding="utf-8")
    assert any("mouse_deltas" in error for error in validate(session))


def test_validate_rejects_invalid_events_csv(tmp_path: Path):
    session = tmp_path / "raw"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text(
        "timestamp_ns,kind,code,value\nnot-a-time,key_down,key:w,nan\n",
        encoding="utf-8",
    )
    (session / "mouse_deltas.csv").write_text("timestamp_ns,dx,dy\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n0,100\n", encoding="utf-8"
    )
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard", "start_ts_ns": 0, "end_ts_ns": 1000,
        "num_frames": 1,
    }), encoding="utf-8")

    errors = validate(session)

    assert any("events.csv" in error for error in errors)


def test_validate_rejects_frame_file_id_mismatch(tmp_path: Path):
    session = tmp_path / "raw"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000001.jpg").write_bytes(b"jpg")
    for name, content in {
        "events.csv": "timestamp_ns,kind,code,value\n",
        "mouse_deltas.csv": "timestamp_ns,dx,dy\n",
        "frame_timestamps.csv": "frame_id,timestamp_ns\n0,100\n",
    }.items():
        (session / name).write_text(content, encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard", "num_frames": 1,
    }), encoding="utf-8")

    errors = validate(session)

    assert any("文件名" in error or "frame_id" in error for error in errors)


def test_validate_rejects_nonfinite_mouse_delta(tmp_path: Path):
    session = tmp_path / "raw"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    for name, content in {
        "events.csv": "timestamp_ns,kind,code,value\n",
        "mouse_deltas.csv": "timestamp_ns,dx,dy\n100,nan,1\n",
        "frame_timestamps.csv": "frame_id,timestamp_ns\n0,100\n",
    }.items():
        (session / name).write_text(content, encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard", "num_frames": 1,
    }), encoding="utf-8")

    assert any("有限" in error or "dx" in error for error in validate(session))


def test_validate_rejects_negative_frame_timestamp(tmp_path: Path):
    session = tmp_path / "raw"
    frames = session / "frames"
    frames.mkdir(parents=True)
    (frames / "00000000.jpg").write_bytes(b"jpg")
    for name, content in {
        "events.csv": "timestamp_ns,kind,code,value\n",
        "mouse_deltas.csv": "timestamp_ns,dx,dy\n",
        "frame_timestamps.csv": "frame_id,timestamp_ns\n0,-1\n",
    }.items():
        (session / name).write_text(content, encoding="utf-8")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard",
    }), encoding="utf-8")

    assert any("非负" in error for error in validate(session))
