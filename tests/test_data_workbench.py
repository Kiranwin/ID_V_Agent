from __future__ import annotations

import csv
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from pathlib import Path

import pytest


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_valid_session(path: Path, *, frame_count: int = 3) -> Path:
    frames = path / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    for frame_id in range(frame_count):
        (frames / f"{frame_id:08d}.jpg").write_bytes(b"jpeg")
    write_csv(path / "events.csv", ("timestamp_ns", "kind", "code", "value"), [])
    write_csv(path / "mouse_deltas.csv", ("timestamp_ns", "dx", "dy"), [])
    write_csv(
        path / "frame_timestamps.csv",
        ("frame_id", "timestamp_ns"),
        [{"frame_id": i, "timestamp_ns": 100 + i * 100} for i in range(frame_count)],
    )
    (path / "meta.json").write_text(
        json.dumps({
            "recording_type": "vla_raw",
            "vla_schema_version": "vla.action_chunk.v3",
            "mode": "standard",
            "effective_fps": 10.0,
            "num_frames": frame_count,
        }),
        encoding="utf-8",
    )
    return path


def make_v4_ready_session(path: Path) -> Path:
    make_valid_session(path, frame_count=60)
    write_csv(
        path / "intent_segments.csv",
        ("segment_id", "start_frame", "end_frame", "intent", "candidate_reason", "notes"),
        [{"segment_id": "seg_000", "start_frame": 0, "end_frame": 59,
          "intent": "travel", "candidate_reason": "test", "notes": ""}],
    )
    return path


def test_session_store_lists_and_summarizes_session(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore

    make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    store = SessionStore(tmp_path / "sessions")

    assert [item["name"] for item in store.list_sessions()] == ["s1"]
    summary = store.summary("s1")
    assert summary["frame_count"] == 3
    assert summary["frame_start"] == 0
    assert summary["frame_end"] == 2
    assert summary["frame_ids"] == [0, 1, 2]
    assert summary["raw_errors"] == []


def test_session_store_rejects_path_escape(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore

    store = SessionStore(tmp_path / "sessions")
    with pytest.raises(ValueError, match="session"):
        store.resolve_session("..\\outside")


def test_atomic_intent_write_preserves_existing_file_on_validation_error(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore

    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    target = session / "intent_segments.csv"
    target.write_text("original", encoding="utf-8")
    store = SessionStore(tmp_path / "sessions")

    with pytest.raises(ValueError):
        store.atomic_write_intents(
            "s1", [{"start_frame": 0, "end_frame": 2, "intent": "bad"}]
        )
    assert target.read_text(encoding="utf-8") == "original"


def test_extract_actions_refuses_invalid_raw_session(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore
    from idv_agent.data_tools.workbench import DataWorkbench

    session = tmp_path / "sessions" / "bad"
    session.mkdir(parents=True)
    (session / "meta.json").write_text("{}", encoding="utf-8")
    workbench = DataWorkbench(SessionStore(tmp_path / "sessions"))

    with pytest.raises(ValueError, match="raw"):
        workbench.extract_actions("bad")


def test_init_intents_refuses_existing_file_without_overwrite(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore
    from idv_agent.data_tools.workbench import DataWorkbench

    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    write_csv(
        session / "per_frame_actions.csv",
        ("frame_idx", "category_name", "text_action"),
        [{"frame_idx": i, "category_name": "NOOP", "text_action": ""} for i in range(3)],
    )
    (session / "intent_segments.csv").write_text("sentinel", encoding="utf-8")
    workbench = DataWorkbench(SessionStore(tmp_path / "sessions"))

    with pytest.raises(FileExistsError):
        workbench.init_intents("s1")
    assert (session / "intent_segments.csv").read_text(encoding="utf-8") == "sentinel"


def start_test_server(root: Path):
    from idv_agent.scripts.data_workbench import create_server

    server = create_server(root, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def request(server, path: str, *, method: str = "GET", payload: object = None):
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request_obj = Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=body,
        method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        return urlopen(request_obj, timeout=3)
    except HTTPError as exc:
        return exc


def get_json(server, path: str) -> dict:
    response = request(server, path)
    return json.loads(response.read().decode("utf-8"))


def test_http_api_lists_sessions_and_returns_summary(tmp_path: Path):
    make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    server = start_test_server(tmp_path / "sessions")
    try:
        assert get_json(server, "/api/sessions")["sessions"][0]["name"] == "s1"
        summary = get_json(server, "/api/sessions/s1/summary")
        assert summary["frame_count"] == 3
    finally:
        server.shutdown()
        server.server_close()


def test_http_api_rejects_unknown_session(tmp_path: Path):
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/api/sessions/unknown/summary")
        assert response.status == 404
        assert "error" in json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()


def test_http_api_builds_actions_and_candidate_intents(tmp_path: Path):
    make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    server = start_test_server(tmp_path / "sessions")
    try:
        action_response = request(server, "/api/sessions/s1/actions", method="POST", payload={})
        assert action_response.status == 200
        assert json.loads(action_response.read())["action_count"] == 3

        intent_response = request(server, "/api/sessions/s1/intents/init", method="POST", payload={})
        assert intent_response.status == 200
        assert json.loads(intent_response.read())["segment_count"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_workbench_builds_and_audits_vla_chunks_v4(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore
    from idv_agent.data_tools.workbench import DataWorkbench

    session = make_v4_ready_session(tmp_path / "sessions" / "s1")
    result = DataWorkbench(SessionStore(tmp_path / "sessions")).build_vla_chunks_v4("s1")

    assert result["output"] == "vla_chunks_v4.jsonl"
    assert result["audit_output"] == "vla_chunks_v4.audit.json"
    assert result["chunk_count"] > 0
    assert result["gate_pass"] is True
    output = session / result["output"]
    assert output.is_file()
    assert (session / result["audit_output"]).is_file()
    assert (session / "per_frame_actions.csv").is_file()
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(records) == result["chunk_count"]
    assert all(record["schema_version"] == "vla.action_chunk.v4" for record in records)

    summary = SessionStore(tmp_path / "sessions").summary("s1")
    assert summary["v4_chunks"] == {
        "filename": "vla_chunks_v4.jsonl",
        "audit_filename": "vla_chunks_v4.audit.json",
        "exists": True,
        "count": result["chunk_count"],
        "audit_exists": True,
        "audit_gate_pass": True,
        "audit_conflicts": 0,
    }


def test_http_api_builds_and_audits_vla_chunks_v4(tmp_path: Path):
    make_v4_ready_session(tmp_path / "sessions" / "s1")
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/api/sessions/s1/chunks/build", method="POST", payload={})
        body = json.loads(response.read())
        assert response.status == 200
        assert body["output"] == "vla_chunks_v4.jsonl"
        assert body["audit_output"] == "vla_chunks_v4.audit.json"
        assert body["gate_pass"] is True
        assert body["chunk_count"] > 0
    finally:
        server.shutdown()
        server.server_close()


def test_workbench_chunk_build_requires_reviewed_intents(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore
    from idv_agent.data_tools.workbench import DataWorkbench

    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=60)
    workbench = DataWorkbench(SessionStore(tmp_path / "sessions"))
    with pytest.raises(ValueError, match="intent_segments"):
        workbench.build_vla_chunks_v4("s1")


def test_http_api_saves_valid_intents_and_serves_frame(tmp_path: Path):
    make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(
            server,
            "/api/sessions/s1/intents",
            method="PUT",
            payload=[{
                "segment_id": "seg_000",
                "start_frame": 0,
                "end_frame": 2,
                "intent": "travel",
                "candidate_reason": "test",
                "notes": "",
            }],
        )
        assert response.status == 200
        assert json.loads(response.read())["segment_count"] == 1

        frame = request(server, "/api/sessions/s1/frames/0")
        assert frame.status == 200
        assert frame.read() == b"jpeg"
    finally:
        server.shutdown()
        server.server_close()


def test_http_api_rejects_session_path_escape(tmp_path: Path):
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/api/sessions/%2E%2E%2Foutside/summary")
        assert response.status in (400, 404)
        assert "error" in json.loads(response.read())
    finally:
        server.shutdown()
        server.server_close()


def test_http_root_serves_workbench_markup(tmp_path: Path):
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/")
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert 'id="session-list"' in body
        assert 'id="intent-table"' in body
        assert "per-frame-actions" in body
        assert 'id="key-stats"' in body
        assert 'id="event-table"' in body
        assert 'id="frame-viewer"' in body
        assert 'id="frame-input"' in body
        assert 'id="frame-slider"' in body
        assert 'id="frame-previous"' in body
        assert 'id="frame-play"' in body
        assert 'id="frame-next"' in body
        assert 'id="build-chunks"' in body
        assert "Generate vla_chunks_v4" in body
        assert 'id="focused-frame"' not in body
        assert 'id="preview-strip"' not in body
    finally:
        server.shutdown()
        server.server_close()


def test_frame_viewer_supports_direct_frame_jump_and_playback_controls(tmp_path: Path):
    server = start_test_server(tmp_path / "sessions")
    try:
        response = request(server, "/static/app.js")
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert 'frame-input' in body
        assert 'event.key === "Enter"' in body
        assert 'toggleFramePlayback' in body
    finally:
        server.shutdown()
        server.server_close()


def test_session_summary_includes_key_events_mapped_to_nearest_frames(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore

    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=4)
    write_csv(
        session / "events.csv",
        ("timestamp_ns", "kind", "code", "value"),
        [
            {"timestamp_ns": 160, "kind": "key_down", "code": "key:w", "value": 1},
            {"timestamp_ns": 240, "kind": "key_up", "code": "key:w", "value": 0},
            {"timestamp_ns": 360, "kind": "key_down", "code": "key:q", "value": 1},
        ],
    )

    summary = SessionStore(tmp_path / "sessions").summary("s1")

    assert summary["event_count"] == 3
    assert summary["key_event_count"] == 3
    assert summary["key_counts"] == {"W": 2, "Q": 1}
    assert summary["events"][0]["key"] == "W"
    assert summary["events"][0]["frame_id"] == 1


def test_session_summary_merges_consecutive_keydowns_without_touching_raw_counts(tmp_path: Path):
    from idv_agent.data_tools.session_store import SessionStore

    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=5)
    write_csv(
        session / "events.csv",
        ("timestamp_ns", "kind", "code", "value"),
        [
            {"timestamp_ns": 110, "kind": "key_down", "code": "key:w", "value": 1},
            {"timestamp_ns": 120, "kind": "key_down", "code": "key:w", "value": 1},
            {"timestamp_ns": 130, "kind": "key_down", "code": "key:w", "value": 1},
            {"timestamp_ns": 140, "kind": "key_up", "code": "key:w", "value": 0},
        ],
    )

    summary = SessionStore(tmp_path / "sessions").summary("s1")

    assert summary["event_count"] == 4
    assert summary["key_counts"] == {"W": 4}
    assert len(summary["events"]) == 2
    assert summary["events"][0]["kind"] == "key_down"
    assert summary["events"][0]["repeat_count"] == 3
    assert summary["events"][0]["frame_id"] == 0
    assert summary["events"][0]["end_frame_id"] == 0
    assert summary["events"][1]["kind"] == "key_up"


def test_state_replay_accumulates_mouse_deltas_per_frame_window():
    import pandas as pd
    from idv_agent.labels.state_machine import StateReplay

    replay = StateReplay(
        pd.DataFrame(columns=["timestamp_ns", "kind", "code", "value"]),
        mouse_deltas=pd.DataFrame([
            {"timestamp_ns": 110, "dx": 2, "dy": -1},
            {"timestamp_ns": 120, "dx": 3, "dy": 4},
            {"timestamp_ns": 220, "dx": -1, "dy": 2},
        ]),
    )

    first = replay.frame_state(150, 100)
    second = replay.frame_state(250, 150)

    assert (first.mouse_dx, first.mouse_dy) == (5.0, 3.0)
    assert (second.mouse_dx, second.mouse_dy) == (-1.0, 2.0)


def test_http_summary_exposes_event_timeline(tmp_path: Path):
    session = make_valid_session(tmp_path / "sessions" / "s1", frame_count=3)
    write_csv(
        session / "events.csv",
        ("timestamp_ns", "kind", "code", "value"),
        [{"timestamp_ns": 160, "kind": "key_down", "code": "key:q", "value": 1}],
    )
    server = start_test_server(tmp_path / "sessions")
    try:
        summary = get_json(server, "/api/sessions/s1/summary")
        assert summary["key_counts"] == {"Q": 1}
        assert summary["events"][0]["frame_id"] == 1
    finally:
        server.shutdown()
        server.server_close()
