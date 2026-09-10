import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from idv_agent.scripts.prepare_mvp_v6 import (
    prepare, replay_window, validate_workspace, export,
)


def _raw(root: Path):
    session = root / "ep"
    (session / "frames").mkdir(parents=True)
    for index in range(50):
        (session / "frames" / f"{index:08d}.jpg").write_bytes(b"image")
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "vla_schema_version": "vla.action_chunk.v3",
        "mode": "standard", "target_fps": 20, "num_frames": 50,
        "start_ts_ns": 0, "end_ts_ns": 2_450_000_000,
    }))
    (session / "frame_timestamps.csv").write_text("frame_id,timestamp_ns\n" + "".join(
        f"{index},{index*50_000_000}\n" for index in range(50)))
    (session / "events.csv").write_text(
        "timestamp_ns,kind,code,value\n1000000000,key_down,key:q,1\n1100000000,key_up,key:q,1\n")
    (session / "mouse_deltas.csv").write_text("timestamp_ns,dx,dy\n600000000,25,0\n")
    return session


def test_events_are_integrated_by_time_with_repeat_edges_deduplicated():
    events = [
        {"timestamp_ns": 0, "kind": "key_down", "code": "key:w"},
        {"timestamp_ns": 50, "kind": "key_down", "code": "key:w"},
        {"timestamp_ns": 100, "kind": "key_up", "code": "key:w"},
        {"timestamp_ns": 120, "kind": "key_down", "code": "key:q"},
        {"timestamp_ns": 150, "kind": "key_down", "code": "key:q"},
    ]
    mouse = [{"timestamp_ns": 0, "dx": 9, "dy": 0},
             {"timestamp_ns": 200, "dx": -4, "dy": 0}]
    result = replay_window(events, mouse, 0, 200)
    assert result["key_occupancy"]["key:w"] == 0.5
    assert result["move_proposal"] is None
    assert result["q_down_ns"] == [120]
    assert result["mouse_dx_counts"] == -4


def test_prepare_preserves_raw_and_future_is_not_in_history(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    report = validate_workspace(workspace)
    assert report["reviewed"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        export(workspace, tmp_path / "train.jsonl")
    rows = [json.loads(line) for line in (workspace / "decisions.jsonl").read_text().splitlines()]
    assert all(max(row["history_frames"]) == row["frame"] for row in rows)
    (session / "frames" / "00000000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="raw evidence changed"):
        validate_workspace(workspace)


def test_replay_is_immutable_in_annotation_workspace(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    path = workspace / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["replay"]["mouse_dx_counts"] = 123
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="replay evidence was edited"):
        validate_workspace(workspace)


def test_reviewed_export_separates_future_outcome_from_model_input(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    path = workspace / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["review"] = {"status": "reviewed", "annotator": "human-a", "reviewer": "human-b"}
        labels = row["labels"]
        labels["facts"].update(scope="in_scope", cipher_visibility="absent", interact_prompt=False, decoding=False)
        labels["decision"].update(phase="search", steering="search_sweep", reviewed_through_frame=row["frame"])
        labels["action_review"].update(source="exclude", reason="incomplete demonstration")
        labels["outcome"].update(status="unknown")
    chosen = next(row for row in rows if len(row["history_frames"]) >= 3 and
                  row["replay"] is not None and not row["replay"]["q_down_ns"])
    chosen["labels"]["action_review"].update(source="accept_replay", move=chosen["replay"]["move_proposal"],
                                             camera_command=chosen["replay"]["quantized_command_proposal"], q=False)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(split="train", scenario_group="same-spawn-take1")
    manifest_path.write_text(json.dumps(manifest))
    result = export(workspace, tmp_path / "train.jsonl")
    assert result["reviewed"] == len(rows)
    exported = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert any(row["masks"]["navigation"] for row in exported)
    # Phase is a current top-level label and remains supervised even when the
    # opening frame has too little history for a navigation action target.
    assert all(row["masks"]["phase"] for row in exported)
    assert any(row["masks"]["phase"] and not row["masks"]["navigation"] for row in exported)
    assert all("outcome" not in row["model_input"] and "phase" not in row["model_input"] for row in exported)
    assert all("outcome" in row["audit_only"] for row in exported)


def test_short_history_masks_navigation_but_not_reviewed_phase_or_q(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    path = workspace / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["review"] = {"status": "reviewed", "annotator": "human-a", "reviewer": "human-b"}
        labels = row["labels"]
        labels["facts"].update(scope="in_scope", cipher_visibility="absent",
                               interact_prompt=False, decoding=False)
        labels["decision"].update(phase="search", steering="search_sweep",
                                  reviewed_through_frame=row["frame"])
        labels["action_review"].update(source="exclude", reason="boundary")
        labels["outcome"].update(status="unknown")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(split="train", scenario_group="same-spawn")
    manifest_path.write_text(json.dumps(manifest))
    export(workspace, tmp_path / "train.jsonl")
    exported = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    opening = next(row for row in exported if row["id"].endswith(":00000000"))
    assert opening["masks"]["phase"] is True
    assert opening["masks"]["navigation"] is False
    assert opening["masks"]["q"] is True


def test_q_press_and_truncated_tail_cannot_be_marked_confirmed_decode(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    path = workspace / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    row = rows[-2]
    row["review"] = {"status": "reviewed", "annotator": "a", "reviewer": "b"}
    labels = row["labels"]
    labels["facts"].update(scope="in_scope", cipher_visibility="absent", decoding=False)
    labels["decision"].update(reviewed_through_frame=row["frame"])
    labels["action_review"]["source"] = "exclude"
    labels["outcome"].update(status="confirmed_decode", first_decode_frame=48,
                              confirmed_through_frame=49, evidence=["Q was pressed"])
    path.write_text("".join(json.dumps(item) + "\n" for item in rows))
    with pytest.raises(ValueError, match="post-action evidence|1 second"):
        validate_workspace(workspace)


def test_recorder_retains_pre_input_and_post_q_frames_by_default(tmp_path):
    from idv_agent.scripts.record_vla import _finalize_session
    session = _raw(tmp_path)
    args = SimpleNamespace(fps=20, mode="standard", task_name="decode",
                           task_instruction="decode", window_title="test", max_width=1334,
                           jpeg_quality=90, note="test", _raw_mouse=SimpleNamespace(deltas=[]))
    timestamps = [(index, index * 50_000_000) for index in range(50)]
    assert _finalize_session(args=args, session_id="ep", session_dir=session,
                             frame_timestamps=timestamps, n_frames=50, start_ts=0,
                             end_ts=2_450_000_000) == session
    meta = json.loads((session / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_frames"] == 50
    assert meta["effective_fps"] == 20
    assert meta["idle_boundary_trimmed"] is False
