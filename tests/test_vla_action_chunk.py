import json
from pathlib import Path

import pytest

from idv_agent.scripts.build_vla_chunks import build
from idv_agent.vla.action_chunk import (
    ACTION_CHUNK_HORIZON,
    HISTORY_FRAMES,
    VLA_SCHEMA_VERSION,
    VLA_SCHEMA_VERSION_V5,
    validate_record,
    validate_v5_record,
)


def _record():
    return {
        "schema_version": VLA_SCHEMA_VERSION,
        "episode_id": "ep1",
        "anchor_frame": 3,
        "mode": "standard",
        "intent": "decipher",
        "task": {"name": "find_cipher_and_decode", "instruction": "找到密码机",
                 "mode_token": "<mode:standard>"},
        "observations": {
            "frames": [
                {"path": f"frames/{i:08d}.jpg", "frame_index": i, "timestamp_ns": i * 1000}
                for i in (0, 1, 2)
            ],
            "fps": 30,
            "history_actions": [],
        },
        "action_chunk": [
            {"move_dir": 1, "camera_dx": 0,
             "camera_dy": 0, "buttons": [0, 0, 0, 0, 0, 0], "duration_frames": 6}
            for _ in range(4)
        ],
        "quality": {"source": "teacher", "outcome": "success"},
        "alignment": {"mode": "causal_future", "action_delay_frames": 1,
                       "observation_end_frame": 3, "action_start_frame": 5,
                       "observation_end_timestamp_ns": 3000,
                       "action_start_timestamp_ns": 4000,
                       "action_end_timestamp_ns": 9000},
    }


def test_vla_record_contract_is_independent_of_legacy_schema():
    record = _record()
    validate_record(record)
    assert len(record["observations"]["frames"]) == HISTORY_FRAMES
    assert len(record["action_chunk"]) == ACTION_CHUNK_HORIZON


def test_vla_record_rejects_absolute_frame_path():
    record = _record()
    record["observations"]["frames"][0]["path"] = "C:/secret/frame.jpg"
    with pytest.raises(ValueError, match="相对路径"):
        validate_record(record)


def test_build_vla_chunks(tmp_path: Path):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(60):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text(
        "timestamp_ns,kind,code,value\n0,key_down,key:w,1\n1000000000,key_up,key:w,0\n",
        encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{i * 1000}\n" for i in range(60)),
        encoding="utf-8")
    # Raw Input dx/dy are relative pixels.  18 px/frame becomes 0.09 in the
    # extractor (cam_pixel_scale=200), and six frames aggregate to a +1 bucket.
    (session / "mouse_deltas.csv").write_text(
        "timestamp_ns,dx,dy\n" + "".join(f"{i * 1000},18,0\n" for i in range(60)),
        encoding="utf-8")
    (session / "meta.json").write_text(
        json.dumps({"recording_type": "vla_raw", "mode": "standard",
                    "num_frames": 60, "target_fps": 30}), encoding="utf-8")
    output = tmp_path / "vla.jsonl"
    count = build(session, output, stride=3, macro_frames=6, outcome="success")
    assert count == 10
    rec = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    validate_record(rec)
    assert rec["action_chunk"][0]["move_dir"] == 1
    # Six frame-wise 0.09 increments represent a net turn and must not be
    # averaged back into the neutral camera bucket.
    assert rec["action_chunk"][0]["camera_dx"] == 1
    assert rec["alignment"]["action_start_frame"] == 8


def test_build_v5_buckets_camera_from_raw_pixel_displacement(tmp_path: Path):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(80):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{i * 1000}\n" for i in range(80)),
        encoding="utf-8",
    )
    (session / "mouse_deltas.csv").write_text(
        "timestamp_ns,dx,dy\n" + "".join(f"{i * 1000},3,0\n" for i in range(80)),
        encoding="utf-8",
    )
    (session / "intent_segments.csv").write_text(
        "segment_id,start_frame,end_frame,intent\nseg_000,0,79,travel\n", encoding="utf-8"
    )
    (session / "meta.json").write_text(
        json.dumps({"recording_type": "vla_raw", "mode": "standard",
                    "num_frames": 80, "target_fps": 30}), encoding="utf-8")

    output = tmp_path / "vla_v5.jsonl"
    build(session, output, stride=3, history=8, macro_frames=6,
          schema_version=VLA_SCHEMA_VERSION_V5)

    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    validate_v5_record(record)
    assert record["action_chunk"][0]["camera_dx_px"] == 18.0
    assert record["action_chunk"][0]["camera_dx"] == 1


def test_v5_separates_anchor_sampling_from_history_sampling(tmp_path: Path):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(140):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{i * 1000}\n" for i in range(140)),
        encoding="utf-8",
    )
    (session / "mouse_deltas.csv").write_text(
        "timestamp_ns,dx,dy\n" + "".join(f"{i * 1000},0,0\n" for i in range(140)),
        encoding="utf-8",
    )
    (session / "intent_segments.csv").write_text(
        "segment_id,start_frame,end_frame,intent\nseg_000,0,139,travel\n",
        encoding="utf-8",
    )
    (session / "meta.json").write_text(
        json.dumps({"recording_type": "vla_raw", "mode": "standard",
                    "num_frames": 140, "target_fps": 30}), encoding="utf-8")
    output = tmp_path / "vla_v5.jsonl"

    build(session, output, anchor_stride=12, history_stride=3, history=8,
          macro_frames=6, schema_version=VLA_SCHEMA_VERSION_V5)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert [row["anchor_frame"] for row in records] == [47, 59, 71, 83, 95, 107]
    assert [frame["frame_index"] for frame in records[0]["observations"]["frames"]] == [26, 29, 32, 35, 38, 41, 44, 47]
    assert [frame["frame_index"] for frame in records[1]["observations"]["frames"]] == [38, 41, 44, 47, 50, 53, 56, 59]
    assert len(records[0]["observations"]["history_actions"]) == 8
    assert all("duration_frames" not in action
               for action in records[0]["observations"]["history_actions"])


def test_v5_history_actions_are_six_frame_macro_summaries(tmp_path: Path):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(140):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{i * 1000}\n" for i in range(140)),
        encoding="utf-8",
    )
    (session / "mouse_deltas.csv").write_text(
        "timestamp_ns,dx,dy\n" + "".join(f"{i * 1000},3,0\n" for i in range(80)),
        encoding="utf-8",
    )
    (session / "intent_segments.csv").write_text(
        "segment_id,start_frame,end_frame,intent\nseg_000,0,139,travel\n",
        encoding="utf-8",
    )
    (session / "meta.json").write_text(
        json.dumps({"recording_type": "vla_raw", "mode": "standard",
                    "num_frames": 140, "target_fps": 30}), encoding="utf-8")
    output = tmp_path / "vla_v5.jsonl"

    build(session, output, anchor_stride=12, history_stride=3, history=8,
          macro_frames=6, schema_version=VLA_SCHEMA_VERSION_V5)
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    history = record["observations"]["history_actions"]
    assert record["anchor_frame"] == 47
    assert [action["camera_dx_px"] for action in history] == [18.0] * 8
    assert [action["camera_dx"] for action in history] == [1] * 8


def test_v5_rejects_non_six_duration():
    from idv_agent.vla.action_chunk import validate_v5_record

    slow_label = {"valid": True, "segment_id": "seg_000", "intent": "travel",
                  "subgoal": "move_to_target", "subgoal_source": "rule",
                  "subgoal_rule_version": "subgoal.v1"}
    action = {"move_dir": 0, "camera_dx": 0, "camera_dy": 0,
              "camera_dx_px": 0.0, "camera_dy_px": 0.0,
              "buttons": [0, 0, 0, 0, 0, 0], "duration_frames": 5}
    history_action = {key: value for key, value in action.items()
                      if key != "duration_frames"}
    row = _record()
    row.update({
        "schema_version": VLA_SCHEMA_VERSION_V5,
        "anchor_frame": 23,
        "slow_label": slow_label,
        "loss_mask": {"slow": 1, "fast": 1},
        "action_chunk": [action.copy() for _ in range(ACTION_CHUNK_HORIZON)],
    })
    row.pop("intent")
    row["observations"]["frames"] = [
        {"path": f"frames/{i:08d}.jpg", "frame_index": i,
         "timestamp_ns": i * 1000, "slow_label": slow_label}
        for i in range(16, 24)
    ]
    row["observations"]["history_actions"] = [history_action.copy() for _ in range(8)]
    row["alignment"].update({
        "observation_end_frame": 23,
        "action_start_frame": 25,
        "action_end_frame": 44,
        "observation_end_timestamp_ns": 23000,
        "action_start_timestamp_ns": 25000,
        "action_end_timestamp_ns": 44000,
    })
    try:
        validate_v5_record(row)
    except ValueError as exc:
        assert "duration_frames" in str(exc)
    else:
        raise AssertionError("v5 duration must remain fixed at six frames")


def test_build_rejects_nonpositive_sampling_steps(tmp_path: Path):
    with pytest.raises(ValueError, match="stride"):
        build(tmp_path / "missing", tmp_path / "vla.jsonl", anchor_stride=0)


def test_v5_builder_rejects_noncanonical_temporal_protocol(tmp_path: Path):
    with pytest.raises(ValueError, match="history_stride"):
        build(tmp_path / "missing", tmp_path / "vla.jsonl", history=8,
              history_stride=2, schema_version=VLA_SCHEMA_VERSION_V5)
    with pytest.raises(ValueError, match="macro_frames"):
        build(tmp_path / "missing", tmp_path / "vla.jsonl", history=8,
              macro_frames=5, schema_version=VLA_SCHEMA_VERSION_V5)


def test_build_rejects_missing_per_frame_action_after_extraction(tmp_path: Path, monkeypatch):
    """A missing source action must not be converted to a synthetic stop label."""
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(40):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "mouse_deltas.csv").write_text("timestamp_ns,dx,dy\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{i * 1000}\n" for i in range(40)),
        encoding="utf-8",
    )
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "mode": "standard", "num_frames": 40,
        "target_fps": 30,
    }), encoding="utf-8")

    def write_incomplete_actions(path):
        (path / "per_frame_actions.csv").write_text(
            "frame_idx,timestamp_ns,move_x,move_y,cam_dx,cam_dy,category_name,held_keys\n"
            + "".join(f"{i},{i * 1000},0,1,0,0,MOVE,key:w\n" for i in range(39)),
            encoding="utf-8",
        )

    monkeypatch.setattr("idv_agent.scripts.build_vla_chunks.extract_session", write_incomplete_actions)
    with pytest.raises(ValueError, match="覆盖|缺少"):
        build(session, tmp_path / "vla.jsonl", history=3)


def test_build_rejects_nonfinite_per_frame_action(tmp_path: Path, monkeypatch):
    session = tmp_path / "ep"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(40):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpg")
    (session / "events.csv").write_text("timestamp_ns,kind,code,value\n", encoding="utf-8")
    (session / "mouse_deltas.csv").write_text("timestamp_ns,dx,dy\n", encoding="utf-8")
    (session / "frame_timestamps.csv").write_text(
        "frame_id,timestamp_ns\n" + "".join(f"{i},{i * 1000}\n" for i in range(40)),
        encoding="utf-8",
    )
    (session / "meta.json").write_text(json.dumps({
        "recording_type": "vla_raw", "mode": "standard", "num_frames": 40,
        "target_fps": 30,
    }), encoding="utf-8")

    def write_nonfinite_actions(path):
        rows = "".join(
            f"{i},{i * 1000},{'nan' if i == 5 else 0},1,0,0,MOVE,key:w\n"
            for i in range(40)
        )
        (path / "per_frame_actions.csv").write_text(
            "frame_idx,timestamp_ns,move_x,move_y,cam_dx,cam_dy,category_name,held_keys\n" + rows,
            encoding="utf-8",
        )

    monkeypatch.setattr("idv_agent.scripts.build_vla_chunks.extract_session", write_nonfinite_actions)
    with pytest.raises(ValueError, match="有限|move_x"):
        build(session, tmp_path / "vla.jsonl", history=3)


def test_mode_token_is_required_and_canonical():
    record = _record()
    record["mode"] = "blackjack"
    with pytest.raises(ValueError, match="mode_token"):
        validate_record(record)
