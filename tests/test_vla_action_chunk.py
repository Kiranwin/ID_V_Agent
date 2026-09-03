import json
from pathlib import Path

import pytest

from idv_agent.scripts.build_vla_chunks import build
from idv_agent.vla.action_chunk import (
    ACTION_CHUNK_HORIZON,
    HISTORY_FRAMES,
    VLA_SCHEMA_VERSION,
    validate_record,
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


def test_mode_token_is_required_and_canonical():
    record = _record()
    record["mode"] = "blackjack"
    with pytest.raises(ValueError, match="mode_token"):
        validate_record(record)
