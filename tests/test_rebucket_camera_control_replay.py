import json
from pathlib import Path


def test_rebucket_preserves_human_fields_and_updates_only_replay(tmp_path: Path):
    from idv_agent.scripts.rebucket_camera_control_replay import rebucket

    raw_root = tmp_path / "raw"
    session = raw_root / "episode"
    session.mkdir(parents=True)
    (session / "vla_chunks_v5.jsonl").write_text(json.dumps({
        "anchor_frame": 21,
        "action_chunk": [{"camera_dx": -1, "camera_dy": 0}],
    }) + "\n", encoding="utf-8")
    source = tmp_path / "labels.jsonl"
    original = {
        "schema_version": "act.camera_control_annotation.v2", "id": "episode_00000021",
        "split": "train", "session": "episode", "frame": 21, "image_path": "frame.jpg",
        "existing_grounding": {"cipher_bbox_xyxy_norm": None, "target_side": "none",
                               "interact_prompt": 0, "cipher_reachable": 0},
        "replay_camera_dx": -2, "replay_camera_dy": 1,
        "camera_control_phase": "target_align", "camera_target_id": "target_cipher",
        "camera_steering_mode": "target_center", "path_strategy": "unknown",
        "desired_turn_dx": -1, "desired_turn_dy": 0, "status": "complete",
    }
    source.write_text(json.dumps(original) + "\n", encoding="utf-8")
    output = tmp_path / "rebucketed.jsonl"

    report = rebucket(source, raw_root, output)
    actual = json.loads(output.read_text(encoding="utf-8"))

    assert report["changed_rows"] == 1
    assert (actual["replay_camera_dx"], actual["replay_camera_dy"]) == (-1, 0)
    for field in ("camera_control_phase", "camera_target_id", "camera_steering_mode",
                  "path_strategy", "desired_turn_dx", "desired_turn_dy", "status"):
        assert actual[field] == original[field]


def test_rebucket_uses_event_centered_supplement_when_raw_has_no_anchor(tmp_path: Path):
    from idv_agent.scripts.rebucket_camera_control_replay import rebucket

    raw_root = tmp_path / "raw"; (raw_root / "episode").mkdir(parents=True)
    (raw_root / "episode" / "vla_chunks_v5.jsonl").write_text("", encoding="utf-8")
    supplemental = tmp_path / "mvp" / "val"; supplemental.mkdir(parents=True)
    (supplemental / "episode.jsonl").write_text(json.dumps({
        "episode_id": "episode", "anchor_frame": 186,
        "action_chunk": [{"camera_dx": 0, "camera_dy": 1}],
    }) + "\n", encoding="utf-8")
    source = tmp_path / "labels.jsonl"
    source.write_text(json.dumps({
        "schema_version": "act.camera_control_annotation.v2", "id": "episode_00000186",
        "split": "val", "session": "episode", "frame": 186, "image_path": "frame.jpg",
        "existing_grounding": {"cipher_bbox_xyxy_norm": None, "target_side": "none",
                               "interact_prompt": 0, "cipher_reachable": 0},
        "replay_camera_dx": 2, "replay_camera_dy": 0,
        "camera_control_phase": "hold", "camera_target_id": "target_cipher",
        "camera_steering_mode": "hold", "path_strategy": "unknown",
        "desired_turn_dx": 0, "desired_turn_dy": 0, "status": "complete",
    }) + "\n", encoding="utf-8")

    rebucket(source, raw_root, tmp_path / "output.jsonl", supplemental_data=tmp_path / "mvp")
    actual = json.loads((tmp_path / "output.jsonl").read_text(encoding="utf-8"))
    assert (actual["replay_camera_dx"], actual["replay_camera_dy"]) == (0, 1)
