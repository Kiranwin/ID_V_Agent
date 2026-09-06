from __future__ import annotations

import json


def test_alignment_audit_groups_camera_replay_by_observation_grounding(tmp_path):
    from idv_agent.scripts.audit_grounding_camera_alignment import audit

    session = tmp_path / "raw" / "ep"
    (session / "frames").mkdir(parents=True)
    for frame in (0, 3, 6, 9, 12, 15, 18, 21):
        (session / "frames" / f"{frame:08d}.jpg").write_bytes(b"x")
    row = {
        "schema_version": "vla.action_chunk.v5", "episode_id": "ep", "anchor_frame": 21,
        "source_root": str(session), "mode": "standard",
        "task": {"instruction": "find", "mode_token": "<mode:standard>"},
        "observations": {"fps": 30., "history_actions": [{"move_dir": 0, "camera_dx": 0, "camera_dy": 0,
                      "camera_dx_px": 0., "camera_dy_px": 0., "buttons": [0] * 6} for _ in range(8)],
          "frames": [{"path": f"frames/{frame:08d}.jpg", "frame_index": frame, "timestamp_ns": frame,
            "slow_label": {"valid": True, "segment_id": "s", "intent": "travel", "subgoal": "move_to_target",
                           "subgoal_source": "rule", "subgoal_rule_version": "subgoal.v1"}} for frame in (0,3,6,9,12,15,18,21)]},
        "action_chunk": [{"move_dir": 0, "camera_dx": 2, "camera_dy": -1,
          "camera_dx_px": 110., "camera_dy_px": -25., "buttons": [0] * 6, "duration_frames": 6}] * 4,
        "alignment": {"mode": "causal_future", "action_delay_frames": 1, "observation_end_frame": 21,
          "action_start_frame": 23, "action_end_frame": 46, "observation_end_timestamp_ns": 21,
          "action_start_timestamp_ns": 23, "action_end_timestamp_ns": 46},
        "slow_label": {"valid": True, "intent": "travel", "subgoal": "move_to_target",
          "subgoal_source": "rule", "subgoal_rule_version": "subgoal.v1"},
        "quality": {"source": "human", "outcome": "success"}, "loss_mask": {"slow": 1, "fast": 1},
    }
    data = tmp_path / "data.jsonl"; data.write_text(json.dumps(row) + "\n", encoding="utf-8")
    labels = tmp_path / "labels.jsonl"; labels.write_text(json.dumps({
        "session": "ep", "frame": 21, "cipher_bbox_xyxy_norm": [.1,.1,.2,.2],
        "cipher_reachable": 1, "target_side": "right", "interact_prompt": 1}) + "\n", encoding="utf-8")
    report = audit(str(data), labels)
    assert report["aligned_chunks"] == 1
    assert report["groups"]["side=right"]["dx"]["2"] == 4
    assert report["groups"]["prompt=1"]["dy"]["-1"] == 4
