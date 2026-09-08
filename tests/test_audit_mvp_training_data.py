from __future__ import annotations

import json


def _record(session: str, anchor: int, dx: int = 0) -> dict:
    return {
        "schema_version": "vla.action_chunk.v5", "episode_id": session,
        "anchor_frame": anchor, "source_root": "", "mode": "standard",
        "task": {"instruction": "find", "mode_token": "<mode:standard>"},
        "observations": {"frames": [{"path": f"frames/{i:08d}.jpg", "frame_index": i,
                                        "timestamp_ns": i,
                                        "slow_label": {"valid": True, "segment_id": "s", "intent": "travel",
                                                       "subgoal": "move_to_target", "subgoal_source": "rule",
                                                       "subgoal_rule_version": "subgoal.v1"}}
                                    for i in range(anchor - 21, anchor + 1, 3)],
                         "history_actions": [{"move_dir": 0, "camera_dx": 0, "camera_dy": 0,
                                              "camera_dx_px": 0., "camera_dy_px": 0., "buttons": [0] * 6}
                                             for _ in range(8)], "fps": 30.0},
        "action_chunk": [{"move_dir": 0, "camera_dx": dx, "camera_dy": 0,
                          "camera_dx_px": {-2: -110., -1: -25., 0: 0., 1: 25., 2: 110.}[dx],
                          "camera_dy_px": 0., "buttons": [0] * 6, "duration_frames": 6}] * 4,
        "alignment": {"mode": "causal_future", "action_delay_frames": 1,
                      "observation_end_frame": anchor, "action_start_frame": anchor + 2,
                      "action_end_frame": anchor + 25, "observation_end_timestamp_ns": anchor,
                      "action_start_timestamp_ns": anchor + 2, "action_end_timestamp_ns": anchor + 25},
        "slow_label": {"valid": True, "segment_id": "s", "intent": "travel", "subgoal": "move_to_target",
                       "subgoal_source": "rule", "subgoal_rule_version": "subgoal.v1"},
        "quality": {"source": "human", "outcome": "success"}, "loss_mask": {"slow": 1, "fast": 1},
    }


def _control(session: str, frame: int, split: str, replay_dx: int) -> dict:
    return {"schema_version": "act.camera_control_annotation.v2", "id": f"{session}_{frame:08d}",
            "split": split, "session": session, "frame": frame, "status": "complete",
            "camera_control_phase": "target_acquire", "camera_target_id": "target_cipher",
            "camera_steering_mode": "target_center", "path_strategy": "unknown",
            "desired_turn_dx": replay_dx, "desired_turn_dy": 0,
            "replay_camera_dx": replay_dx, "replay_camera_dy": 0}


def _grounding(session: str, frame: int, split: str) -> dict:
    return {"schema_version": "act.grounding_auxiliary.v1", "id": f"{session}_{frame:08d}",
            "split": split, "session": session, "frame": frame, "cipher_bbox_xyxy_norm": None,
            "cipher_reachable": 0, "target_side": "none", "interact_prompt": 0}


def _setup(tmp_path):
    from idv_agent.scripts.audit_mvp_training_data import audit

    dataset = tmp_path / "dataset"
    sources = {}
    for split, session in (("train", "train_ep"), ("val", "val_ep")):
        source = tmp_path / "raw" / session
        (source / "frames").mkdir(parents=True)
        for index in range(50):
            (source / "frames" / f"{index:08d}.jpg").write_bytes(b"x")
        row = _record(session, 21, 1 if split == "train" else -1)
        row["source_root"] = str(source)
        (dataset / split).mkdir(parents=True)
        (dataset / split / f"{session}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        sources[split] = (session, row)
    grounding = tmp_path / "grounding.jsonl"
    grounding.write_text("".join(json.dumps(_grounding(session, 21, split)) + "\n"
                                  for split, (session, _) in sources.items()), encoding="utf-8")
    train_control = tmp_path / "control_train.jsonl"
    train_control.write_text(json.dumps(_control("train_ep", 21, "train", 1)) + "\n", encoding="utf-8")
    val_control = tmp_path / "control_val.jsonl"
    val_control.write_text(json.dumps(_control("val_ep", 21, "val", -1)) + "\n", encoding="utf-8")
    balance = tmp_path / "balance.json"
    balance.write_text(json.dumps({"method": "inverse_sqrt_clamped_mean1", "execution_horizon": 1,
                                   "clamp": [0.35, 3.0], "normalized_per_head": True,
                                   "camera_bucket_edges_px": [12., 67.5],
                                   "camera_bucket_command_px": {"-2": -110., "-1": -25., "0": 0., "1": 25., "2": 110.},
                                   "heads": {"move": {"counts": [1., 0., 0., 0., 0., 0., 0., 0., 0.]},
                                             "camera_dx": {"counts": [0., 0., 0., 1., 0.]},
                                             "camera_dy": {"counts": [0., 0., 1., 0., 0.]}}}), encoding="utf-8")
    return audit, dataset, grounding, train_control, val_control, balance


def test_training_data_audit_accepts_split_isolated_h0_provenance(tmp_path):
    audit, dataset, grounding, train_control, val_control, balance = _setup(tmp_path)
    result = audit(dataset=dataset, grounding_annotations=grounding, control_train=train_control,
                   control_val=val_control, class_balance=balance)
    assert result["gate_pass"]
    assert result["camera_control"]["train_rows"] == 1


def test_training_data_audit_rejects_control_replay_mismatch(tmp_path):
    audit, dataset, grounding, train_control, val_control, balance = _setup(tmp_path)
    row = json.loads(train_control.read_text(encoding="utf-8"))
    row["replay_camera_dx"] = -1
    train_control.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = audit(dataset=dataset, grounding_annotations=grounding, control_train=train_control,
                   control_val=val_control, class_balance=balance)
    assert not result["gate_pass"]
    assert any("replay_camera_dx" in error for error in result["errors"])


def test_training_data_audit_rejects_control_split_leakage(tmp_path):
    audit, dataset, grounding, train_control, val_control, balance = _setup(tmp_path)
    row = json.loads(val_control.read_text(encoding="utf-8"))
    row["split"] = "train"
    val_control.write_text(json.dumps(row) + "\n", encoding="utf-8")
    result = audit(dataset=dataset, grounding_annotations=grounding, control_train=train_control,
                   control_val=val_control, class_balance=balance)
    assert not result["gate_pass"]
    assert any("期望 'val'" in error for error in result["errors"])
