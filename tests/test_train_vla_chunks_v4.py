import json


def _row(*, sample_subgoal="find_cipher", anchor_subgoal="move_to_target"):
    label = {
        "valid": True,
        "segment_id": "seg_000",
        "intent": "travel",
        "subgoal": sample_subgoal,
        "subgoal_source": "rule",
        "subgoal_rule_version": "subgoal.v1",
    }
    anchor_label = dict(label, subgoal=anchor_subgoal)
    return {
        "schema_version": "vla.action_chunk.v4",
        "anchor_frame": 3,
        "episode_id": "ep",
        "mode": "standard",
        "task": {"name": "find_cipher_and_decode", "instruction": "找到密码机",
                 "mode_token": "<mode:standard>"},
        "slow_label": label,
        "observations": {"fps": 30, "history_actions": [], "frames": [
            {"path": f"frames/{index:08d}.jpg", "frame_index": index,
             "timestamp_ns": index * 1000, "slow_label": anchor_label if index == 3 else label}
            for index in (1, 2, 3)
        ]},
        "action_chunk": [
            {"move_dir": 1, "camera_dx": 0, "camera_dy": 0,
             "buttons": [0, 0, 0, 0, 0, 0], "duration_frames": 6}
            for _ in range(4)
        ],
        "alignment": {"mode": "causal_future", "action_delay_frames": 1,
                      "observation_end_frame": 3, "action_start_frame": 5,
                      "action_end_frame": 28, "observation_end_timestamp_ns": 3000,
                      "action_start_timestamp_ns": 4000, "action_end_timestamp_ns": 28000},
        "quality": {"source": "teacher", "outcome": "unknown"},
        "loss_mask": {"slow": 1, "fast": 1},
    }


def test_audit_detects_subgoal_conflict_at_anchor(tmp_path):
    from idv_agent.scripts.train_vla_chunks_v4 import audit

    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    report = audit(path)

    assert report["slow_label_conflict_count"] == 1
    assert report["slow_label_conflicts"][0]["mismatched_fields"] == ["subgoal"]
    assert report["gate_pass"] is False


def test_audit_accepts_matching_subgoal_at_anchor(tmp_path):
    from idv_agent.scripts.train_vla_chunks_v4 import audit

    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(_row(sample_subgoal="move_to_target",
                                    anchor_subgoal="move_to_target")) + "\n", encoding="utf-8")
    report = audit(path)

    assert report["slow_label_conflict_count"] == 0
    assert report["gate_pass"] is True


def test_audit_rejects_invalid_record_and_reports_action_distribution(tmp_path):
    from idv_agent.scripts.train_vla_chunks_v4 import audit

    valid = _row(sample_subgoal="move_to_target", anchor_subgoal="move_to_target")
    invalid = _row(sample_subgoal="move_to_target", anchor_subgoal="move_to_target")
    invalid["action_chunk"][0]["move_dir"] = 99
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join((json.dumps(valid), json.dumps(invalid))) + "\n", encoding="utf-8")

    report = audit(path)

    assert report["record_error_count"] == 1
    assert report["action_distribution"]["move_dir"]["north"] == 4
    assert report["gate_pass"] is False


def test_audit_rejects_action_duration_alignment_mismatch(tmp_path):
    from idv_agent.scripts.train_vla_chunks_v4 import audit

    row = _row(sample_subgoal="move_to_target", anchor_subgoal="move_to_target")
    row["alignment"]["action_end_frame"] = 27
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    report = audit(path)

    assert report["alignment_error_count"] == 1
    assert report["gate_pass"] is False


def test_audit_reports_malformed_json_without_silently_passing(tmp_path):
    from idv_agent.scripts.train_vla_chunks_v4 import audit

    path = tmp_path / "chunks.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")

    report = audit(path)

    assert report["parse_error_count"] == 1
    assert report["gate_pass"] is False


def test_v4_contract_requires_action_end_frame():
    from idv_agent.vla.action_chunk import validate_v4_record

    row = _row(sample_subgoal="move_to_target", anchor_subgoal="move_to_target")
    del row["alignment"]["action_end_frame"]

    try:
        validate_v4_record(row)
    except ValueError as exc:
        assert "action_end_frame" in str(exc)
    else:
        raise AssertionError("missing action_end_frame must be rejected")
