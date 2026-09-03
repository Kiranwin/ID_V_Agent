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
        "anchor_frame": 3,
        "episode_id": "ep",
        "slow_label": label,
        "observations": {"history_actions": [], "frames": [
            {"frame_index": 3, "slow_label": anchor_label},
        ]},
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
