from __future__ import annotations

import json

import pytest


def _row(*, status="complete", phase="target_align", target="target_cipher", dx=1, dy=0):
    return {"schema_version": "act.camera_control_annotation.v1", "id": "ep_00000021",
            "session": "ep", "frame": 21, "status": status,
            "camera_control_phase": phase, "camera_target_id": target,
            "desired_turn_dx": dx, "desired_turn_dy": dy}


def test_camera_control_validator_accepts_complete_control_target(tmp_path):
    from idv_agent.scripts.prepare_camera_control_annotations import validate

    path = tmp_path / "complete.jsonl"
    path.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    assert validate(path) == {"schema": "act.camera_control_annotation.v1", "rows": 1,
                              "complete": 1, "pending": 0, "require_complete": True}


def test_camera_control_validator_rejects_pending_and_invalid_hold(tmp_path):
    from idv_agent.scripts.prepare_camera_control_annotations import validate

    pending = tmp_path / "pending.jsonl"
    pending.write_text(json.dumps(_row(status="pending", phase=None, target=None, dx=None, dy=None)) + "\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="pending"):
        validate(pending)
    assert validate(pending, require_complete=False)["pending"] == 1
    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text(json.dumps(_row(phase="hold", target="target_cipher", dx=1, dy=0)) + "\n",
                       encoding="utf-8")
    with pytest.raises(ValueError, match="hold"):
        validate(invalid)
