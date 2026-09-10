"""End-to-end V6 editor contracts, using isolated sessions only."""
import csv
import json
from pathlib import Path

import pytest

from idv_agent.data_tools.review_store import ReviewConflict
from idv_agent.data_tools.session_store import SessionStore
from idv_agent.data_tools.workbench import DataWorkbench
from idv_agent.scripts.prepare_mvp_v6 import raw_manifest
from idv_agent.scripts.import_mvp_anylabeling import normalize_review_edit
from test_import_mvp_anylabeling import _workspace, _import
from test_data_workbench import start_server, request


def setup_editor(tmp_path):
    session, draft = _workspace(tmp_path / "raw_sessions")
    imported = tmp_path / "annotations" / "mvp_v6" / "ep_xany_import_v1"
    _import(draft, imported)
    workbench = DataWorkbench(SessionStore(session.parent))
    return session, imported, workbench


def completed_patch(row):
    return {"scope": "in_scope", "phase": "interact" if row["q_target"] == "1" else "search",
            "target_candidate": "unknown", "target_track_id": "", "steering": "hold",
            "path": "unknown", "action_source": "exclude", "reason": "isolated test",
            "decoding": "unknown", "decoding_evidence_frames": [], "reviewed": "reviewed"}


def test_editor_saves_to_session_csv_restores_and_keeps_raw_and_import_intact(tmp_path):
    session, imported, workbench = setup_editor(tmp_path)
    csv_path = session / "navigation_review_with_state.csv"
    snapshot = imported / "navigation_review_with_state.csv"
    original = csv_path.read_bytes()
    source_hashes = raw_manifest(session)
    data = workbench.review_rows("ep")
    assert Path(data["path"]) == csv_path
    saved = workbench.update_review_row("ep", row_id=data["rows"][0]["id"],
                                       patch={"phase": "search", "reason": "browser draft"},
                                       revision=data["revision"], reviewer="alice")
    assert saved["progress"]["reviewed"] == 0
    reopened = DataWorkbench(SessionStore(session.parent)).review_rows("ep")
    assert reopened["rows"][0]["reason"] == "browser draft"
    assert reopened["revision"] == saved["revision"]
    assert snapshot.read_bytes() == original
    assert raw_manifest(session) == source_hashes
    assert (session / "review_history" / f"{data['revision']}.csv").read_bytes() == original
    with pytest.raises(ReviewConflict):
        workbench.update_review_row("ep", row_id=data["rows"][0]["id"], patch={"reason": "stale tab"},
                                    revision=data["revision"], reviewer="bob")
    assert csv_path.read_bytes() != original
    assert reopened["rows"][0]["reason"] == "browser draft"


@pytest.mark.parametrize("patch", [
    {"q_target": "0"}, {"phase_proposal": "search"}, {"phase": "kite"},
    {"target_candidate": "shape_9999"}, {"decoding_evidence_frames": [999]},
    {"reviewed": "reviewed", "phase": "maintain_decode", "decoding": "unknown"},
])
def test_invalid_patch_never_changes_csv(tmp_path, patch):
    session, _, workbench = setup_editor(tmp_path)
    path = session / "navigation_review_with_state.csv"
    before = path.read_bytes()
    data = workbench.review_rows("ep")
    with pytest.raises(ValueError):
        workbench.update_review_row("ep", row_id=data["rows"][0]["id"], patch=patch,
                                    revision=data["revision"], reviewer="user")
    assert path.read_bytes() == before


def test_full_browser_review_export_uses_saved_session_csv(tmp_path):
    session, imported, workbench = setup_editor(tmp_path)
    original = (imported / "decisions.jsonl").read_bytes()
    assert not workbench.finish_review("ep", reviewer="user")["exported"]
    data = workbench.review_rows("ep")
    for row in data["rows"]:
        data = workbench.update_review_row("ep", row_id=row["id"], patch=completed_patch(row),
                                           revision=data["revision"], reviewer="user")
    assert workbench.check_review("ep", reviewer="user")["pass"]
    result = workbench.finish_review("ep", reviewer="user")
    assert result["exported"]
    exported = [json.loads(line) for line in Path(result["training_file"]).read_text(encoding="utf-8").splitlines()]
    assert all(row["masks"]["q"] for row in exported)
    assert all(not row["masks"]["navigation"] for row in exported)
    assert exported[-1]["targets"]["action"]["q"] is True
    assert (imported / "decisions.jsonl").read_bytes() == original
    assert result["reviewed"] == len(data["rows"])
    another = workbench.finish_review("ep", reviewer="user")
    assert another["training_file"] != result["training_file"]


def test_http_editor_save_reload_conflict_and_retired_endpoints(tmp_path):
    session, _, workbench = setup_editor(tmp_path)
    server = start_server(session.parent)
    prefix = "/api/sessions/ep/mvp-v6"
    try:
        data = json.loads(request(server, prefix + "/rows").read())
        payload = {"row_id": data["rows"][0]["id"], "patch": {"phase": "search"},
                   "revision": data["revision"], "reviewer": "user"}
        response = request(server, prefix + "/save-row", method="POST", payload=payload)
        assert response.status == 200
        assert json.loads(response.read())["rows"][0]["phase"] == "search"
        assert request(server, prefix + "/save-row", method="POST", payload=payload).status == 409
        assert json.loads(request(server, prefix + "/rows").read())["rows"][0]["phase"] == "search"
        check = request(server, prefix + "/check-review", method="POST", payload={"reviewer": "user"})
        assert not json.loads(check.read())["pass"]
        q_export = request(server, prefix + "/export-q-all", method="POST", payload={})
        assert json.loads(q_export.read())["frames"] == 50
        for path in ("/api/sessions/ep/actions", "/api/sessions/ep/intents/init", "/api/sessions/ep/chunks/build"):
            assert request(server, path, method="POST", payload={}).status == 404
        assert request(server, "/static/camera_control.html").status == 404
        assert request(server, "/api/sessions/ep/intents", method="PUT", payload={}).status == 405
        assert request(server, prefix + "/save-row", method="POST", payload=[]).status == 400
        page = request(server, "/").read().decode("utf-8")
        assert 'id="annotation-form"' in page and 'id="save-draft"' in page and 'id="complete"' in page
    finally:
        server.shutdown(); server.server_close()


def test_reopening_import_does_not_overwrite_existing_session_labels(tmp_path):
    session, imported, workbench = setup_editor(tmp_path)
    data = workbench.review_rows("ep")
    workbench.update_review_row("ep", row_id=data["rows"][0]["id"], patch={"reason": "keep me"},
                                revision=data["revision"], reviewer="user")
    from idv_agent.scripts.import_mvp_anylabeling import ensure_session_review_csv
    ensure_session_review_csv(imported)
    assert workbench.review_rows("ep")["rows"][0]["reason"] == "keep me"


def test_boundary_rows_keep_phase_and_q_while_navigation_is_automatically_excluded(tmp_path):
    _, _, workbench = setup_editor(tmp_path)
    data = workbench.review_rows("ep")
    boundary = next(row for row in data["rows"] if "insufficient_history" in row["issues"])
    patch = completed_patch(boundary)
    patch.update(phase="search", action_source="accept_replay", reason="")
    saved = workbench.update_review_row(
        "ep", row_id=boundary["id"], patch=patch,
        revision=data["revision"], reviewer="user",
    )
    row = next(item for item in saved["rows"] if item["id"] == boundary["id"])
    assert row["reviewed"] == "reviewed"
    assert row["phase"] == "search"
    assert row["action_source"] == "exclude"
    assert "自动排除导航监督" in row["reason"]


def test_missing_future_action_window_does_not_block_intent_completion(tmp_path):
    _, _, workbench = setup_editor(tmp_path)
    data = workbench.review_rows("ep")
    boundary = next(row for row in reversed(data["rows"])
                    if "incomplete_action_window" in row["issues"])
    patch = completed_patch(boundary)
    patch.update(phase="interact", action_source="accept_replay", reason="")
    saved = workbench.update_review_row(
        "ep", row_id=boundary["id"], patch=patch,
        revision=data["revision"], reviewer="user",
    )
    row = next(item for item in saved["rows"] if item["id"] == boundary["id"])
    assert row["phase"] == "interact"
    assert row["action_source"] == "exclude"
    assert row["q_target"] in {"0", "1"}


def test_boundary_row_can_complete_with_unknown_phase_without_losing_q(tmp_path):
    _, _, workbench = setup_editor(tmp_path)
    data = workbench.review_rows("ep")
    boundary = next(row for row in data["rows"] if "insufficient_history" in row["issues"])
    patch = completed_patch(boundary)
    patch.update(phase="", action_source="accept_replay", reason="")
    saved = workbench.update_review_row(
        "ep", row_id=boundary["id"], patch=patch,
        revision=data["revision"], reviewer="user",
    )
    row = next(item for item in saved["rows"] if item["id"] == boundary["id"])
    assert row["reviewed"] == "reviewed"
    assert row["phase"] == ""
    assert row["action_source"] == "exclude"
    assert row["q_target"] in {"0", "1"}


def test_prompt_row_automatically_excludes_navigation_but_keeps_interact_phase(tmp_path):
    _, _, workbench = setup_editor(tmp_path)
    data = workbench.review_rows("ep")
    prompt = next(row for row in data["rows"] if row["q_target"] == "1")
    patch = completed_patch(prompt)
    patch.update(phase="interact", action_source="accept_replay", reason="")
    saved = workbench.update_review_row(
        "ep", row_id=prompt["id"], patch=patch,
        revision=data["revision"], reviewer="user",
    )
    row = next(item for item in saved["rows"] if item["id"] == prompt["id"])
    assert row["phase"] == "interact"
    assert row["action_source"] == "exclude"


def test_v6_defaults_are_applied_server_side(tmp_path):
    _, _, workbench = setup_editor(tmp_path)
    data = workbench.review_rows("ep")
    row = next(item for item in data["rows"] if item["q_target"] == "0" and not item["issues"])
    decision = next(item for item in (json.loads(line) for line in (Path(data["workspace"]) / "decisions.jsonl").read_text(encoding="utf-8").splitlines())
                    if item["id"] == row["id"])
    decision["visual_import"]["detections"] = [{"candidate": "shape_0", "label": "cipher_visible",
                                                  "bbox": [0.1, 0.1, 0.2, 0.2]}]
    defaults = normalize_review_edit(decision, {"phase": "search"})
    assert defaults["scope"] == "in_scope"
    assert defaults["steering"] == "search_sweep"
    assert defaults["path"] == "direct"
    assert defaults["action_source"] == "accept_replay"
    assert defaults["target_candidate"] == "shape_0"
    assert defaults["decoding"] == "false"
