import csv
import json

import pytest
from PIL import Image

from idv_agent.scripts.prepare_mvp_v6 import prepare, read_csv
from idv_agent.scripts.import_mvp_anylabeling import import_completed, apply_navigation, REVIEW_FIELDS
from test_mvp_v6_annotations import _raw


def _workspace(tmp_path):
    session = _raw(tmp_path)
    for image in (session / "frames").glob("*.jpg"):
        Image.new("RGB", (64, 32)).save(image)
        frame = int(image.stem)
        shapes = []
        if frame >= 40:
            shapes.append({"label": "interact_prompt", "shape_type": "rectangle",
                           "points": [[10, 5], [22, 24]]})
        if frame == 12:
            shapes.extend({"label": "cipher_visible", "shape_type": "rectangle",
                           "points": [[x, 5], [x+10, 24]]} for x in (10, 30))
        image.with_suffix(".json").write_text(json.dumps({
            "imagePath": image.name, "imageWidth": 64, "imageHeight": 32,
            "checked": False, "shapes": shapes,
        }), encoding="utf-8")
    workspace = tmp_path / "workspace"
    prepare(session, workspace)
    return session, workspace


def _import(workspace, output):
    return import_completed(workspace, output, annotator="user",
                            completion_note="user confirmed annotation complete",
                            split="train", scenario_group="same-scene")


def test_import_all_frames_and_no_hidden_largest_target_selection(tmp_path):
    session, workspace = _workspace(tmp_path)
    out = tmp_path / "import"
    report = _import(workspace, out)
    assert report["frames"] == 50 and report["q_positive"] == 10 and report["q_negative"] == 40
    q_rows = [json.loads(line) for line in (out / "prompt_q_all_frames.jsonl").read_text().splitlines()]
    assert all(row["targets"]["q"] for row in q_rows[-10:])
    assert all(not row["targets"]["q"] for row in q_rows[:40])
    rows = [json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()]
    multiple = next(row for row in rows if row["frame"] == 12)
    assert len(multiple["visual_import"]["detections"]) == 2
    assert multiple["labels"]["facts"]["target_bbox"] is None
    assert multiple["review"]["status"] == "pending"
    assert multiple["review"]["interact_prompt"]["status"] == "reviewed"
    assert (out / "source_annotations" / "00000012.json").read_bytes() == (session / "frames" / "00000012.json").read_bytes()


def test_missing_json_does_not_become_a_q_negative(tmp_path):
    session, workspace = _workspace(tmp_path)
    (session / "frames" / "00000000.json").unlink()
    out = tmp_path / "import"
    with pytest.raises(ValueError, match="missing is unannotated"):
        _import(workspace, out)
    assert not out.exists()


def test_out_of_bounds_annotation_fails_before_writing(tmp_path):
    session, workspace = _workspace(tmp_path)
    path = session / "frames" / "00000049.json"
    row = json.loads(path.read_text())
    row["shapes"][0]["points"][0][0] = -1
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="outside image"):
        _import(workspace, tmp_path / "import")


def test_review_csv_can_finish_navigation_without_invalidating_q_tail(tmp_path):
    _, workspace = _workspace(tmp_path)
    imported = tmp_path / "import"
    _import(workspace, imported)
    path = imported / "navigation_review.csv"
    edits = read_csv(path)
    for row in edits:
        row.update(reviewed="reviewed", scope="in_scope", target_candidate="unknown",
                   phase="interact" if row["interact_prompt"] == "1" else "search",
                   steering="hold", path="unknown", action_source="exclude", reason="test fixture")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(edits)
    result = apply_navigation(imported, path, tmp_path / "reviewed", reviewer="user")
    assert result["full_export_written"]
    rows = [json.loads(line) for line in (tmp_path / "reviewed" / "training_v6.jsonl").read_text().splitlines()]
    assert not any(row["masks"]["navigation"] for row in rows)
    assert all(row["masks"]["q"] for row in rows)
    assert rows[-1]["targets"]["action"]["q"] is True


def test_top_level_maintain_decoding_uses_visual_state_without_vetoing_fast_q(tmp_path):
    _, workspace = _workspace(tmp_path)
    imported = tmp_path / "import"
    _import(workspace, imported)
    path = imported / "navigation_review.csv"
    edits = read_csv(path)
    for row in edits:
        row.update(reviewed="reviewed", scope="in_scope", target_candidate="unknown",
                   phase="search", steering="hold", path="unknown", action_source="exclude",
                   reason="test fixture", decoding="false")
    # The two labels answer different questions. The prompt-Q contract is
    # still positive if the annotated prompt remains visible in decoding.
    edits[-1].update(phase="maintain_decode", decoding="true",
                     decoding_evidence_frames=json.dumps([int(edits[-1]["frame"])]))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(edits)
    result = apply_navigation(imported, path, tmp_path / "reviewed", reviewer="user")
    assert result["decoding_positive"] == 1
    assert result["decoding_negative"] == len(edits) - 1
    rows = [json.loads(line) for line in (tmp_path / "reviewed" / "training_v6.jsonl").read_text().splitlines()]
    assert rows[-1]["targets"]["facts"]["decoding"] is True
    assert rows[-1]["targets"]["decision"]["phase"] == "maintain_decode"
    assert rows[-1]["targets"]["action"]["q"] is True
    assert rows[-1]["masks"]["decoding"] and rows[-1]["masks"]["q"]


def test_prompt_without_cipher_box_keeps_q_and_auto_excludes_navigation(tmp_path):
    _, workspace = _workspace(tmp_path)
    imported = tmp_path / "import"
    _import(workspace, imported)
    path = imported / "navigation_review.csv"
    edits = read_csv(path)
    row = next(item for item in edits if item["interact_prompt"] == "1" and int(item["frame"]) != 12)
    row.update(reviewed="reviewed", scope="in_scope", target_candidate="unknown",
               phase="interact", steering="hold", path="unknown",
               action_source="accept_replay", reason="")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader(); writer.writerows(edits)
    result = apply_navigation(imported, path, tmp_path / "reviewed", reviewer="user")
    assert result["full_export_written"] is False
    saved = [json.loads(line) for line in (tmp_path / "reviewed" / "decisions.jsonl").read_text().splitlines()]
    linked = next(item for item in saved if item["id"] == row["id"])
    assert linked["labels"]["action_review"]["source"] == "exclude"
    assert linked["labels"]["action_review"]["q"] is None


def test_prompt_without_cipher_box_can_be_excluded_while_q_stays_valid(tmp_path):
    _, workspace = _workspace(tmp_path)
    imported = tmp_path / "import"
    _import(workspace, imported)
    path = imported / "navigation_review.csv"
    edits = read_csv(path)
    row = next(item for item in edits if item["interact_prompt"] == "1" and int(item["frame"]) != 12)
    row.update(reviewed="reviewed", scope="in_scope", target_candidate="unknown",
               phase="interact", steering="hold", path="unknown",
               action_source="exclude", reason="Q-only target box missing")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader(); writer.writerows(edits)
    result = apply_navigation(imported, path, tmp_path / "reviewed", reviewer="user")
    row_out = next(item for item in (tmp_path / "reviewed" / "decisions.jsonl").read_text().splitlines()
                   if '"frame": 40' in item)
    assert json.loads(row_out)["labels"]["action_review"]["source"] == "exclude"
    assert result["full_export_written"] is False


def test_prompt_with_one_cipher_candidate_does_not_require_navigation_link(tmp_path):
    _, workspace = _workspace(tmp_path)
    imported = tmp_path / "import"
    _import(workspace, imported)
    path = imported / "navigation_review.csv"
    edits = read_csv(path)
    row = next(item for item in edits if item["interact_prompt"] == "1" and int(item["frame"]) == 40)
    row.update(reviewed="reviewed", scope="in_scope", target_candidate="unknown",
               phase="interact", steering="hold", path="unknown",
               action_source="exclude", reason="Q-only fixture")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader(); writer.writerows(edits)
    apply_navigation(imported, path, tmp_path / "reviewed", reviewer="user")
    saved = [json.loads(line) for line in (tmp_path / "reviewed" / "decisions.jsonl").read_text().splitlines()]
    linked = next(item for item in saved if item["frame"] == 40)
    assert linked["labels"]["facts"]["target_bbox"] is None
    assert linked["labels"]["facts"]["prompt_target_link"] is None
    assert linked["labels"]["action_review"]["source"] == "exclude"


def test_current_decoding_cannot_use_post_observation_evidence(tmp_path):
    _, workspace = _workspace(tmp_path)
    imported = tmp_path / "import"
    _import(workspace, imported)
    path = imported / "navigation_review.csv"
    edits = read_csv(path)
    edits[0].update(reviewed="reviewed", scope="in_scope", target_candidate="unknown",
                    phase="maintain_decode", steering="hold", path="unknown",
                    action_source="exclude", decoding="true", decoding_evidence_frames="[1]")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(edits)
    with pytest.raises(ValueError, match="never future frames"):
        apply_navigation(imported, path, tmp_path / "reviewed", reviewer="user")
    assert not (tmp_path / "reviewed").exists()
