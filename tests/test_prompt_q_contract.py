import json

import pytest
import torch

from idv_agent.scripts.prepare_mvp_v6 import (
    prepare, export, export_q, validate_labels, migrate_prompt_q, validate_workspace, PROTOCOL,
)
from idv_agent.training.prompt_q_loss import prompt_q_loss
from idv_agent.vla.prompt_q import prompt_q_target
from test_mvp_v6_annotations import _raw


@pytest.mark.parametrize("value,token,mask", [(True, "Q", True), (False, "NO_Q", True), (None, None, False)])
def test_prompt_is_the_sole_q_target(value, token, mask):
    assert prompt_q_target(value) == {"q": value, "token": token, "mask": mask}


def test_prompt_q_export_accepts_tail_frames_repeats_and_pending_navigation(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(split="train", scenario_group="same-recording")
    manifest_path.write_text(json.dumps(manifest))
    path = workspace / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    # End-of-recording positive examples have no complete future action and
    # have already followed a recorded Q edge. Neither condition vetoes Q.
    for row, label in zip(rows[-4:], [None, False, True, True]):
        row["labels"]["facts"]["interact_prompt"] = label
        row["review"]["interact_prompt"] = {"status": "reviewed", "annotator": "a", "reviewer": "b"}
    rows[0]["labels"]["facts"]["interact_prompt"] = False
    rows[0]["review"]["interact_prompt"] = {"status": "reviewed", "annotator": "a", "reviewer": "b"}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "q.jsonl"
    result = export_q(workspace, out)
    exported = [json.loads(line) for line in out.read_text().splitlines()]
    assert result["q_positive"] == 2 and result["q_negative"] == 2
    assert [row["targets"]["token"] for row in exported] == ["NO_Q", "NO_Q", "Q", "Q"]
    assert all(row["masks"]["q"] for row in exported)
    assert all(set(row["model_input"]) == {"image_path", "observed_ns"} for row in exported)
    assert all(row["review"]["status"] == "pending" for row in rows)
    with pytest.raises(ValueError, match="incomplete"):
        export(workspace, tmp_path / "navigation.jsonl")


def test_full_export_q_survives_excluded_navigation_and_replay_disagreement(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    path = workspace / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["review"].update(status="reviewed", annotator="a", reviewer="b")
        labels = row["labels"]
        labels["facts"].update(scope="in_scope", cipher_visibility="unknown", interact_prompt=True)
        labels["decision"].update(reviewed_through_frame=row["frame"])
        labels["action_review"].update(source="exclude")
        # No phase, prompt box, target link or decoding truth is needed for Q.
        validate_labels(row)
    rows[0]["labels"]["action_review"]["q"] = False
    with pytest.raises(ValueError, match="must equal interact_prompt"):
        validate_labels(rows[0])
    rows[0]["labels"]["action_review"]["q"] = None
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = workspace / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(split="train", scenario_group="same-recording")
    manifest_path.write_text(json.dumps(manifest))
    export(workspace, tmp_path / "full.jsonl")
    exported = [json.loads(line) for line in (tmp_path / "full.jsonl").read_text().splitlines()]
    assert all(row["targets"]["action"]["q"] is True and row["masks"]["q"] for row in exported)
    assert all(not row["masks"]["navigation"] for row in exported)
    assert any(row["audit_only"]["replay"] is None for row in exported)
    assert all("last_executed_q_ns" in row["model_input"] for row in exported)


def test_q_loss_reinforces_consecutive_positives_and_masks_unknown():
    logits = torch.zeros(4, requires_grad=True)
    loss = prompt_q_loss(logits, torch.tensor([1., 1., 0., float("nan")]),
                         torch.tensor([True, True, True, False]))
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[1] < 0
    assert logits.grad[2] > 0 and logits.grad[3] == 0


def test_known_absence_is_negative_even_when_human_replay_contains_q(tmp_path):
    session = _raw(tmp_path)
    workspace = tmp_path / "annotations"
    prepare(session, workspace)
    rows = [json.loads(line) for line in (workspace / "decisions.jsonl").read_text().splitlines()]
    row = next(row for row in rows if row["replay"] and row["replay"]["q_down_ns"])
    row["labels"]["facts"].update(scope="in_scope", cipher_visibility="unknown", interact_prompt=False)
    row["labels"]["decision"].update(phase="search", reviewed_through_frame=row["frame"])
    row["labels"]["action_review"].update(source="accept_replay", move=row["replay"]["move_proposal"],
                                           camera_command=row["replay"]["quantized_command_proposal"], q=False)
    validate_labels(row)
    row["labels"]["action_review"]["q"] = True
    with pytest.raises(ValueError, match="must equal interact_prompt"):
        validate_labels(row)


def test_prompt_q_loss_can_supervise_the_deployed_sgan_q_logit():
    from idv_agent.model.state_guided_action import StateGuidedActionNetwork
    torch.manual_seed(23)
    core = StateGuidedActionNetwork(feature_dim=8, hidden_dim=8)
    out = core(features=torch.randn(1, 3, 8),
               valid_mask=torch.ones(1, 3, dtype=torch.bool),
               relative_times_s=torch.tensor([[-0.4, -0.2, 0.0]]),
               execution_feedback=torch.zeros(1, 2))
    # 部署时唯一的 Q 判据就是这个 logit；Q 监督不需要额外的辅助分类头。
    logits = out.q_logits
    assert torch.equal(logits, out.facts.prompt_logits)
    prompt_q_loss(logits, torch.tensor([1.]), torch.ones(1, dtype=torch.bool)).backward()
    assert core.prompt_q.weight.grad.abs().sum() > 0
    assert core.prompt_trunk[0].weight.grad.abs().sum() > 0


def test_migration_preserves_facts_and_old_review_without_touching_raw(tmp_path):
    session = _raw(tmp_path)
    old = tmp_path / "old"
    prepare(session, old)
    manifest_path = old / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["protocol"] = {**PROTOCOL, "id": "m28_sparse_visual_v6_draft1"}
    manifest["protocol"].pop("q_contract")
    manifest_path.write_text(json.dumps(manifest))
    path = old / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["labels"]["facts"]["prompt"] = row["labels"]["facts"].pop("interact_prompt")
    rows[-1]["labels"]["facts"]["prompt"] = True
    rows[-1]["labels"]["action_review"]["q"] = False
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = path.read_bytes()
    new = tmp_path / "new"
    migrate_prompt_q(old, new)
    assert path.read_bytes() == original
    assert validate_workspace(new)["raw_hashes_match"]
    migrated = json.loads((new / "decisions.jsonl").read_text().splitlines()[-1])
    assert migrated["labels"]["facts"]["interact_prompt"] is True
    assert migrated["labels"]["action_review"]["q"] is None
    assert migrated["migration_audit"]["old_action_q"] is False
