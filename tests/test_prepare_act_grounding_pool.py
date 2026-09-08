"""Contract tests for the same-session ACT visual-grounding annotation pool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _record(*, session: Path, anchor: int, split: str, q_at_step: int | None = None,
            camera_dx: int = 0, camera_dy: int = 0) -> dict:
    actions = []
    for step in range(4):
        actions.append({
            "move_dir": 0,
            "camera_dx": camera_dx if step == 0 else 0,
            "camera_dy": camera_dy if step == 0 else 0,
            "buttons": [int(step == q_at_step), 0, 0, 0, 0, 0],
            "duration_frames": 6,
        })
    return {
        "schema_version": "vla.action_chunk.v5",
        "episode_id": session.name,
        "anchor_frame": anchor,
        "source_root": str(session),
        "observations": {"frames": [{
            "path": f"frames/{anchor:08d}.jpg", "frame_index": anchor,
        }]},
        "action_chunk": actions,
        "alignment": {
            "observation_end_frame": anchor,
            "action_start_frame": anchor + 2,
            "action_end_frame": anchor + 25,
        },
        "slow_label": {"valid": True, "intent": "travel"},
        # The input document is authoritative for split; this field is only
        # convenient when inspecting the synthetic fixture.
        "fixture_split": split,
    }


def _write_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def _frames(session: Path) -> None:
    frames = session / "frames"
    frames.mkdir(parents=True)
    for index in range(80):
        (frames / f"{index:08d}.jpg").write_bytes(f"frame-{index}".encode())


def test_prepare_pool_preserves_canonical_split_and_records_source_reasons(tmp_path):
    from idv_agent.scripts.prepare_act_grounding_pool import prepare

    raw = tmp_path / "raw"
    train_session, val_session = raw / "train_s", raw / "val_s"
    _frames(train_session)
    _frames(val_session)
    dataset = tmp_path / "mvp"
    # Q is emitted at action_start=12; its full 12 +/- 6 context is required.
    _write_record(dataset / "train" / "train_s.jsonl", _record(
        session=train_session, anchor=10, split="train", q_at_step=0))
    # This independent anchor is retained as a rare, non-zero camera example.
    with (dataset / "train" / "train_s.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_record(session=train_session, anchor=40, split="train", camera_dx=-2)) + "\n")
    _write_record(dataset / "val" / "val_s.jsonl", _record(
        session=val_session, anchor=30, split="val", camera_dy=1))

    output = tmp_path / "annotation_pool"
    report = prepare(dataset, output, camera_per_bucket=1)

    assert report["q_events"] == 1
    assert report["frames"] == {"train": 14, "val": 1}
    assert (output / "images" / "train" / "train_s_00000006.jpg").read_bytes() == b"frame-6"
    assert (output / "images" / "val" / "val_s_00000030.jpg").is_file()
    assert not list((output / "images" / "train").glob("val_s_*.jpg"))
    assert (output / "labels" / "train" / "train_s_00000040.txt").read_text(encoding="utf-8") == ""

    manifest = [json.loads(line) for line in (output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    q_frame = next(row for row in manifest if row["frame"] == 12)
    camera_frame = next(row for row in manifest if row["frame"] == 40)
    assert q_frame["split"] == "train"
    assert q_frame["reasons"] == [{"kind": "q_window", "q_frame": 12, "offset": 0}]
    assert camera_frame["reasons"][0]["kind"] == "camera_nonzero"
    assert camera_frame["reasons"][0]["camera_dx_bucket"] == -2
    assert camera_frame["reasons"][0]["decision_frame"] == 40

    annotations = [json.loads(line) for line in (output / "act_grounding_annotations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {"cipher_bbox_xyxy_norm", "cipher_reachable", "target_side", "interact_prompt"} <= set(annotations[0])
    assert all(row["cipher_bbox_xyxy_norm"] is None for row in annotations)
    assert all(row["target_side"] is None for row in annotations)


def test_prepare_pool_refuses_to_overwrite_existing_annotation_directory(tmp_path):
    from idv_agent.scripts.prepare_act_grounding_pool import prepare

    dataset = tmp_path / "mvp"
    session = tmp_path / "raw" / "s1"
    _frames(session)
    _write_record(dataset / "train" / "s1.jsonl", _record(session=session, anchor=10, split="train"))
    output = tmp_path / "annotation_pool"
    output.mkdir()
    sentinel = output / "human_annotation.txt"
    sentinel.write_text("do-not-overwrite", encoding="utf-8")

    with pytest.raises(ValueError, match="已有文件"):
        prepare(dataset, output, camera_per_bucket=1)
    assert sentinel.read_text(encoding="utf-8") == "do-not-overwrite"


def test_validate_pool_rejects_unfilled_or_invalid_auxiliary_rows(tmp_path):
    from idv_agent.scripts.prepare_act_grounding_pool import validate_annotations

    pool = tmp_path / "pool"
    pool.mkdir()
    manifest = [{"id": "s1_00000010", "split": "train", "session": "s1", "frame": 10}]
    (pool / "manifest.jsonl").write_text(json.dumps(manifest[0]) + "\n", encoding="utf-8")
    blank = {**manifest[0], "schema_version": "act.grounding_auxiliary.v1",
             "cipher_bbox_xyxy_norm": None, "cipher_reachable": None,
             "target_side": None, "interact_prompt": None}
    annotations = pool / "act_grounding_annotations.jsonl"
    annotations.write_text(json.dumps(blank) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="未完成"):
        validate_annotations(pool)

    completed = {**blank, "cipher_bbox_xyxy_norm": [0.1, 0.2, 0.4, 0.8],
                 "cipher_reachable": 1, "target_side": "left", "interact_prompt": 0}
    annotations.write_text(json.dumps(completed) + "\n", encoding="utf-8")
    assert validate_annotations(pool) == {"frames": 1, "splits": {"train": 1, "val": 0}}


def test_populate_annotations_uses_prompt_side_and_treats_missing_sidecar_as_no_cipher(tmp_path):
    from idv_agent.scripts.prepare_act_grounding_pool import (
        populate_annotations_from_yolo, validate_annotations,
    )

    pool = tmp_path / "pool"
    (pool / "labels" / "train").mkdir(parents=True)
    manifest = [
        {"id": "s_00000001", "split": "train", "session": "s", "frame": 1,
         "label": "labels/train/s_00000001.txt"},
        {"id": "s_00000002", "split": "train", "session": "s", "frame": 2,
         "label": "labels/train/s_00000002.txt"},
        {"id": "s_00000003", "split": "train", "session": "s", "frame": 3,
         "label": "labels/train/s_00000003.txt"},
    ]
    (pool / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in manifest), encoding="utf-8")
    # visible cipher at x=.2 plus prompt at x=.8: the prompt controls target_side.
    (pool / "labels" / "train" / "s_00000001.txt").write_text(
        "0 0.200000 0.500000 0.200000 0.400000\n2 0.800000 0.800000 0.100000 0.100000\n",
        encoding="utf-8")
    # A prompt without a cipher box violates the current annotation contract.
    (pool / "labels" / "train" / "s_00000002.txt").write_text(
        "0 0.500000 0.500000 0.200000 0.400000\n2 0.500000 0.800000 0.100000 0.100000\n", encoding="utf-8")
    # Empty label is the user's explicit completed no-cipher label.
    (pool / "labels" / "train" / "s_00000003.txt").write_text("", encoding="utf-8")
    template = [{**row, "schema_version": "act.grounding_auxiliary.v1",
                 "cipher_bbox_xyxy_norm": None, "cipher_reachable": None,
                 "target_side": None, "interact_prompt": None}
                for row in manifest]
    annotations = pool / "act_grounding_annotations.jsonl"
    annotations.write_text("".join(json.dumps(row) + "\n" for row in template), encoding="utf-8")

    report = populate_annotations_from_yolo(pool)
    rows = {row["id"]: row for row in
            (json.loads(line) for line in annotations.read_text(encoding="utf-8").splitlines())}
    assert report == {"frames": 3, "cipher_visible": 2, "cipher_highlight": 0,
                      "interact_prompt": 2, "no_cipher": 1}
    assert rows["s_00000001"]["cipher_bbox_xyxy_norm"] == [0.1, 0.3, 0.3, 0.7]
    assert rows["s_00000001"]["target_side"] == "right"
    assert rows["s_00000001"]["cipher_reachable"] == 1
    assert rows["s_00000002"]["cipher_bbox_xyxy_norm"] == [0.4, 0.3, 0.6, 0.7]
    assert rows["s_00000002"]["target_side"] == "center"
    assert rows["s_00000002"]["cipher_reachable"] == 1
    assert rows["s_00000003"]["cipher_reachable"] == 0
    assert (pool / "act_grounding_annotations.before_yolo_auto.jsonl").is_file()
    assert validate_annotations(pool) == {"frames": 3, "splits": {"train": 3, "val": 0}}


def test_populate_annotations_rejects_prompt_without_visible_cipher(tmp_path):
    from idv_agent.scripts.prepare_act_grounding_pool import populate_annotations_from_yolo

    pool = tmp_path / "pool"
    (pool / "labels" / "train").mkdir(parents=True)
    item = {"id": "s_00000001", "split": "train", "session": "s", "frame": 1,
            "label": "labels/train/s_00000001.txt"}
    (pool / "manifest.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
    (pool / "labels" / "train" / "s_00000001.txt").write_text(
        "2 0.500000 0.800000 0.100000 0.100000\n", encoding="utf-8")
    row = {**item, "schema_version": "act.grounding_auxiliary.v1",
           "cipher_bbox_xyxy_norm": None, "cipher_reachable": None,
           "target_side": None, "interact_prompt": None}
    (pool / "act_grounding_annotations.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="interact_prompt.*密码机框"):
        populate_annotations_from_yolo(pool)
