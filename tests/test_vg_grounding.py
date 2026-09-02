from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path


def test_vg_dry_run_requires_valid_rows(tmp_path: Path):
    from PIL import Image
    from idv_agent.scripts.train_vg_grounding import train

    image = tmp_path / "frame.jpg"
    Image.new("RGB", (16, 16), "black").save(image)
    data = tmp_path / "vg.jsonl"
    row = {"schema_version": "vg.grounding.v2", "id": "x", "image": str(image),
           "question": "密码机在哪里？", "target": {"source_class": "cipher_visible",
           "bbox_xyxy_norm": [0.1, 0.2, 0.3, 0.4]}, "split": "train"}
    data.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    args = Namespace(data=str(data), init_checkpoint="", model_path="", output_dir=str(tmp_path / "out"),
                     repo_root=".", split=None, val_fraction=0.1, batch_size=1, steps=2,
                     lr=1e-3, box_weight=2.0, max_length=512, device="cpu", seed=42, dry_run=True)
    result = train(args)
    assert result["stage"] == "M2_VG"
    assert result["classes"]["cipher_visible"] == 1
    assert result["train"] + result["val"] == 1


def test_vg_training_requires_parent(tmp_path: Path):
    from idv_agent.scripts.train_vg_grounding import train

    data = tmp_path / "vg.jsonl"
    image = tmp_path / "frame.jpg"
    from PIL import Image
    Image.new("RGB", (16, 16), "black").save(image)
    data.write_text(json.dumps({"image": str(image), "target": None}), encoding="utf-8")
    args = Namespace(data=str(data), init_checkpoint="", model_path="", output_dir=str(tmp_path / "out"),
                     repo_root=".", split=None, val_fraction=0.0, batch_size=1, steps=1,
                     lr=1e-3, box_weight=2.0, max_length=512, device="cpu", seed=42, dry_run=False)
    try:
        train(args)
    except ValueError as exc:
        assert "init-checkpoint" in str(exc)
    else:
        raise AssertionError("VG training must require M1_WK parent")
