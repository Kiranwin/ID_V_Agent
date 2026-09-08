from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_m1_manifest_roundtrip(tmp_path: Path):
    from idv_agent.training.checkpoint_manifest import (
        build_manifest, load_manifest, sha256_file, write_manifest,
    )

    data = tmp_path / "wk.jsonl"
    data.write_text('{"id":"wk_1"}\n', encoding="utf-8")
    manifest = build_manifest(
        stage="M1_WK", parent=None, base_model="Qwen3-VL-4B-Instruct",
        adapters=["lora_wk"], frozen=["vision_tower"],
        trainable=["language_lora:q_proj"],
        data={"wk": str(data), "wk_sha256": sha256_file(data)},
        counts={"train": 1, "val": 0}, schema_versions=["wk.sft.v1"],
        precision="fp32", artifacts={"adapter": "adapter"},
    )
    path = tmp_path / "manifest.json"
    write_manifest(path, manifest)
    loaded = load_manifest(path)
    assert loaded["manifest_schema_version"] == "checkpoint.manifest.v1"
    assert loaded["stage"] == "M1_WK"
    assert loaded["data"]["wk_sha256"] == sha256_file(data)


def test_manifest_rejects_missing_parent_for_downstream():
    from idv_agent.training.checkpoint_manifest import validate_manifest

    with pytest.raises(ValueError, match="必须声明 parent"):
        validate_manifest({
            "manifest_schema_version": "checkpoint.manifest.v1",
            "stage": "M3_ACT", "parent": None, "base_model": "base",
            "adapters": ["lora_wk"], "frozen": [], "trainable": [],
            "data": {}, "counts": {"train": 1, "val": 0},
            "precision": "fp32", "training": {}, "schema_versions": [],
            "artifacts": {},
        })


def test_base_initialized_act_manifest_may_omit_parent_only_with_explicit_marker():
    from idv_agent.training.checkpoint_manifest import validate_manifest

    manifest = {
        "manifest_schema_version": "checkpoint.manifest.v1", "stage": "M3_ACT", "parent": None,
        "base_model": "base", "adapters": ["act_heads"], "frozen": [], "trainable": [],
        "data": {}, "counts": {"train": 1, "val": 0}, "precision": "fp32", "training": {},
        "schema_versions": [], "artifacts": {}, "initialization": "base_without_m2",
    }
    assert validate_manifest(manifest)["initialization"] == "base_without_m2"
