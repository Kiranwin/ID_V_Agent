"""Checkpoint ``manifest.json`` protocol for the WK/VG/ACT inheritance chain.

The manifest is intentionally small and JSON-only so a later training stage
can validate a parent checkpoint before loading any weights.  It is metadata,
not a replacement for the PEFT adapter config or model files.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


MANIFEST_SCHEMA_VERSION = "checkpoint.manifest.v1"
STAGES = {"M1_WK", "M2_VG", "M3_ACT", "MODE_LORA"}
PARENT_REQUIRED = {"M2_VG", "M3_ACT", "MODE_LORA"}


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"manifest 数据文件不存在: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest 字段 {field} 必须是非空字符串")
    return value


def validate_manifest(manifest: Mapping[str, Any], *, checkpoint_dir: str | Path | None = None,
                      require_artifacts: bool = False) -> dict[str, Any]:
    """Validate and return a normalized copy of a checkpoint manifest."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest 必须是 JSON 对象")
    version = _require_text(manifest.get("manifest_schema_version"), "manifest_schema_version")
    if version != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"不支持的 manifest schema: {version}")
    stage = _require_text(manifest.get("stage"), "stage")
    if stage not in STAGES:
        raise ValueError(f"不支持的 checkpoint stage: {stage}")
    base_model = _require_text(manifest.get("base_model"), "base_model")
    parent = manifest.get("parent")
    if stage in PARENT_REQUIRED and not isinstance(parent, str) or (stage in PARENT_REQUIRED and not str(parent).strip()):
        raise ValueError(f"{stage} 必须声明 parent checkpoint")
    if stage == "M1_WK" and parent not in (None, ""):
        raise ValueError("M1_WK 的 parent 必须为空")
    adapters = manifest.get("adapters")
    if not isinstance(adapters, list) or not all(isinstance(item, str) and item.strip() for item in adapters):
        raise ValueError("manifest.adapters 必须是非空字符串列表")
    if stage == "M1_WK" and "lora_wk" not in adapters:
        raise ValueError("M1_WK 必须包含 lora_wk adapter")
    for field in ("frozen", "trainable", "schema_versions"):
        value = manifest.get(field)
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            raise ValueError(f"manifest.{field} 必须是字符串列表")
    data = manifest.get("data")
    if not isinstance(data, Mapping):
        raise ValueError("manifest.data 必须是对象")
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("manifest.counts 必须是对象")
    for name in ("train", "val"):
        if name in counts and (not isinstance(counts[name], int) or counts[name] < 0):
            raise ValueError(f"manifest.counts.{name} 必须是非负整数")
    if require_artifacts and checkpoint_dir is not None:
        root = Path(checkpoint_dir)
        artifacts = manifest.get("artifacts", {})
        if not isinstance(artifacts, Mapping):
            raise ValueError("manifest.artifacts 必须是对象")
        for name, relative in artifacts.items():
            if not isinstance(relative, str) or not relative.strip():
                raise ValueError(f"manifest.artifacts.{name} 无效")
            if not (root / relative).exists():
                raise ValueError(f"manifest artifact 不存在: {root / relative}")
    return dict(manifest)


def build_manifest(*, stage: str, base_model: str | Path, parent: str | None,
                   adapters: list[str], frozen: list[str], trainable: list[str],
                   data: Mapping[str, Any], counts: Mapping[str, int],
                   schema_versions: list[str], precision: str, training: Mapping[str, Any] | None = None,
                   artifacts: Mapping[str, str] | None = None, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a protocol-compliant manifest with stable provenance fields."""
    if stage not in STAGES:
        raise ValueError(f"不支持的 checkpoint stage: {stage}")
    manifest: dict[str, Any] = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent": parent,
        "base_model": str(base_model),
        "adapters": list(adapters),
        "frozen": list(frozen),
        "trainable": list(trainable),
        "data": dict(data),
        "counts": dict(counts),
        "precision": precision,
        "training": dict(training or {}),
        "schema_versions": list(schema_versions),
        "artifacts": dict(artifacts or {}),
    }
    if extra:
        manifest.update(dict(extra))
    validate_manifest(manifest)
    return manifest


def write_manifest(path: str | Path, manifest: Mapping[str, Any], *, checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    path = Path(path)
    normalized = validate_manifest(manifest, checkpoint_dir=checkpoint_dir, require_artifacts=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return normalized


def load_manifest(path: str | Path, *, require_artifacts: bool = False) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"manifest 不存在: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"manifest JSON 无效: {path}") from exc
    return validate_manifest(value, checkpoint_dir=path.parent, require_artifacts=require_artifacts)
