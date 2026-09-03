"""Incremental on-disk cache for deterministic frozen vision features."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Iterable

import torch


POOLING_VERSION = "mean_dim1_v1"


def normalize_frame_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve()).replace("\\", "/")


class RawFeatureCache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.index_path = self.root / "index.json"
        self.index = json.loads(self.index_path.read_text(encoding="utf-8")) if self.index_path.is_file() else {
            "schema": "vla.raw_feature_cache.v1", "pooling": POOLING_VERSION,
            "features": {}, "shards": {}, "raw_dim": None, "dtype": None,
        }

    def __contains__(self, path: str | Path) -> bool:
        return normalize_frame_path(path) in self.index["features"]

    def get(self, path: str | Path) -> torch.Tensor:
        key = normalize_frame_path(path)
        entry = self.index["features"][key]
        shard = torch.load(self.root / entry["shard"], map_location="cpu", weights_only=True)
        return shard[entry["offset"]]

    def add_shard(self, session_id: str, paths: Iterable[str | Path], features: torch.Tensor) -> int:
        paths = [normalize_frame_path(p) for p in paths]
        if features.ndim != 2 or len(paths) != features.shape[0]:
            raise ValueError("paths/features shape 不匹配")
        new = [(p, i) for i, p in enumerate(paths) if p not in self.index["features"]]
        if not new:
            return 0
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session_id)[:80] or "session"
        shard_name = f"shard_{safe}_{len(self.index['shards']):05d}.pt"
        selected = torch.stack([features[i].detach().cpu() for _, i in new])
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.root / (shard_name + ".tmp")
        torch.save(selected, tmp)
        os.replace(tmp, self.root / shard_name)
        for offset, (path, _i) in enumerate(new):
            self.index["features"][path] = {"shard": shard_name, "offset": offset}
        self.index["shards"][shard_name] = {"session_id": session_id, "count": len(new)}
        self.index["raw_dim"] = int(features.shape[1])
        self.index["dtype"] = str(features.dtype)
        return len(new)

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.index_path)

    @property
    def count(self) -> int:
        return len(self.index["features"])


def collect_unique_frame_paths(data_paths: Iterable[str | Path]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for data_path in data_paths:
        path = Path(data_path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                root = Path(record.get("source_root", path.parent))
                session = str(root)
                bucket = groups.setdefault(session, [])
                for frame in record["observations"]["frames"]:
                    absolute = normalize_frame_path(root / frame["path"])
                    if absolute not in bucket:
                        bucket.append(absolute)
    return groups
