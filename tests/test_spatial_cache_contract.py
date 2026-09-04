from __future__ import annotations

import torch
from PIL import Image
import pytest


def test_spatial_cache_round_trip_preserves_grid_tokens(tmp_path):
    from idv_agent.training.raw_feature_cache import RawFeatureCache

    path = tmp_path / "frame.jpg"
    tokens = torch.arange(64 * 8, dtype=torch.float32).reshape(1, 64, 8)
    cache = RawFeatureCache(tmp_path / "cache", spatial_k=8)
    assert cache.add_shard("session", [path], tokens) == 1
    cache.save()

    loaded = RawFeatureCache(tmp_path / "cache", spatial_k=8)
    assert loaded.index["pooling"] == "spatial_grid_k8_v1"
    assert loaded.get(path).shape == (64, 8)
    assert torch.equal(loaded.get(path), tokens[0])


def test_encode_batch_projects_spatial_cache_without_reencoding(tmp_path):
    from idv_agent.scripts.train_vla import encode_batch
    from idv_agent.training.raw_feature_cache import RawFeatureCache

    class Adapter(torch.nn.Module):
        hidden_size = 1
        spatial_k = 8

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.frames = 0

        def encode_task_once(self, instruction, mode, *, task_id):
            return (instruction, mode)

        def project_raw_features(self, raw, _task):
            assert raw.shape == (1, 64, 1)
            return raw.mean(dim=1) * self.weight

        def encode_frames(self, images, task_cache, *, micro_batch_size=None):
            self.frames += len(images)
            raise AssertionError("spatial cache hit must not re-encode")

    path = tmp_path / "frame.png"
    Image.new("RGB", (1, 1), (1, 0, 0)).save(path)
    cache = RawFeatureCache(tmp_path / "cache", spatial_k=8)
    cache.add_shard("session", [path], torch.ones(1, 64, 1))
    cache.save()
    batch = {"frame_paths": [[path]], "task_instruction": ["task"],
             "mode_id": torch.zeros(1, dtype=torch.long)}
    adapter = Adapter()
    out = encode_batch(adapter, batch, device=torch.device("cpu"), raw_feature_cache=cache)
    assert out.shape == (1, 1, 1)
    assert adapter.frames == 0


def test_spatial_cache_rejects_legacy_pooling_metadata(tmp_path):
    from idv_agent.training.raw_feature_cache import RawFeatureCache

    legacy = RawFeatureCache(tmp_path / "cache")
    legacy.save()
    with pytest.raises(ValueError, match="pooling"):
        RawFeatureCache(tmp_path / "cache", spatial_k=8)
