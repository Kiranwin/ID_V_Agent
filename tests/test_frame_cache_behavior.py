from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
import pytest


class _Adapter(torch.nn.Module):
    hidden_size = 1

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.tasks = []
        self.frames = 0

    def encode_task_once(self, instruction, mode, *, task_id):
        self.tasks.append((instruction, mode, task_id))
        return (instruction, mode)

    def encode_frame(self, image, task_cache):
        self.frames += 1
        value = float(image.getpixel((0, 0))[0])
        return self.scale.reshape(1) * value


def _batch(path: Path, instructions):
    return {
        "frame_paths": [[path, None] for _ in instructions],
        "task_instruction": list(instructions),
        "mode_id": torch.zeros(len(instructions), dtype=torch.long),
    }


def test_encode_batch_reuses_duplicate_frames_with_gradients(tmp_path):
    from idv_agent.scripts.train_vla import encode_batch

    path = tmp_path / "frame.png"
    Image.new("RGB", (1, 1), (3, 0, 0)).save(path)
    adapter = _Adapter()
    features = encode_batch(adapter, _batch(path, ["same", "same"]), device=torch.device("cpu"))

    assert adapter.frames == 1
    features.sum().backward()
    assert adapter.scale.grad is not None
    assert float(adapter.scale.grad) == 6.0


def test_frame_cache_key_includes_task_context(tmp_path):
    from idv_agent.scripts.train_vla import encode_batch

    path = tmp_path / "frame.png"
    Image.new("RGB", (1, 1), (2, 0, 0)).save(path)
    adapter = _Adapter()
    cache = {}
    batch = _batch(path, ["instruction-a", "instruction-b"])
    encode_batch(adapter, batch, device=torch.device("cpu"), frame_cache=cache)

    assert adapter.frames == 2
    assert set(cache) == {
        ("instruction-a", 0, str(path)),
        ("instruction-b", 0, str(path)),
    }


def test_contiguous_subset_preserves_adjacent_records():
    from idv_agent.scripts.train_vla import _contiguous_subset

    dataset = [{"episode_id": "a", "i": i} for i in range(5)] + [{"episode_id": "b", "i": i} for i in range(5)]
    subset = _contiguous_subset(dataset, 4)
    assert [item["i"] for item in subset] == [0, 1, 2, 3]


class _BatchAdapter(_Adapter):
    def __init__(self):
        super().__init__()
        self.batch_calls = []

    def encode_frames(self, images, task_cache, *, micro_batch_size=None):
        self.batch_calls.append(len(images))
        self.frames += len(images)
        values = [float(image.getpixel((0, 0))[0]) for image in images]
        return torch.tensor(values).reshape(-1, 1) * self.scale


def test_encode_batch_uses_batched_vision_encoder_for_uncached_frames():
    from idv_agent.scripts.train_vla import encode_batch

    adapter = _BatchAdapter()
    paths = [Path(f"frame-{index}.png") for index in range(3)]
    # Avoid filesystem I/O: patch the image loader at the module boundary.
    import idv_agent.scripts.train_vla as train_vla
    original = train_vla._load_images
    train_vla._load_images = lambda _paths: [Image.new("RGB", (1, 1), (i + 1, 0, 0))
                                             for i, _ in enumerate(_paths)]
    try:
        batch = {"frame_paths": [paths], "task_instruction": ["same"],
                 "mode_id": torch.zeros(1, dtype=torch.long)}
        features = encode_batch(adapter, batch, device=torch.device("cpu"),
                                vision_micro_batch_size=2)
    finally:
        train_vla._load_images = original

    assert adapter.batch_calls == [2, 1]
    assert torch.equal(features[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))


def test_qwen_batch_singleton_keeps_batch_dimension():
    from types import SimpleNamespace
    from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter
    from idv_agent.model.temporal import TaskConditionCache

    class Visual(torch.nn.Module):
        def forward(self, hidden_states=None, grid_thw=None, **kwargs):
            return hidden_states.mean(dim=1, keepdim=True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = Visual()
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2, vocab_size=8))

    class Processor:
        def image_processor(self, *, images, return_tensors="pt"):
            return {"pixel_values": torch.as_tensor(images).reshape(1, 1, 2),
                    "image_grid_thw": torch.tensor([[1, 1, 1]])}

    adapter = Qwen3VLBackboneAdapter(Model(), processor=Processor())
    adapter.visual_projection = torch.nn.Linear(2, 2, bias=False)
    adapter.visual_projection.weight.data.copy_(torch.eye(2))
    adapter.condition_projection = torch.nn.Linear(4, 2, bias=False)
    adapter.condition_projection.weight.data.zero_()
    task = TaskConditionCache(torch.zeros(2), torch.zeros(2), task_id="t")
    result = adapter.encode_frames([[[1.0, 2.0]]], task)
    assert result.shape == (1, 2)


def test_evaluate_moves_cpu_targets_before_gpu_mask_indexing(monkeypatch):
    """Evaluation must accept the collator's CPU targets with CUDA outputs."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA to reproduce device-mismatch indexing")
    import idv_agent.scripts.train_vla as train_vla

    class Module(torch.nn.Module):
        pass

    adapter, core = Module(), Module()
    device = torch.device("cuda")
    fast = SimpleNamespace(
        move_logits=torch.zeros(1, 2, 9, device=device),
        button_logits=torch.zeros(1, 2, 6, device=device),
    )
    slow = SimpleNamespace(intent_logits=torch.zeros(1, 8, device=device))
    output = SimpleNamespace(fast=fast, slow=slow)
    batch = {
        "fast_loss_mask": torch.tensor([[True, False]]),
        "move_target": torch.tensor([[1, 0]]),
        "button_target": torch.zeros(1, 2, 6),
        "intent_target": torch.tensor([0]),
    }
    monkeypatch.setattr(train_vla, "forward_loss", lambda *args, **kwargs: ({"total": torch.tensor(1.)}, output))
    monkeypatch.setattr(train_vla, "_slow_accuracy", lambda *args, **kwargs: 1.0)

    result = train_vla._evaluate(adapter, core, [batch], device, amp_enabled=False)
    assert result["move_target_counts"][1] == 1
    assert result["button_target_positive_counts"][0] == 0


def test_qwen_batch_splits_unmerged_flattened_visual_tokens():
    """Qwen may return one flattened row per patch (before spatial merge)."""
    from types import SimpleNamespace
    from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter
    from idv_agent.model.temporal import TaskConditionCache

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                text_config=SimpleNamespace(hidden_size=2, vocab_size=8),
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )

        def get_image_features(self, pixel_values, image_grid_thw, **kwargs):
            # Two images, each grid has 4 unmerged patch rows.  The real
            # Qwen3-VL path returns [sum(t*h*w), D] in this case.
            return torch.arange(16, dtype=torch.float32).reshape(8, 2)

    class Processor:
        def image_processor(self, *, images, return_tensors="pt"):
            return {
                "pixel_values": torch.zeros(1, 1, 2),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }

    adapter = Qwen3VLBackboneAdapter(Model(), processor=Processor())
    adapter.visual_projection = torch.nn.Linear(2, 2, bias=False)
    adapter.visual_projection.weight.data.copy_(torch.eye(2))
    adapter.condition_projection = torch.nn.Linear(4, 2, bias=False)
    adapter.condition_projection.weight.data.zero_()
    task = TaskConditionCache(torch.zeros(2), torch.zeros(2), task_id="t")
    result = adapter.encode_frames([[[1.0, 2.0]], [[3.0, 4.0]]], task)
    assert result.shape == (2, 2)
    assert torch.equal(result[:, 0], torch.tensor([3.0, 11.0]))


def test_raw_feature_projection_matches_encode_frames_exactly():
    from types import SimpleNamespace
    from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter
    from idv_agent.model.temporal import TaskConditionCache

    class Visual(torch.nn.Module):
        def forward(self, hidden_states=None, grid_thw=None, **kwargs):
            return hidden_states.mean(dim=1, keepdim=True)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.visual = Visual()
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2, vocab_size=8))

    class Processor:
        def image_processor(self, *, images, return_tensors="pt"):
            return {"pixel_values": torch.as_tensor(images).reshape(1, 1, 2),
                    "image_grid_thw": torch.tensor([[1, 1, 1]])}

    adapter = Qwen3VLBackboneAdapter(Model(), processor=Processor())
    adapter.visual_projection = torch.nn.Linear(2, 2, bias=False)
    adapter.condition_projection = torch.nn.Linear(4, 2, bias=False)
    task = TaskConditionCache(torch.tensor([1., 2.]), torch.tensor([3., 4.]), task_id="t")
    images = [[[1., 2.]], [[3., 4.]]]
    direct = adapter.encode_frames(images, task)
    raw = adapter.encode_raw_frames(images)
    projected = adapter.project_raw_features(raw, task)
    assert torch.equal(direct, projected)


def test_raw_feature_projection_matches_single_frame_exactly():
    from types import SimpleNamespace
    from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter
    from idv_agent.model.temporal import TaskConditionCache
    class Visual(torch.nn.Module):
        def forward(self, hidden_states=None, grid_thw=None, **kwargs):
            return hidden_states.mean(dim=1, keepdim=True)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.visual = Visual()
            self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2, vocab_size=8))
    class Processor:
        def image_processor(self, *, images, return_tensors="pt"):
            return {"pixel_values": torch.as_tensor(images).reshape(1, 1, 2),
                    "image_grid_thw": torch.tensor([[1, 1, 1]])}
    adapter = Qwen3VLBackboneAdapter(Model(), processor=Processor())
    adapter.visual_projection = torch.nn.Linear(2, 2, bias=False)
    adapter.condition_projection = torch.nn.Linear(4, 2, bias=False)
    task = TaskConditionCache(torch.tensor([1., 2.]), torch.tensor([3., 4.]), task_id="t")
    image = [[[1., 2.]]]
    assert torch.equal(adapter.encode_frame(image, task),
                       adapter.project_raw_features(adapter.encode_raw_frames([image]), task)[0])


def test_incremental_raw_cache_adds_only_new_paths(tmp_path):
    from idv_agent.training.raw_feature_cache import RawFeatureCache
    cache = RawFeatureCache(tmp_path / "cache")
    features = torch.tensor([[1., 2.], [3., 4.]])
    assert cache.add_shard("session-a", [tmp_path / "a.png", tmp_path / "b.png"], features) == 2
    cache.save()
    loaded = RawFeatureCache(tmp_path / "cache")
    assert loaded.add_shard("session-b", [tmp_path / "b.png", tmp_path / "c.png"],
                            torch.tensor([[9., 9.], [5., 6.]])) == 1
    loaded.save()
    assert loaded.count == 3
    assert torch.equal(loaded.get(tmp_path / "b.png"), features[1])
    assert torch.equal(loaded.get(tmp_path / "c.png"), torch.tensor([5., 6.]))


def test_raw_cache_reuses_loaded_shard(tmp_path, monkeypatch):
    from idv_agent.training.raw_feature_cache import RawFeatureCache
    cache = RawFeatureCache(tmp_path / "cache")
    path = tmp_path / "a.png"
    cache.add_shard("s", [path], torch.tensor([[1., 2.]])); cache.save()
    loaded = RawFeatureCache(tmp_path / "cache")
    calls = []
    original = torch.load
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: (calls.append(args[0]), original(*args, **kwargs))[1])
    loaded.get(path); loaded.get(path)
    assert len(calls) == 1


def test_encode_batch_uses_raw_cache_and_keeps_projection_gradient(tmp_path):
    from idv_agent.scripts.train_vla import encode_batch
    from idv_agent.training.raw_feature_cache import RawFeatureCache
    class RawAdapter(_Adapter):
        def project_raw_features(self, raw, task_cache):
            return raw * self.scale
    path = tmp_path / "frame.png"
    Image.new("RGB", (1, 1), (9, 0, 0)).save(path)
    cache = RawFeatureCache(tmp_path / "cache")
    cache.add_shard("s", [path], torch.tensor([[4.]])); cache.save()
    adapter = RawAdapter()
    batch = _batch(path, ["same"])
    features = encode_batch(adapter, batch, device=torch.device("cpu"), raw_feature_cache=cache)
    assert adapter.frames == 0
    features.sum().backward()
    assert float(adapter.scale.grad) == 4.0
