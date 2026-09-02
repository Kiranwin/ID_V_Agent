from pathlib import Path

import torch
from PIL import Image


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
