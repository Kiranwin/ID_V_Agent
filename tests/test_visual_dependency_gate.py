from pathlib import Path

import pytest
import torch
from PIL import Image


def test_image_transform_zero_and_blur_are_explicit():
    from idv_agent.scripts.evaluate_visual_dependency import _image_transform

    image = Image.new("RGB", (4, 3), (100, 120, 140))
    zero = _image_transform("image_zero")(image)
    blur = _image_transform("image_blur")(image)

    assert zero.getpixel((0, 0)) == (0, 0, 0)
    assert blur.size == image.size
    assert _image_transform("normal") is None
    with pytest.raises(ValueError, match="condition"):
        _image_transform("unknown")


def test_image_shuffle_uses_other_sample_and_is_deterministic():
    from idv_agent.scripts.evaluate_visual_dependency import _shuffle_frame_paths

    paths = [
        [Path("a/0.jpg"), Path("a/1.jpg")],
        [Path("b/0.jpg"), Path("b/1.jpg")],
        [Path("c/0.jpg"), None],
    ]
    shuffled = _shuffle_frame_paths(paths)

    assert shuffled == [paths[1], paths[2], paths[0]]
    assert all(left != right for left, right in zip(paths, shuffled))


def test_history_zero_clears_only_history_tensor():
    from idv_agent.scripts.evaluate_visual_dependency import _zero_history

    batch = {
        "history_actions": torch.ones(2, 72),
        "move_target": torch.ones(2, 4, dtype=torch.long),
    }
    zeroed = _zero_history(batch)

    assert torch.equal(zeroed["history_actions"], torch.zeros(2, 72))
    assert torch.equal(zeroed["move_target"], batch["move_target"])
    assert torch.equal(batch["history_actions"], torch.ones(2, 72))


def test_dependency_gate_requires_each_image_metric_to_drop_by_fixed_threshold():
    from idv_agent.scripts.evaluate_visual_dependency import _dependency_gate

    normal = {"move_accuracy": 0.80, "camera_accuracy": 0.40, "intent_accuracy": 0.90}
    degraded = {"move_accuracy": 0.67, "camera_accuracy": 0.34, "intent_accuracy": 0.76}
    result = _dependency_gate(normal, degraded)

    assert result["pass"] is True
    assert result["metrics"]["move_accuracy"]["threshold"] == pytest.approx(0.12)
    assert result["metrics"]["camera_accuracy"]["threshold"] == pytest.approx(0.06)

    failed = _dependency_gate(normal, {**degraded, "intent_accuracy": 0.80})
    assert failed["pass"] is False
    assert "intent_accuracy" in failed["failures"]


def test_dependency_gate_fails_closed_for_missing_metric():
    from idv_agent.scripts.evaluate_visual_dependency import _dependency_gate

    result = _dependency_gate(
        {"move_accuracy": 0.8, "camera_accuracy": 0.8},
        {"move_accuracy": 0.1, "camera_accuracy": 0.1},
    )

    assert result["pass"] is False
    assert "intent_accuracy" in result["failures"]


def test_cli_returns_nonzero_when_any_checkpoint_fails(tmp_path, monkeypatch):
    from idv_agent.scripts import evaluate_visual_dependency as evaluator

    def fake_evaluate(checkpoint, args):
        passed = Path(checkpoint).name == "good.pt"
        return {"checkpoint": str(checkpoint), "visual_dependency_gate_pass": passed}

    monkeypatch.setattr(evaluator, "evaluate_checkpoint", fake_evaluate)
    output = tmp_path / "gate.json"
    exit_code = evaluator.main([
        "--data", "data.jsonl", "--model-path", "model", "--init-checkpoint", "m2",
        "--checkpoint", str(tmp_path / "good.pt"),
        "--checkpoint", str(tmp_path / "bad.pt"),
        "--output", str(output),
    ])

    report = __import__("json").loads(output.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["visual_dependency_gate_pass"] is False
    assert len(report["checkpoints"]) == 2
