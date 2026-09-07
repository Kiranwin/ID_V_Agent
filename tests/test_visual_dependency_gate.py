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
    shuffled = _shuffle_frame_paths(paths, episode_ids=["a", "b", "c"])

    assert shuffled == [paths[1], paths[2], paths[0]]
    assert all(left != right for left, right in zip(paths, shuffled))


def test_image_shuffle_never_uses_another_window_from_the_same_session():
    from idv_agent.scripts.evaluate_visual_dependency import _shuffle_frame_paths

    paths = [[Path(f"a/{index}.jpg")] for index in range(3)] + [[Path(f"b/{index}.jpg")] for index in range(2)]
    episodes = ["a", "a", "a", "b", "b"]
    shuffled = _shuffle_frame_paths(paths, episode_ids=episodes)

    assert [path[0].parts[0] for path in shuffled] == ["b", "b", "b", "a", "a"]
    with pytest.raises(ValueError, match="不同 session"):
        _shuffle_frame_paths(paths, episode_ids=["a"] * len(paths))


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


def test_camera_prior_override_is_applied_after_checkpoint_restore():
    from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
    from idv_agent.scripts.evaluate_visual_dependency import _override_camera_prior_scale

    core = SharedFastSlowVLA(frame_feature_dim=4, temporal_dim=4, history_action_dim=72)
    core.camera_prior_scale = 0.5
    _override_camera_prior_scale(core, 0.0)
    assert core.camera_prior_scale == 0.0
    with pytest.raises(ValueError, match="只能为 0"):
        _override_camera_prior_scale(core, 0.5)


def test_episode_diverse_subset_uses_distinct_episode_representatives_first():
    from idv_agent.scripts.train_vla import _episode_diverse_subset

    class TinyDataset:
        rows = [
            {"episode_id": "a", "anchor_frame": 21},
            {"episode_id": "a", "anchor_frame": 33},
            {"episode_id": "a", "anchor_frame": 45},
            {"episode_id": "b", "anchor_frame": 21},
            {"episode_id": "b", "anchor_frame": 33},
            {"episode_id": "b", "anchor_frame": 45},
            {"episode_id": "c", "anchor_frame": 21},
            {"episode_id": "c", "anchor_frame": 33},
            {"episode_id": "c", "anchor_frame": 45},
        ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    selected = _episode_diverse_subset(TinyDataset(), 3)

    assert [selected[index]["episode_id"] for index in range(len(selected))] == ["a", "b", "c"]
    assert [selected[index]["anchor_frame"] for index in range(len(selected))] == [33, 33, 33]


def test_evaluation_stratified_subset_keeps_rare_intent_and_camera_targets():
    from idv_agent.scripts.train_vla import _evaluation_stratified_subset

    def row(episode, anchor, intent, dx, dy):
        return {
            "episode_id": episode,
            "anchor_frame": anchor,
            "intent_target": torch.tensor(intent),
            "move_target": torch.tensor([1, 1, 1, 1]),
            "camera_dx_target": torch.tensor([dx, dx, dx, dx]),
            "camera_dy_target": torch.tensor([dy, dy, dy, dy]),
        }

    class TinyDataset:
        rows = [
            *[row(f"travel_{i}", i, 4, 2, 2) for i in range(8)],
            row("decipher", 99, 0, 0, 4),
            row("turn_left", 100, 4, 1, 1),
        ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    selected = _evaluation_stratified_subset(TinyDataset(), 4)
    rows = [selected[index] for index in range(len(selected))]

    assert {int(row["intent_target"]) for row in rows} == {0, 4}
    assert {int(value) for row in rows for value in row["camera_dx_target"]} >= {0, 1, 2}
    assert {int(value) for row in rows for value in row["camera_dy_target"]} >= {1, 2, 4}


def test_evaluation_stratified_subset_reserves_four_examples_per_available_intent():
    from idv_agent.scripts.train_vla import _evaluation_stratified_subset

    class TinyDataset:
        rows = []
        for index in range(12):
            intent = 0 if index < 4 else 4
            rows.append({
                "episode_id": f"episode_{index}", "anchor_frame": index,
                "intent_target": torch.tensor(intent),
                "move_target": torch.tensor([1 if intent == 0 else index, 1, 1, 1]),
                "camera_dx_target": torch.tensor([2 if intent == 0 else index % 5, 2, 2, 2]),
                "camera_dy_target": torch.tensor([2 if intent == 0 else (index + 1) % 5, 2, 2, 2]),
            })

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    selected = _evaluation_stratified_subset(TinyDataset(), 8)
    counts = {intent: 0 for intent in (0, 4)}
    for index in range(len(selected)):
        counts[int(selected[index]["intent_target"])] += 1

    assert counts == {0: 4, 4: 4}


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


def test_output_change_summary_reports_logit_shift_and_argmax_flip_per_head():
    from idv_agent.scripts.evaluate_visual_dependency import _output_change_summary

    normal = {
        "move": torch.tensor([[[3.0, 1.0], [0.0, 2.0]]]),
        "camera_dx": torch.tensor([[[1.0, 0.0]]]),
        "camera_dy": torch.tensor([[[0.0, 1.0]]]),
        "intent": torch.tensor([[2.0, 0.0]]),
    }
    changed = {
        "move": torch.tensor([[[1.0, 3.0], [0.0, 2.0]]]),
        "camera_dx": torch.tensor([[[0.0, 1.0]]]),
        "camera_dy": torch.tensor([[[0.1, 0.9]]]),
        "intent": torch.tensor([[0.0, 2.0]]),
    }
    result = _output_change_summary(normal, changed)

    assert result["move"]["argmax_flip_rate"] == pytest.approx(0.5)
    assert result["camera_dx"]["argmax_flip_rate"] == pytest.approx(1.0)
    assert result["camera_dy"]["argmax_flip_rate"] == pytest.approx(0.0)
    assert result["intent"]["argmax_flip_rate"] == pytest.approx(1.0)
    assert result["move"]["mean_abs_logit_delta"] > 0


def test_macro_recall_reveals_majority_class_collapse_hidden_by_accuracy():
    from idv_agent.scripts.evaluate_visual_dependency import _macro_recall

    # Same number of correct predictions can hide a complete rare-class loss.
    confusion_normal = [[8, 2], [1, 9]]
    confusion_black = [[10, 0], [10, 0]]
    assert _macro_recall(confusion_normal) == pytest.approx(0.85)
    assert _macro_recall(confusion_black) == pytest.approx(0.5)


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
