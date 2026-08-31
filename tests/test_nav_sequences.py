"""局部预测导航样本构造测试。"""

import csv
import json
from pathlib import Path

from idv_agent.scripts.build_nav_sequences import MACRO_ACTIONS, _macro, build


def test_macro_action_keeps_camera_as_auxiliary():
    rows = [
        {"frame_idx": str(i), "move_x": "0", "move_y": "1",
         "cam_dx": "0.2", "cam_dy": "-0.1"}
        for i in range(6)
    ]
    out = _macro(rows)
    assert out["action"] == "forward"
    assert out["move_y"] == 1.0
    assert out["cam_dx"] == 0.2
    assert out["cam_dy"] == -0.1
    assert out["source_frames"] == list(range(6))
    assert MACRO_ACTIONS == ("forward", "left", "right", "stop_observe")


def test_build_emits_four_macro_steps(tmp_path: Path):
    session = tmp_path / "session_x"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(60):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpeg")
    with (session / "per_frame_actions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_id", "move_x", "move_y", "cam_dx", "cam_dy"])
        writer.writeheader()
        for i in range(60):
            writer.writerow({"frame_id": i, "move_x": 0, "move_y": 1,
                             "cam_dx": 0.1, "cam_dy": 0})

    output = tmp_path / "nav.jsonl"
    count = build(session, output, history=3, stride=3, horizon=4, macro_frames=6)
    assert count == 10  # anchors 6..33, with a four-macro (24-frame) future
    sample = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert len(sample["history_frames"]) == 3
    assert len(sample["future_actions"]) == 4
    assert all(step["action"] == "forward" for step in sample["future_actions"])
    assert all(step["cam_dx"] == 0.1 for step in sample["future_actions"])
    assert sample["future_actions"][0]["source_frames"] == [7, 8, 9, 10, 11, 12]


def test_macro_tolerates_blank_csv_values():
    out = _macro([{"frame_idx": 1, "move_x": "", "move_y": None,
                   "cam_dx": "bad", "cam_dy": ""}])
    assert out["action"] == "stop_observe"
    assert out["move_x"] == out["move_y"] == 0.0
    assert out["cam_dx"] == out["cam_dy"] == 0.0
