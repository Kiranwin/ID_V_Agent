from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_infer_states_and_generate_frame_qa(tmp_path):
    from idv_agent.scripts.generate_frame_qa import generate
    from idv_agent.scripts.infer_frame_states import infer

    session = tmp_path / "data/vla_raw_sessions/s1"
    frame_dir = session / "frames"
    frame_dir.mkdir(parents=True)
    dataset = tmp_path / "data/vg"
    image_dir = dataset / "images/train"
    label_dir = dataset / "labels/train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    manifest_rows = []
    for frame in range(3):
        source = frame_dir / f"{frame:08d}.jpg"
        image = image_dir / f"s1_{frame:08d}.jpg"
        Image.new("RGB", (100, 50), "black").save(source)
        Image.new("RGB", (100, 50), "black").save(image)
        label = label_dir / f"s1_{frame:08d}.txt"
        label.write_text("2 0.5 0.5 0.1 0.1\n" if frame == 2 else "", encoding="utf-8")
        manifest_rows.append({"image": image.relative_to(tmp_path).as_posix(),
                              "label": label.relative_to(tmp_path).as_posix(),
                              "session": "s1", "source": source.relative_to(tmp_path).as_posix(),
                              "split": "train"})
    _write_csv(dataset / "manifest.csv", ("image", "label", "session", "source", "split"), manifest_rows)
    _write_csv(session / "frame_timestamps.csv", ("frame_id", "timestamp_ns"), [
        {"frame_id": 0, "timestamp_ns": 100},
        {"frame_id": 1, "timestamp_ns": 200},
        {"frame_id": 2, "timestamp_ns": 300},
    ])
    _write_csv(session / "events.csv", ("timestamp_ns", "kind", "code", "value"), [
        {"timestamp_ns": 150, "kind": "key_down", "code": "key:w", "value": 1},
        {"timestamp_ns": 250, "kind": "key_up", "code": "key:w", "value": 1},
        {"timestamp_ns": 260, "kind": "key_down", "code": "key:q", "value": 1},
        {"timestamp_ns": 350, "kind": "key_up", "code": "key:q", "value": 1},
    ])
    _write_csv(session / "mouse_positions.csv", ("timestamp_ns", "x", "y"), [
        {"timestamp_ns": 100, "x": 10, "y": 10},
        {"timestamp_ns": 300, "x": 10, "y": 10},
    ])
    states_path = dataset / "frame_states.jsonl"
    summary = infer(dataset, repo_root=tmp_path, output=states_path)
    assert summary["states"] == {"decoding": 1, "idle": 1, "walking": 1, "chased": 0}

    state_rows = [json.loads(line) for line in states_path.read_text(encoding="utf-8").splitlines()]
    assert [row["state"] for row in state_rows] == ["idle", "walking", "decoding"]
    annotations = dataset / "vg_annotations.jsonl"
    annotations.write_text("".join(json.dumps({
        "id": f"s1_{frame:08d}", "image": manifest_rows[frame]["image"],
        "source_session": "s1", "source_frame": frame, "split": "train",
        "intent_hint": "travel", "objects": ([{
            "label": "cipher_machine", "source_class": "cipher_highlight",
            "visibility": "highlighted", "bbox_xyxy_norm": [0.4, 0.2, 0.6, 0.8],
        }] if frame == 2 else []),
    }, ensure_ascii=False) + "\n" for frame in range(3)), encoding="utf-8")
    qa_path = dataset / "frame_qa.jsonl"
    count = generate(dataset, annotations=annotations, states=states_path,
                     output=qa_path, intent="search")
    assert count == 12
    qa = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines()]
    assert {row["task"] for row in qa} == {"state_qa", "presence_qa", "attribute_qa", "intent_qa"}
    assert all(row["intent"] == "search" for row in qa)


def test_intent_segment_template_and_validation(tmp_path):
    import csv
    from idv_agent.scripts.init_intent_segments import init
    from idv_agent.scripts.validate_intent_segments import validate

    session = tmp_path / "episode"
    (session / "frames").mkdir(parents=True)
    for i in range(30):
        (session / "frames" / f"{i:08d}.jpg").write_bytes(b"jpeg")
    with (session / "per_frame_actions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame_idx", "category_name", "text_action"))
        writer.writeheader()
        for i in range(30):
            writer.writerow({"frame_idx": i, "category_name": "MOVE" if i < 15 else "INTERACT_HOLD",
                             "text_action": ""})
    output = session / "intent_segments.csv"
    assert init(session, output, min_segment_frames=3) == 2
    rows = list(csv.DictReader(output.open(encoding="utf-8-sig", newline="")))
    rows[0]["intent"] = "search"
    rows[1]["intent"] = "decipher"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    assert validate(output, frame_start=0, frame_end=29, require_full_coverage=True) == 2
