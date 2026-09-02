from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path


def _record(index: int, topic: str = "rule") -> dict:
    return {
        "id": f"wk_{index:06d}",
        "question": f"问题 {index} 是什么？",
        "answer": f"这是经过审核的答案 {index}。",
        "mode": "standard",
        "topic": topic,
        "episode_id": f"ep_{index}",
        "verification": {"status": "reviewed"},
    }


def test_wk_dry_run_and_topic_split(tmp_path: Path):
    from idv_agent.scripts.train_wk_lora import train

    path = tmp_path / "wk.jsonl"
    path.write_text("\n".join(json.dumps(_record(i, "rule" if i % 2 else "state")) for i in range(10)),
                    encoding="utf-8")
    args = Namespace(data=str(path), model_path="", output_dir=str(tmp_path / "out"),
                     val_data=None, val_fraction=0.2, max_length=128, batch_size=1,
                     steps=2, lr=2e-4, device="cpu", seed=7,
                     gradient_checkpointing=False, dry_run=True)
    result = train(args)
    assert result["stage"] == "M1_WK"
    assert result["train_records"] + result["val_records"] == 10
    assert result["val_records"] >= 2
    assert result["preview"][0][0].startswith("<mode:standard>")


def test_wk_rejects_unreviewed(tmp_path: Path):
    from idv_agent.scripts.train_wk_lora import read_jsonl

    path = tmp_path / "wk.jsonl"
    row = _record(1)
    row["verification"] = {"status": "needs_review"}
    path.write_text(json.dumps(row), encoding="utf-8")
    try:
        read_jsonl(path)
    except ValueError as exc:
        assert "未审核" in str(exc)
    else:
        raise AssertionError("unreviewed WK sample must be rejected")
