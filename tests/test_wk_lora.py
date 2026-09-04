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


def test_wk_no_validation_reports_all_train_and_semantic_eval(tmp_path: Path):
    from idv_agent.scripts.train_wk_lora import train

    path = tmp_path / "wk.jsonl"
    path.write_text("\n".join(json.dumps(_record(i)) for i in range(10)), encoding="utf-8")
    args = Namespace(data=str(path), model_path="", output_dir=str(tmp_path / "out"),
                     val_data=None, val_fraction=0.0, max_length=128, batch_size=1,
                     steps=2, lr=2e-4, device="cpu", seed=7,
                     gradient_checkpointing=False, dry_run=True, semantic_variants=2)
    result = train(args)
    assert result["train_records"] == 10
    assert result["val_records"] == 0
    assert result["semantic_eval_records"] == 20


def test_semantic_variants_are_deterministic_and_preserve_target():
    from idv_agent.scripts.validate_wk_inference import build_semantic_eval_records

    row = _record(3)
    variants = build_semantic_eval_records([row], variants_per_record=2)
    assert [item["id"] for item in variants] == ["wk_000003::variant_1", "wk_000003::variant_2"]
    assert all(item["answer"] == row["answer"] for item in variants)
    assert all(item["mode"] == row["mode"] for item in variants)
    assert all(item["source_id"] == row["id"] for item in variants)
    assert len({item["question"] for item in variants}) == 2


def test_semantic_gate_requires_eos_low_repetition_and_mode_coverage():
    from idv_agent.scripts.validate_wk_inference import summarize_gate

    rows = [
        {"mode": "standard", "char_f1": 0.8, "ended_with_eos": True, "repeat_rate": 0.0},
        {"mode": "joint_hunt", "char_f1": 0.8, "ended_with_eos": True, "repeat_rate": 0.0},
        {"mode": "blackjack", "char_f1": 0.8, "ended_with_eos": True, "repeat_rate": 0.0},
    ]
    assert summarize_gate(rows)["pass"] is True
    rows[0]["ended_with_eos"] = False
    assert summarize_gate(rows)["pass"] is False


def test_wk_dataset_supervises_eos_token():
    from idv_agent.scripts.train_wk_lora import WKTextDataset

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 99
        eos_token = "<eos>"

        def __call__(self, text, **kwargs):
            ids = [10 + (ord(ch) % 80) for ch in text]
            return {"input_ids": ids}

    item = WKTextDataset([_record(1)], Tokenizer(), 128)[0]
    assert int(item["input_ids"][-1]) == 99
    assert int(item["labels"][-1]) == 99


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
