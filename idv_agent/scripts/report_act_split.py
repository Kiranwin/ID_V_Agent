"""Build and validate a session-level ACT train/validation split report."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from idv_agent.scripts.train_vla import _dataset_paths
from idv_agent.training.vla_dataset import VLASequenceDataset
from idv_agent.vla.action_chunk import INTENTS


def _counts(paths: list[str]) -> tuple[int, Counter[str], list[str]]:
    dataset = VLASequenceDataset(paths, verify_images=False)
    counts: Counter[str] = Counter()
    sessions: set[str] = set()
    for index in range(len(dataset)):
        sample = dataset[index]
        intent_id = int(sample["intent_target"])
        if intent_id >= 0:
            counts[INTENTS[intent_id]] += 1
        sessions.add(str(sample["episode_id"]))
    return len(dataset), counts, sorted(sessions)


def report(data: str, val_data: str, output: Path) -> dict:
    train_paths = _dataset_paths(data)
    val_paths = _dataset_paths(val_data)
    if not train_paths or not val_paths:
        raise ValueError("train/val 至少各需要一个 vla_chunks_v4.jsonl")
    train_n, train_counts, train_sessions = _counts(train_paths)
    val_n, val_counts, val_sessions = _counts(val_paths)
    observed = sorted(set(train_counts) | set(val_counts), key=INTENTS.index)
    missing_train = [name for name in observed if not train_counts[name]]
    missing_val = [name for name in observed if not val_counts[name]]
    overlap = sorted(set(train_sessions) & set(val_sessions))
    result = {
        "schema": "act.session_split.v1",
        "train": {"sessions": train_sessions, "chunks": train_n,
                  "intent_counts": dict(train_counts),
                  "intent_ratios": {k: round(v / train_n, 6) for k, v in train_counts.items()}},
        "val": {"sessions": val_sessions, "chunks": val_n,
                "intent_counts": dict(val_counts),
                "intent_ratios": {k: round(v / val_n, 6) for k, v in val_counts.items()}},
        "observed_intents": observed,
        "missing_in_train": missing_train,
        "missing_in_val": missing_val,
        "session_overlap": overlap,
        "gate_pass": not missing_train and not missing_val and not overlap,
        "note": "仅对数据中实际出现的 intent 做双侧门禁；未在任何 session 出现的词汇不伪造样本。",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["gate_pass"]:
        raise SystemExit(2)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/act_split_intent_distribution.json"))
    args = parser.parse_args(argv)
    report(args.data, args.val_data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
