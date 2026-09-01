"""Validate and clean the text-only World Knowledge (WK) JSONL dataset.

WK is deliberately kept separate from VG image annotations and ACT trajectory
data.  Only records with ``verification.status == 'verified'`` are emitted to
the train/val/test files; ``needs_review`` records remain in the audit file.

Example:
    python -m idv_agent.scripts.clean_wk_dataset
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


MODES = {"standard", "joint_hunt", "blackjack"}
TOPICS = {"interaction", "object", "state", "rule", "safety"}
SOURCE_TYPES = {"official", "manual", "verified_community", "project_record"}
STATUSES = {"verified", "needs_review", "rejected"}
SPLITS = {"train", "val", "test", "candidate"}
REQUIRED = {"id", "question", "answer", "mode", "topic", "source", "verification", "episode_id", "split"}
VAGUE_ANSWERS = {"不知道", "不清楚", "视情况而定", "按实际情况", "待定"}


def _normalize(value: str) -> str:
    return re.sub(r"\s+", "", value).lower()


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any] | None, str | None]]:
    rows: list[tuple[int, dict[str, Any] | None, str | None]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append((line_no, None, f"invalid_json: {exc.msg}"))
                continue
            rows.append((line_no, item, None))
    return rows


def _validate_shape(record: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED - set(record))
    if missing:
        errors.append(f"missing_fields:{','.join(missing)}")
        return errors
    for field in ("id", "question", "answer", "episode_id", "split"):
        if not isinstance(record[field], str) or not record[field].strip():
            errors.append(f"empty_{field}")
    if record.get("mode") not in MODES:
        errors.append("invalid_mode")
    if record.get("topic") not in TOPICS:
        errors.append("invalid_topic")
    if record.get("split") not in SPLITS:
        errors.append("invalid_split")

    source = record.get("source")
    if not isinstance(source, dict):
        errors.append("invalid_source")
    else:
        for field in ("title", "url_or_path", "source_type", "quote_or_excerpt"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                errors.append(f"invalid_source_{field}")
        if source.get("source_type") not in SOURCE_TYPES:
            errors.append("invalid_source_type")
        source_ref = source.get("url_or_path")
        if source.get("source_type") == "project_record" and isinstance(source_ref, str):
            if not (repo_root / source_ref).is_file():
                errors.append("missing_source_path")

    verification = record.get("verification")
    if not isinstance(verification, dict):
        errors.append("invalid_verification")
    else:
        if verification.get("status") not in STATUSES:
            errors.append("invalid_verification_status")
        if not isinstance(verification.get("review_note"), str) or not verification["review_note"].strip():
            errors.append("invalid_review_note")

    answer = record.get("answer")
    if isinstance(answer, str):
        if len(answer.strip()) < 8:
            errors.append("ambiguous_answer")
        if _normalize(answer) in {_normalize(item) for item in VAGUE_ANSWERS}:
            errors.append("ambiguous_answer")
    return errors


def clean(input_path: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    parsed = _read_jsonl(input_path)
    reason_counts: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    question_answers: dict[str, str] = {}

    for line_no, record, parse_error in parsed:
        if parse_error:
            reason_counts["invalid_json"] += 1
            rejected.append({"line": line_no, "reason": parse_error})
            continue
        assert record is not None
        errors = _validate_shape(record, repo_root)
        if record.get("verification", {}).get("status") == "rejected":
            errors.append("status_rejected")
        if record.get("id") in seen_ids:
            errors.append("duplicate_id")
        if isinstance(record.get("id"), str):
            seen_ids.add(record["id"])
        question = record.get("question")
        answer = record.get("answer")
        if isinstance(question, str) and isinstance(answer, str) and question.strip():
            key = _normalize(question)
            previous = question_answers.get(key)
            if previous is not None:
                errors.append("duplicate_question" if previous == _normalize(answer) else "conflicting_answer")
            else:
                question_answers[key] = _normalize(answer)
        if errors:
            for error in sorted(set(errors)):
                reason_counts[error] += 1
            rejected.append({"line": line_no, "id": record.get("id"), "reason": sorted(set(errors)), "record": record})
        else:
            candidates.append(record)

    # Near-duplicate questions are rejected after basic validation.  Comparing
    # normalized text keeps cosmetic punctuation from defeating the audit.
    accepted: list[dict[str, Any]] = []
    accepted_questions: list[str] = []
    for record in candidates:
        question = _normalize(record["question"])
        duplicate_of = next(
            (accepted[index]["id"] for index, other in enumerate(accepted_questions)
             if SequenceMatcher(None, question, other).ratio() >= 0.94),
            None,
        )
        if duplicate_of is not None:
            reason_counts["near_duplicate_question"] += 1
            rejected.append({"id": record["id"], "reason": ["near_duplicate_question"], "duplicate_of": duplicate_of, "record": record})
            continue
        accepted.append(record)
        accepted_questions.append(question)

    # A source/episode is indivisible: no paraphrases from the same source
    # bundle may cross train/val/test. Candidate records are not training data.
    episode_splits: dict[str, set[str]] = defaultdict(set)
    for record in accepted:
        if record["verification"]["status"] == "verified":
            episode_splits[record["episode_id"]].add(record["split"])
    leaking_episodes = {episode for episode, splits in episode_splits.items() if len(splits) > 1}
    if leaking_episodes:
        kept: list[dict[str, Any]] = []
        for record in accepted:
            if record["episode_id"] in leaking_episodes and record["verification"]["status"] == "verified":
                reason_counts["episode_split_leakage"] += 1
                rejected.append({"id": record["id"], "reason": ["episode_split_leakage"], "record": record})
            else:
                kept.append(record)
        accepted = kept

    output_dir.mkdir(parents=True, exist_ok=True)
    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    write_jsonl(output_dir / "wk_cleaned.jsonl", accepted)
    write_jsonl(output_dir / "wk_rejected.jsonl", rejected)
    training = [item for item in accepted if item["verification"]["status"] == "verified"]
    for split in ("train", "val", "test"):
        write_jsonl(output_dir / f"wk_{split}.jsonl", [item for item in training if item["split"] == split])
    write_jsonl(output_dir / "wk_needs_review.jsonl", [item for item in accepted if item["verification"]["status"] == "needs_review"])

    summary: dict[str, Any] = {
        "input": str(input_path).replace("\\", "/"),
        "pre_clean_records": len(parsed),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "training_eligible_records": len(training),
        "needs_review_records": sum(item["verification"]["status"] == "needs_review" for item in accepted),
        "counts_by_split": dict(sorted(Counter(item["split"] for item in training).items())),
        "counts_by_mode": dict(sorted(Counter(item["mode"] for item in accepted).items())),
        "counts_by_topic": dict(sorted(Counter(item["topic"] for item in accepted).items())),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "episode_split_leakage": sorted(leaking_episodes),
    }
    (output_dir / "cleaning_report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="校验和清洗 WK JSONL 数据集")
    parser.add_argument("--input", type=Path, default=Path("data/wk_vg/wk/raw_candidates.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/wk_vg/wk"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    summary = clean(args.input, args.output_dir, args.repo_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
