# -*- coding: utf-8 -*-
"""Validate the WK v2 training JSONL and knowledge gaps.

Checks:
    - JSONL 可解析、必填字段齐全
    - 条目总数 50-100，五类 topic 每类 >= 8
    - id / question 唯一，近重复问题检测
    - answer 不含「候选/待核验/不得进入训练/仅作为清单」等元描述
    - mode / topic / source_type / verification.status 枚举合法
    - source 的 title / url_or_path 非空；project_record 路径存在
    - knowledge_gaps.json 结构合法

Usage:
    python -m idv_agent.scripts.validate_wk_v2
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "idv_agent" / "data" / "wk"
TRAIN_PATH = DATA_DIR / "wk_train.jsonl"
GAPS_PATH = DATA_DIR / "knowledge_gaps.json"

MODES = {"standard", "joint_hunt", "blackjack"}
TOPICS = {"rule", "object", "state", "interaction", "mode"}
SOURCE_TYPES = {"official", "verified_community", "project_record"}
STATUSES = {"reviewed", "needs_review"}
REQUIRED = {"id", "question", "answer", "mode", "topic", "episode_id", "split", "source", "verification"}
SOURCE_REQUIRED = {"title", "url_or_path", "source_type", "quote_or_excerpt"}
BANNED_META = ("候选", "待核验", "不得进入训练", "仅作为清单")
MIN_PER_TOPIC = 8


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def main() -> int:
    errors: list[str] = []
    if not TRAIN_PATH.is_file():
        print(f"ERROR: missing {TRAIN_PATH}")
        return 1

    records: list[dict] = []
    for line_no, line in enumerate(TRAIN_PATH.open("r", encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid_json: {exc.msg}")
            continue
        if not isinstance(item, dict):
            errors.append(f"line {line_no}: not an object")
            continue
        records.append(item)

    total = len(records)
    if not 50 <= total <= 100:
        errors.append(f"total records {total} out of [50, 100]")

    ids: list[str] = []
    questions: list[str] = []
    topic_counter: Counter = Counter()
    mode_counter: Counter = Counter()
    status_counter: Counter = Counter()
    seen_questions: dict[str, int] = {}

    for item in records:
        record_id = item.get("id", "<missing>")
        ids.append(record_id)
        missing = sorted(REQUIRED - set(item))
        if missing:
            errors.append(f"{record_id}: missing_fields:{','.join(missing)}")

        for field in ("id", "question", "answer", "episode_id", "split"):
            if not isinstance(item.get(field), str) or not item[field].strip():
                errors.append(f"{record_id}: empty_{field}")

        mode = item.get("mode")
        if mode not in MODES:
            errors.append(f"{record_id}: invalid_mode:{mode}")
        else:
            mode_counter[mode] += 1

        topic = item.get("topic")
        if topic not in TOPICS:
            errors.append(f"{record_id}: invalid_topic:{topic}")
        else:
            topic_counter[topic] += 1

        split = item.get("split")
        if split != "train":
            errors.append(f"{record_id}: invalid_split:{split}")

        answer = item.get("answer", "")
        for banned in BANNED_META:
            if banned in answer:
                errors.append(f"{record_id}: answer_contains_meta_word:{banned}")

        question = item.get("question", "")
        questions.append(question)
        norm_q = _normalize(question)
        if norm_q:
            if norm_q in seen_questions:
                errors.append(f"{record_id}: duplicate_question_with:{seen_questions[norm_q]}")
            else:
                seen_questions[norm_q] = record_id

        source = item.get("source")
        if not isinstance(source, dict):
            errors.append(f"{record_id}: invalid_source")
        else:
            for field in SOURCE_REQUIRED:
                if not isinstance(source.get(field), str) or not source[field].strip():
                    errors.append(f"{record_id}: invalid_source_{field}")
            st = source.get("source_type")
            if st not in SOURCE_TYPES:
                errors.append(f"{record_id}: invalid_source_type:{st}")
            if st == "project_record":
                ref = source.get("url_or_path", "")
                if not (REPO_ROOT / ref).is_file():
                    errors.append(f"{record_id}: missing_source_path:{ref}")

        verification = item.get("verification")
        if not isinstance(verification, dict):
            errors.append(f"{record_id}: invalid_verification")
        else:
            status = verification.get("status")
            if status not in STATUSES:
                errors.append(f"{record_id}: invalid_status:{status}")
            else:
                status_counter[status] += 1
            if not isinstance(verification.get("review_note", ""), str):
                errors.append(f"{record_id}: invalid_review_note")

    if len(ids) != len(set(ids)):
        dup_ids = {x for x in ids if ids.count(x) > 1}
        errors.append(f"duplicated_ids:{sorted(dup_ids)}")

    for topic in TOPICS:
        if topic_counter[topic] < MIN_PER_TOPIC:
            errors.append(f"topic {topic} has {topic_counter[topic]} < {MIN_PER_TOPIC}")

    # 近重复问题（归一化后相似度 >= 0.94）
    normed = [_normalize(q) for q in questions if q]
    for i in range(len(normed)):
        for j in range(i + 1, len(normed)):
            if not normed[i] or not normed[j]:
                continue
            ratio = SequenceMatcher(None, normed[i], normed[j]).ratio()
            if ratio >= 0.94:
                errors.append(
                    f"near_duplicate_questions:{ids[i]}~{ids[j]}:{ratio:.2f}"
                )

    # knowledge_gaps 结构
    if not GAPS_PATH.is_file():
        errors.append(f"missing {GAPS_PATH}")
    else:
        gaps = json.loads(GAPS_PATH.open("r", encoding="utf-8").read())
        if not isinstance(gaps, dict) or not isinstance(gaps.get("gaps"), list):
            errors.append("invalid knowledge_gaps.json structure")
        else:
            gap_ids = [g.get("id") for g in gaps["gaps"]]
            if len(gap_ids) != len(set(gap_ids)):
                errors.append("duplicated gap ids")

    print("===== WK v2 validation =====")
    print(f"total records: {total}")
    print(f"topic distribution: {dict(sorted(topic_counter.items()))}")
    print(f"mode distribution: {dict(sorted(mode_counter.items()))}")
    print(f"verification distribution: {dict(sorted(status_counter.items()))}")
    print(f"errors: {len(errors)}")
    for err in errors[:80]:
        print(f"  - {err}")

    if errors:
        print("RESULT: FAIL")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
