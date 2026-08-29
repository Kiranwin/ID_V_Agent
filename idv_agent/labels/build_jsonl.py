"""把 session 目录（per_frame_actions.csv + frames + 启发式意图）合并成 BC 训练 JSONL。

格式见 docs/03。副产物 intents.csv（每帧 intent_id/name 便于校对）。

依赖：先跑 extract（生成 per_frame_actions.csv）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from idv_agent.configs.intent import IntentCategory, intent_to_vector
from idv_agent.configs.schema import ExtractParams, NUM_CONTINUOUS
from idv_agent.labels.intent import (
    build_action_skeleton,
    build_full_action,
    build_numeric_action,
    infer_intents_for_frames,
)


HUMAN_PROMPT = "请分析当前游戏画面，输出战术意图与求生者的鼠键操作。"


@dataclass
class BuildJsonlResult:
    jsonl_path: Path
    intents_path: Path
    num_frames: int
    intent_counts: dict[int, int]
    category_counts: dict[int, int]


def _read_meta_fps(session_dir: Path) -> float:
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        eff = meta.get("effective_fps")
        if eff and eff > 0:
            return float(eff)
        target = meta.get("target_fps")
        if target and target > 0:
            return float(target)
    return 30.0


def build_session_jsonl(
    session_dir: Path,
    out_jsonl: Optional[Path] = None,
    out_intents: Optional[Path] = None,
    params: Optional[ExtractParams] = None,
) -> BuildJsonlResult:
    """单个 session 的 JSONL。per_frame_actions.csv 必须已存在。"""
    actions = pd.read_csv(session_dir / "per_frame_actions.csv")
    fps = _read_meta_fps(session_dir)

    # 意图推断（平滑）
    intents, vectors = infer_intents_for_frames(actions, fps)

    session_id = session_dir.name
    out_jsonl = out_jsonl or session_dir / "samples.jsonl"
    out_intents = out_intents or session_dir / "intents.csv"

    rows = []
    intent_rows = []
    for i, (_, row) in enumerate(actions.iterrows()):
        cat_id = int(row["category_id"])
        cont = build_numeric_action([row["move_x"], row["move_y"], row["cam_dx"], row["cam_dy"]])
        intent_id = int(intents[i])
        intent_name = IntentCategory(intent_id).name.lower()
        skel = build_action_skeleton(cat_id, cont)
        full_text = build_full_action(cat_id, cont, row.get("held_keys") or "")
        gpt_text = f"意图: {intent_name}\n动作骨架: {skel}\n完整操作: {full_text}"
        intent_vec = vectors[i].tolist()

        rec = {
            "id": f"{session_id}_frame_{i:06d}",
            "image": f"frames/{i:08d}.jpg",
            "timestamp_ns": int(row["timestamp_ns"]),
            "conversations": [
                {"from": "human", "value": HUMAN_PROMPT},
                {"from": "gpt", "value": gpt_text},
            ],
            "slow_system_output": {
                "intent_id": intent_id,
                "intent_name": intent_name,
                "intent_vector": intent_vec,
                "action_skeleton": skel,
            },
            "fast_system_input": {
                "intent_vector": intent_vec,
                "action_skeleton": skel,
            },
            "fast_system_output": {
                "category_id": cat_id,
                "category_name": str(row["category_name"]),
                "text_action": full_text,
                "numeric_action": cont,
                "decode_calibration": int(row.get("decode_calibration", 0)),
            },
        }
        rows.append(rec)
        intent_rows.append({"frame_id": i, "intent_id": intent_id, "intent_name": intent_name})

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pd.DataFrame(intent_rows).to_csv(out_intents, index=False)

    return BuildJsonlResult(
        jsonl_path=out_jsonl,
        intents_path=out_intents,
        num_frames=len(rows),
        intent_counts=dict(ints := pd.Series([r["slow_system_output"]["intent_id"] for r in rows]).value_counts().items()),
        category_counts=dict(pd.Series([r["fast_system_output"]["category_id"] for r in rows]).value_counts().items()),
    )
