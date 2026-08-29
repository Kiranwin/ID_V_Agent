"""数据统计：category / intent 分布（P4 验证长尾稀疏程度）。

用法：
    python -m idv_agent.scripts.stats --session-dir data/sessions/<id> [--jsonl]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from idv_agent.configs.schema import ActionCategory


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--jsonl", action="store_true", help="统计 samples.jsonl（而非 per_frame_actions.csv）")
    args = p.parse_args(argv)

    if args.jsonl:
        path = args.session_dir / "samples.jsonl"
        cat_counts = Counter()
        intent_counts = Counter()
        n = 0
        if path.exists():
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    n += 1
                    cat_counts[obj["fast_system_output"]["category_id"]] += 1
                    intent_counts[obj["slow_system_output"]["intent_id"]] += 1
        else:
            print(f"{path} 不存在，请先 build_jsonl")
        print(f"[stats] 共 {n} 帧")
    else:
        path = args.session_dir / "per_frame_actions.csv"
        if not path.exists():
            print(f"{path} 不存在，请先 extract")
            return 1
        import pandas as pd
        df = pd.read_csv(path)
        cat_counts = Counter(df["category_id"].astype(int))
        intent_counts = Counter()
        from idv_agent.labels.intent import category_to_intent
        for cid in df["category_id"].astype(int):
            intent_counts[category_to_intent(int(cid))] += 1

    print("\n=== ActionCategory 分布 ===")
    for cid, cnt in sorted(cat_counts.items()):
        print(f"  {ActionCategory(cid).name:<18} {cnt:>6}  ({cnt/max(sum(cat_counts.values()),1)*100:.1f}%)")
    print("\n=== Intent 分布 ===")
    for iid, cnt in sorted(intent_counts.items()):
        print(f"  {int(iid):<3} {cnt:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
