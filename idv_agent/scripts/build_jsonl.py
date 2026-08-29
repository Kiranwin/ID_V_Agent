"""构建 JSONL 训练数据。

用法：
    python -m idv_agent.scripts.build_jsonl --session-dir data/sessions/<id>
（先运行 extract 生成 per_frame_actions.csv）
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idv_agent.labels.build_jsonl import build_session_jsonl


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if not (args.session_dir / "per_frame_actions.csv").exists():
        print("[build_jsonl] 请先运行 extract（缺少 per_frame_actions.csv）")
        return 1
    result = build_session_jsonl(args.session_dir)
    print(f"[build_jsonl] {result.num_frames} 帧 -> {result.jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
