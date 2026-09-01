"""从 session 提取 per_frame_actions.csv。

用法：
    python -m idv_agent.scripts.extract --session-dir data/vla_raw_sessions/<id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idv_agent.configs.schema import ExtractParams
from idv_agent.labels.extract import extract_session


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if not (args.session_dir / "events.csv").exists():
        print(f"[extract] 无 events.csv: {args.session_dir}")
        return 1
    df = extract_session(args.session_dir, params=ExtractParams())
    print(f"[extract] {len(df)} 帧 -> {args.session_dir / 'per_frame_actions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
