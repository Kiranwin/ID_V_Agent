"""从 session 提取 per_frame_actions.csv。

用法：
    python -m idv_agent.scripts.extract --session-dir data/sessions/<id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idv_agent.configs.schema import ExtractParams
from idv_agent.labels.extract import extract_session
from idv_agent.labels.intent import infer_intents_for_frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session-dir", type=Path, required=True)
    args = p.parse_args(argv)

    if not (args.session_dir / "events.csv").exists():
        print(f"[extract] 无 events.csv: {args.session_dir}")
        return 1
    df = extract_session(args.session_dir, params=ExtractParams())
    meta = {}
    mp = args.session_dir / "meta.json"
    fps = json.loads(mp.read_text(encoding="utf-8")).get("effective_fps", 30.0) if mp.exists() else 30.0
    intents, _ = infer_intents_for_frames(df, fps)
    print(f"[extract] {len(df)} 帧 -> {args.session_dir / 'per_frame_actions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
