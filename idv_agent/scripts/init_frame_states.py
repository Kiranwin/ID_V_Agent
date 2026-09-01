"""Create an editable frame-state sidecar for an existing YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def init(dataset: Path, output: Path, *, overwrite: bool = False) -> int:
    manifest = dataset / "manifest.csv"
    if not manifest.is_file():
        raise ValueError(f"缺少 {manifest}")
    if output.exists() and not overwrite:
        raise ValueError(f"已存在 {output}；如需重建请显式使用 --overwrite（会清空人工状态）")
    rows = []
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append({
                "session": row["session"],
                "frame": int(Path(row["source"]).stem),
                "state": "",
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"[states] template={output.resolve()} frames={len(rows)}")
    print("[states] 请将每行 state 填为 decoding / idle / walking / chased")
    return len(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="创建 VG 帧状态标注模板")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        init(args.dataset, args.output or args.dataset / "frame_states.jsonl", overwrite=args.overwrite)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
