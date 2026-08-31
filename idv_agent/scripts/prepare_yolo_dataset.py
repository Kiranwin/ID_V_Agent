"""准备密码机目标检测的 YOLO 数据集目录（不自动生成伪标签）。

示例：
    python -m idv_agent.scripts.prepare_yolo_dataset \
        --sessions 20260831_113220,20260831_132206 \
        --output data/yolo_cipher --stride 5

脚本只复制已有 JPEG，并为每张图片创建空的 ``.txt`` 标签文件；
空标签表示负样本，正样本需使用 CVAT/LabelImg/Roboflow 等工具人工框选。
按 session 划分 train/val，避免同一局相邻帧同时出现在训练和验证集。
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path


CLASSES = ("cipher_visible", "cipher_highlight")


def _session_dirs(root: Path, names: str | None) -> list[Path]:
    if names:
        dirs = [root / n.strip() for n in names.split(",") if n.strip()]
    else:
        dirs = sorted(p for p in root.iterdir() if p.is_dir())
    missing = [str(p) for p in dirs if not p.is_dir()]
    if missing:
        raise ValueError(f"session 目录不存在: {', '.join(missing)}")
    return dirs


def _frames(session_dir: Path) -> list[Path]:
    return sorted(session_dir.glob("frames/*.jpg"))


def prepare(*, sessions_root: Path, session_names: str | None,
            output: Path, stride: int, val_ratio: float,
            seed: int, max_per_session: int | None) -> int:
    if stride <= 0:
        raise ValueError("stride 必须为正整数")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 (0,1) 内")
    sessions = _session_dirs(sessions_root, session_names)
    if len(sessions) < 2:
        raise ValueError("至少需要 2 局 session，才能按局划分 train/val")

    rng = random.Random(seed)
    shuffled = list(sessions)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    val_sessions = {p.name for p in shuffled[:n_val]}

    # Create the standard Ultralytics layout.
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, str | int]] = []
    copied = 0
    for session in sessions:
        split = "val" if session.name in val_sessions else "train"
        frames = _frames(session)[::stride]
        if max_per_session is not None:
            frames = frames[:max_per_session]
        for src in frames:
            # Prefix session id to avoid collisions (all sessions use 00000000.jpg).
            name = f"{session.name}_{src.stem}.jpg"
            dst_img = output / "images" / split / name
            dst_lbl = output / "labels" / split / f"{session.name}_{src.stem}.txt"
            shutil.copy2(src, dst_img)
            dst_lbl.write_text("", encoding="utf-8")
            manifest_rows.append({
                "image": str(dst_img.as_posix()),
                "label": str(dst_lbl.as_posix()),
                "session": session.name,
                "source": str(src.as_posix()),
                "split": split,
            })
            copied += 1

    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.yaml").write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names: {list(CLASSES)!r}\n",
        encoding="utf-8",
    )
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=("image", "label", "session", "source", "split"))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"[yolo] sessions={len(sessions)} train_sessions={len(sessions)-len(val_sessions)} "
          f"val_sessions={len(val_sessions)} images={copied} stride={stride}")
    print(f"[yolo] output={output.resolve()}")
    print("[yolo] classes=" + ", ".join(f"{i}:{n}" for i, n in enumerate(CLASSES)))
    print("[yolo] 空 txt 是负样本；请人工标注后再训练")
    return copied


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="准备密码机 YOLO 标注目录")
    p.add_argument("--sessions-root", type=Path, default=Path("data/sessions"))
    p.add_argument("--sessions", default=None, help="逗号分隔 session id；默认使用全部 session")
    p.add_argument("--output", type=Path, default=Path("data/yolo_cipher"))
    p.add_argument("--stride", type=int, default=5, help="每隔多少帧抽一张")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-session", type=int, default=None)
    args = p.parse_args(argv)
    try:
        prepare(sessions_root=args.sessions_root, session_names=args.sessions,
                output=args.output, stride=args.stride, val_ratio=args.val_ratio,
                seed=args.seed, max_per_session=args.max_per_session)
    except ValueError as exc:
        p.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
