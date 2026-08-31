"""按每个 session 的时间顺序重划分 YOLO 数据集。"""
from __future__ import annotations
import argparse, re, shutil
from collections import defaultdict
from pathlib import Path

CLASSES = ("cipher_visible", "cipher_highlight", "interact_prompt", "decoding_state")

def split_dataset(src: Path, dst: Path, val_ratio: float = 0.2):
    if not 0 < val_ratio < 0.5: raise ValueError("val_ratio 必须在 (0, 0.5) 内")
    groups = defaultdict(list)
    for split in ("train", "val", "test"):
        # Accept both Ultralytics layouts: split/images and images/split.
        if (src/split/"images").is_dir():
            ip, lp = src/split/"images", src/split/"labels"
        else:
            ip, lp = src/"images"/split, src/"labels"/split
        if not ip.is_dir(): continue
        for img in ip.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}: continue
            m = re.match(r"^(.+)_([0-9]+)$", img.stem)
            if not m: raise ValueError(f"文件名无法解析: {img.name}")
            label = lp/(img.stem+".txt")
            if not label.is_file(): raise ValueError(f"缺少标签: {label}")
            groups[m.group(1)].append((int(m.group(2)), img, label))
    if len(groups) < 2: raise ValueError("至少需要两局 session")
    for s in ("train", "val"):
        (dst/s/"images").mkdir(parents=True, exist_ok=True); (dst/s/"labels").mkdir(parents=True, exist_ok=True)
    nt=nv=0
    for sid, rows in sorted(groups.items()):
        rows.sort(key=lambda x:x[0]); cut=max(1,min(len(rows)-1,int(round(len(rows)*(1-val_ratio)))))
        for target, subset in (("train",rows[:cut]),("val",rows[cut:])):
            for _,img,label in subset:
                shutil.copy2(img,dst/target/"images"/img.name); shutil.copy2(label,dst/target/"labels"/label.name)
                if target=="train": nt+=1
                else: nv+=1
    # Use an absolute path because Ultralytics resolves ``path: .`` against
    # its process working directory rather than the YAML location on Windows.
    (dst/"data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: train/images\nval: val/images\n"
        f"nc: {len(CLASSES)}\nnames: {list(CLASSES)!r}\n",
        encoding="utf-8",
    )
    print(f"[yolo-split] sessions={len(groups)} train={nt} val={nv} output={dst.resolve()}")
    return nt,nv

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("src",type=Path); p.add_argument("dst",type=Path); p.add_argument("--val-ratio",type=float,default=.2); a=p.parse_args(argv)
    try: split_dataset(a.src,a.dst,a.val_ratio)
    except ValueError as e: p.error(str(e))
    return 0
if __name__=="__main__": raise SystemExit(main())
