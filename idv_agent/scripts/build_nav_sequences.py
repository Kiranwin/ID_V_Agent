"""从已提取动作和帧构造局部导航时序样本清单。

不复制图片、不生成伪标签；只生成 JSONL 元数据，供后续 3 帧→未来动作模型使用。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

def build(session: Path, output: Path, history: int = 3, stride: int = 3, horizon: int = 8):
    if history < 2 or stride < 1 or horizon < 1: raise ValueError("history/stride/horizon 参数无效")
    frames = sorted((session/"frames").glob("*.jpg"), key=lambda p:int(p.stem))
    if not frames: raise ValueError(f"没有帧: {session/ 'frames'}")
    actions = {}
    csv_path=session/"per_frame_actions.csv"
    if csv_path.is_file():
        import csv
        with csv_path.open(encoding="utf-8-sig", errors="ignore") as f:
            for row in csv.DictReader(f):
                try:
                    idx = row.get("frame_idx", row.get("frame_id", row.get("frame", -1)))
                    actions[int(idx)] = row
                except (TypeError,ValueError): pass
    rows=[]
    for i in range((history-1)*stride, len(frames)-horizon, stride):
        hist=[frames[i-j*stride].relative_to(session).as_posix() for j in range(history-1,-1,-1)]
        future=[]
        for k in range(1,horizon+1):
            row=actions.get(i+k,{})
            future.append({"frame_idx":i+k,"category":row.get("category",row.get("category_name",row.get("category_id",""))),"move_x":row.get("move_x",""),"move_y":row.get("move_y",""),"cam_dx":row.get("cam_dx",""),"cam_dy":row.get("cam_dy","")})
        rows.append({"session":session.name,"anchor_frame":i,"history_frames":hist,"future_actions":future})
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    print(f"[nav] session={session.name} samples={len(rows)} history={history} stride={stride} horizon={horizon} output={output}")
    return len(rows)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("session",type=Path);p.add_argument("--output",type=Path,default=None);p.add_argument("--history",type=int,default=3);p.add_argument("--stride",type=int,default=3);p.add_argument("--horizon",type=int,default=8);a=p.parse_args(argv)
    out=a.output or a.session/"nav_sequences.jsonl"; build(a.session,out,a.history,a.stride,a.horizon); return 0
if __name__=="__main__": raise SystemExit(main())
