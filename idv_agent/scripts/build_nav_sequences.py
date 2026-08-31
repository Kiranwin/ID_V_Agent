"""从已提取动作和帧构造局部预测导航时序样本清单。

不复制图片、不生成伪标签；只生成 JSONL 元数据，供后续 3 帧→未来动作模型使用。
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

MACRO_ACTIONS = ("forward", "left", "right", "stop_observe")

def _number(value, default=0.0):
    """Parse CSV numeric fields without letting blanks/invalid values poison a sample."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _macro(rows):
    """Collapse ~0.2s of frame actions into one navigation macro."""
    import statistics
    if not rows:
        raise ValueError("宏动作至少需要一帧")
    mx = statistics.mean(_number(r.get("move_x")) for r in rows)
    my = statistics.mean(_number(r.get("move_y")) for r in rows)
    cam_dx = statistics.mean(_number(r.get("cam_dx")) for r in rows)
    cam_dy = statistics.mean(_number(r.get("cam_dy")) for r in rows)
    if abs(mx) < 0.25 and my > 0.35:
        action = "forward"
    elif mx < -0.25:
        action = "left"
    elif mx > 0.25:
        action = "right"
    else:
        action = "stop_observe"
    frame_ids = []
    for r in rows:
        frame_id = r.get("frame_idx")
        if frame_id is not None:
            try:
                frame_ids.append(int(frame_id))
            except (TypeError, ValueError):
                pass
    return {"action": action, "move_x": round(mx, 4), "move_y": round(my, 4),
            "cam_dx": round(cam_dx, 4), "cam_dy": round(cam_dy, 4),
            "source_frames": frame_ids}

def build(session: Path, output: Path, history: int = 3, stride: int = 3,
          horizon: int = 4, macro_frames: int = 6):
    if history < 2 or stride < 1 or horizon < 1 or macro_frames < 1:
        raise ValueError("history/stride/horizon/macro_frames 参数无效")
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
                    idx = int(idx)
                    # Normalize the identifier so generated provenance is stable
                    # regardless of whether the CSV calls it frame_id or frame_idx.
                    row["frame_idx"] = idx
                    actions[idx] = row
                except (TypeError, ValueError):
                    pass
    rows=[]
    future_span = horizon * macro_frames
    for i in range((history-1)*stride, len(frames)-future_span, stride):
        hist=[frames[i-j*stride].relative_to(session).as_posix() for j in range(history-1,-1,-1)]
        future=[]
        for k in range(horizon):
            rows_k = [actions.get(i + k*macro_frames + j,
                                  {"frame_idx": i + k*macro_frames + j})
                      for j in range(1, macro_frames + 1)]
            future.append(_macro(rows_k))
        rows.append({"session":session.name,"anchor_frame":i,"history_frames":hist,"future_actions":future})
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    print(f"[nav] session={session.name} samples={len(rows)} history={history} stride={stride} horizon={horizon} macro_frames={macro_frames} output={output}")
    return len(rows)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("session",type=Path);p.add_argument("--output",type=Path,default=None);p.add_argument("--history",type=int,default=3);p.add_argument("--stride",type=int,default=3);p.add_argument("--horizon",type=int,default=4);p.add_argument("--macro-frames",type=int,default=6,help="每个宏动作覆盖帧数，30 FPS 下 6 帧约 0.2 秒");a=p.parse_args(argv)
    out=a.output or a.session/"nav_sequences.jsonl"; build(a.session,out,a.history,a.stride,a.horizon,a.macro_frames); return 0
if __name__=="__main__": raise SystemExit(main())
