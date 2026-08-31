"""M1 采集会话完整性检查（只读）。"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

def validate_session(session_dir: Path) -> list[str]:
    errors=[]
    for name in ("meta.json","events.csv","mouse_positions.csv"):
        if not (session_dir/name).exists(): errors.append(f"缺少 {name}")
    frames_dir=session_dir/"frames"
    if not frames_dir.is_dir(): errors.append("缺少 frames/ 目录")
    if errors: return errors
    try: meta=json.loads((session_dir/"meta.json").read_text(encoding="utf-8"))
    except Exception as exc: return [f"meta.json 无法解析: {exc}"]
    try:
        events=pd.read_csv(session_dir/"events.csv")
        for col in ("timestamp_ns","kind","code"):
            if col not in events.columns: errors.append(f"events.csv 缺少列 {col}")
    except Exception as exc: events=None; errors.append(f"events.csv 无法读取: {exc}")
    try:
        positions=pd.read_csv(session_dir/"mouse_positions.csv")
        for col in ("timestamp_ns","x","y"):
            if col not in positions.columns: errors.append(f"mouse_positions.csv 缺少列 {col}")
    except Exception as exc: positions=None; errors.append(f"mouse_positions.csv 无法读取: {exc}")
    files=sorted(frames_dir.glob("*.jpg"))+sorted(frames_dir.glob("*.png"))
    expected=int(meta.get("num_frames") or 0)
    if expected<=0: errors.append("meta.num_frames 缺失或为 0")
    if expected!=len(files): errors.append(f"帧数不一致: meta={expected}, files={len(files)}")
    tsp=session_dir/"frame_timestamps.csv"
    if tsp.exists():
        try:
            ts=pd.read_csv(tsp)
            if not {"frame_id","timestamp_ns"}.issubset(ts.columns): errors.append("frame_timestamps.csv 缺少 frame_id/timestamp_ns 列")
            elif len(ts)!=len(files): errors.append(f"时间戳数量不一致: timestamps={len(ts)}, files={len(files)}")
            elif ts["timestamp_ns"].duplicated().any(): errors.append("frame_timestamps.csv 存在重复 timestamp_ns")
        except Exception as exc: errors.append(f"frame_timestamps.csv 无法读取: {exc}")
    for label,df in (("events",events),("mouse_positions",positions)):
        if df is not None and len(df) and not df["timestamp_ns"].is_monotonic_increasing:
            errors.append(f"{label}.csv timestamp_ns 未排序")
    return errors

def main(argv=None)->int:
    p=argparse.ArgumentParser(description="检查录制 session 是否可进入 extract/build_jsonl")
    p.add_argument("--session-dir",type=Path,required=True); args=p.parse_args(argv)
    errors=validate_session(args.session_dir)
    if errors:
        print(f"[validate] FAIL {args.session_dir}"); [print(f"  - {e}") for e in errors]; return 1
    print(f"[validate] PASS {args.session_dir}"); return 0

if __name__=="__main__": raise SystemExit(main())
