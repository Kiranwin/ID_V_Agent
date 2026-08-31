"""YOLO-nano 首轮训练入口。"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--data",type=Path,required=True); p.add_argument("--model",default="yolo11n.pt"); p.add_argument("--epochs",type=int,default=50); p.add_argument("--imgsz",type=int,default=640); p.add_argument("--batch",type=int,default=8); p.add_argument("--device",default="0"); p.add_argument("--project",type=Path,default=Path("checkpoints/yolo")); p.add_argument("--name",default="cipher_v1"); p.add_argument("--workers",type=int,default=2); a=p.parse_args(argv)
    # Keep Ultralytics settings/cache inside the workspace.  The managed
    # Windows session may deny writes to %APPDATA%/Ultralytics.
    config_dir = Path(".ultralytics").resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    try:
        from ultralytics import YOLO
    except ImportError as e: raise RuntimeError("缺少 ultralytics，请先 pip install ultralytics") from e
    YOLO(a.model).train(data=str(a.data),epochs=a.epochs,imgsz=a.imgsz,batch=a.batch,device=a.device,project=str(a.project),name=a.name,workers=a.workers,pretrained=True,patience=15,amp=True,plots=True)
    print(f"[yolo-train] best={a.project/a.name/'weights'/'best.pt'}"); return 0
if __name__=="__main__": raise SystemExit(main())
