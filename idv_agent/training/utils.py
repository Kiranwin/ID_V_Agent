"""训练工具：种子/计时/目录/checkpoint 保存。"""

from __future__ import annotations

import random
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_elapsed(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s"


class AverageMeter:
    """带指数加权均值（recent_mean）的标量平均器。"""

    def __init__(self, alpha: float = 0.98):
        self.alpha = alpha
        self.count = 0
        self.sum = 0.0
        self._recent = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.sum += float(value)
        self._recent = (self.alpha * self._recent + (1 - self.alpha) * float(value))

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count else 0.0

    @property
    def recent_mean(self) -> float:
        return self._recent


def auto_timestamped_dir(base: Path, run_name: str) -> Path:
    from datetime import datetime
    return base / f"{run_name}_{datetime.now():%Y%m%d_%H%M%S}"


def prepare_output_dir(output_dir: Path, backup_existing: bool = True,
                       allow_overwrite: bool = False) -> Path:
    """处理输出目录：已有 ckpt 时备份 / 允许覆盖 / 拒绝。

    规则：output_dir 有内容 &&
         backup_existing -> 移到 backup_<ts>/
         否则 allow_overwrite -> 保留（直接覆盖）
         否则 -> 报错（要求换个目录）。
    """
    output_dir = Path(output_dir)
    has_existing = output_dir.exists() and any(output_dir.iterdir())
    if not has_existing:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    if backup_existing:
        from datetime import datetime
        backup = output_dir.parent / f"{output_dir.name}_backup_{datetime.now():%Y%m%d_%H%M%S}"
        shutil.move(str(output_dir), str(backup))
        output_dir.mkdir(parents=True)
        return output_dir
    if allow_overwrite:
        return output_dir
    raise RuntimeError(
        f"{output_dir} 已有内容；加 --allow-overwrite 覆盖或换个输出目录"
    )


def save_checkpoint(path: Path, step: int, model, optimizer=None,
                    extra: Optional[dict] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, str(path))


def load_checkpoint(path: Path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt
