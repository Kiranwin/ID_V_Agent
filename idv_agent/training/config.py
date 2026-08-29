"""训练配置（共享于 bc_fast / bc_slow）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingConfig:
    # 数据
    session_dirs: list[Path] = field(default_factory=list)
    image_size: int = 224
    max_text_len: int = 96

    # 训练
    batch_size: int = 4
    num_steps: int = 1000        # 总迭代步数（step 计，非 epoch）
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 42

    # 设备 / 精度
    device: str = "cpu"          # "cuda" / "cpu"，由调用方按可用性设置
    fp16: bool = False           # 2080Ti 无 BF16，需 FP16 + GradScaler
    num_workers: int = 0         # smoke 用 0；正式训练设 2~4

    # 日志 / 保存
    log_interval: int = 10
    save_interval: int = 200
    output_dir: Path = Path("checkpoints")
    resume_from: Path | None = None
