"""PyTorch Dataset：加载 session JSONL + 帧图。

两种模式：
- tensor 模式：图像用 cv2 + ImageNet normalize（FastController 用），返回像素 tensor。
- PIL 模式：保留 PIL 图像（Qwen3-VL processor 用）。

采样返回：
    pixel_values / image     图像
    category_id / intent_id       离散标签
    category_targets
    numeric_action[4]        连续标签
    intent_vector[16]        慢层输出条件（快层输入）
    skeleton_idx             FastController 骨架条件
    text                     慢层文本分支输入
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class SessionDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: list[Path],
        image_size: int = 224,
        mode: str = "tensor",   # "tensor" | "pil"
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
        session_root: Optional[Path] = None,  # image 相对路径的基。默认取每个 jsonl 的父目录下
    ):
        self.image_size = image_size
        self.mode = mode
        self._norm_mean = np.array(normalize_mean, dtype=np.float32).reshape(1, 1, 3)
        self._norm_std = np.array(normalize_std, dtype=np.float32).reshape(1, 1, 3)
        self.records: list[dict] = []
        self.base_dirs: list[Path] = []

        for jp in jsonl_paths:
            base = session_root or jp.parent
            with jp.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.records.append(rec)
                    self.base_dirs.append(base)

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rec: dict, base: Path):
        img_path = base / rec["image"]
        if self.mode == "pil":
            from PIL import Image
            img = Image.open(img_path).convert("RGB")
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size))
            return img
        # tensor 模式
        import cv2
        img = cv2.imread(str(img_path))          # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        arr = img.astype(np.float32) / 255.0
        arr = (arr - self._norm_mean) / self._norm_std
        return torch.from_numpy(arr.transpose(2, 0, 1))

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        base = self.base_dirs[idx]
        image = self._load_image(rec, base)

        fast_in = rec.get("fast_system_input", {})
        fast_out = rec.get("fast_system_output", {})
        slow_out = rec.get("slow_system_output", {})

        sample = {
            "image": image,
            "category_id": int(fast_out.get("category_id", 0)),
            "numeric_action": np.asarray(fast_out.get("numeric_action", [0, 0, 0, 0]), dtype=np.float32),
            "intent_vector": np.asarray(fast_in.get("intent_vector") or slow_out.get("intent_vector") or [0.0]*16, dtype=np.float32),
            "skeleton_str": str(fast_in.get("action_skeleton") or slow_out.get("action_skeleton") or "(无)"),
            "intent_id": int(slow_out.get("intent_id", 0)),
            "text": rec["conversations"][1]["value"] if len(rec.get("conversations", [])) > 1 else "",
        }
        return sample


def compute_class_weights(jsonl_paths: list[Path], num_categories: int,
                          mode: str = "inverse", eps: float = 1e-6) -> torch.Tensor:
    """按类别频率计算 CE 权重（P4 长尾缓解）。

    mode:
        none         -> 全 1（不加权）
        inverse      -> 1/count，再归一化到均值 1
        sqrt_inverse -> 1/sqrt(count)，更温和
    返回 [num_categories] float32 张量（CPU）。
    """
    counts = np.zeros(num_categories, dtype=np.float64)
    for jp in jsonl_paths:
        jp = Path(jp)
        if not jp.exists():
            continue
        with jp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                c = obj.get("fast_system_output", {}).get("category_id")
                if c is not None and 0 <= c < num_categories:
                    counts[c] += 1
    if mode == "none":
        return torch.ones(num_categories, dtype=torch.float32)

    total = max(counts.sum(), eps)
    freq = counts / total
    with np.errstate(divide="ignore"):
        if mode == "inverse":
            w = 1.0 / (freq + eps)
        elif mode == "sqrt_inverse":
            w = 1.0 / np.sqrt(freq + eps)
        else:
            raise ValueError(f"unknown mode: {mode}")
    # 归一化到均值 1，避免整体改变学习率尺度
    w = w / (w.sum() / max(num_categories, 1))
    return torch.as_tensor(w, dtype=torch.float32)

