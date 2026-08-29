"""FastController 数据 collator：把 Dataset 样本聚合为 batch 张量。

返回张量：
    pixel_values        [B, 3, H, W]
    intent_vector       [B, intent_dim]
    skeleton_idx        [B] long（用给定 SkeletonVocab 编码）
    category_labels     [B] long
    continuous_labels   [B, 4]
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from idv_agent.labels.tokenizer import SkeletonVocab


class FastControllerCollator:
    def __init__(self, skeleton_vocab: Optional[SkeletonVocab] = None):
        self.skeleton_vocab = skeleton_vocab or SkeletonVocab()

    def __call__(self, batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]
        if isinstance(images[0], torch.Tensor):
            pixel_values = torch.stack(images)
        else:  # PIL
            import torchvision.transforms.functional as TF
            from PIL import Image
            pixel_values = torch.stack([TF.to_tensor(img) for img in images])

        intent_vector = torch.as_tensor(
            np.array([b["intent_vector"] for b in batch], dtype=np.float32), dtype=torch.float32
        )
        category_labels = torch.as_tensor([b["category_id"] for b in batch], dtype=torch.long)
        continuous_labels = torch.as_tensor(
            np.array([b["numeric_action"] for b in batch], dtype=np.float32), dtype=torch.float32
        )

        skeleton_idx = torch.as_tensor(
            [self.skeleton_vocab.encode(b["skeleton_str"]) for b in batch], dtype=torch.long
        )

        return {
            "pixel_values": pixel_values,
            "intent_vector": intent_vector,
            "skeleton_idx": skeleton_idx,
            "category_labels": category_labels,
            "continuous_labels": continuous_labels,
        }
