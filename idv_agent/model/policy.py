"""策略抽象：把「给一帧 + 状态 → 出动作」统一成一个接口（P9 落地）。

M1 规则 agent / M2 学习快层 / M3 VLA 都实现 `Policy`，部署层（agent/realtime_agent）
只依赖本接口，调换实现零改动。避免「快层换 VLA 是重新设计」的路径：上层稳定，只换实现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PolicyOutput:
    category_id: int
    continuous: Tuple[float, float, float, float]   # (move_x, move_y, cam_dx, cam_dy)
    confidence: float = 0.0
    raw: object = None


class Policy:
    """策略接口。decide(frame, state) -> PolicyOutput。frame 为 BGR ndarray。"""

    def decide(self, frame, state) -> PolicyOutput:
        raise NotImplementedError


class RulePolicy(Policy):
    """M1：规则决策器（律师版「找→走→破译」兜底）。rule_agent 见 agent/rule_agent.py。"""

    def __init__(self, rule_agent):
        self.rule_agent = rule_agent

    def decide(self, frame, state) -> PolicyOutput:
        cat, cont = self.rule_agent.decide(state)
        return PolicyOutput(category_id=int(cat), continuous=tuple(cont))


class LearnedPolicy(Policy):
    """M2：学习型快层。用 FastControllerWithSkeleton 输出。"""

    def __init__(
        self,
        fast_model,
        device,
        image_size=224,
        normalize_mean=(0.485, 0.456, 0.406),
        normalize_std=(0.229, 0.224, 0.225),
    ):
        self.fast = fast_model.to(device).eval()
        self.device = device
        self.image_size = image_size
        self.mean = normalize_mean
        self.std = normalize_std

    def decide(self, frame, state) -> PolicyOutput:
        import cv2
        import numpy as np
        import torch
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        mean = np.array(self.mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(self.std, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        pixel = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        intent = torch.as_tensor(state["intent_vector"], dtype=torch.float32, device=self.device).unsqueeze(0)
        sk = torch.as_tensor([state["skeleton_idx"]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            sk_emb = self.fast.skeleton_emb(sk)
            out = self.fast.fast(
                pixel_values=pixel, intent_vector=intent, skeleton_embedding=sk_emb
            )
        cat_id = int(torch.argmax(out.category_logits, dim=-1).item())
        cont = out.continuous_vals.squeeze(0).clamp(-1, 1).cpu().tolist()
        conf = float(torch.softmax(out.category_logits, dim=-1).max().item())
        return PolicyOutput(category_id=cat_id, continuous=tuple(cont), confidence=conf)
