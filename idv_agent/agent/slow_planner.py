"""慢系统在线规划 wrapper：图像 → Qwen3-VL generate → 解析 intent + skeleton → 写 SharedState。

设计（与 docs/02 一致）：慢层秒级、异步、不阻塞实时链路。用 `model.generate` 短回复，
从文本抓"意图"/"动作骨架"，回查 IntentCategory + SkeletonVocab。失败时保持上一次状态。

P6 明确：慢层实际刷新受 VLM 单次推断时间限制（4B 约 3-5s，0.2-0.3Hz），不宣称高频。
部署调度只负责「按需/低频重规划」，配合 EventDetector 在关键事件时触发。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from idv_agent.configs.intent import INTENT_VECTOR_DIM, IntentCategory, intent_to_vector
from idv_agent.labels.build_jsonl import HUMAN_PROMPT
from idv_agent.labels.tokenizer import SkeletonVocab
from idv_agent.model.game_actor_critic import GameActorCritic
from idv_agent.model.qwen_backbone_adapter import load_qwen3vl_backbone


_INTENT_RE = re.compile(r"意图\s*[:：]\s*(\w+)")
_SKELETON_RE = re.compile(r"动作骨架\s*[:：]\s*(.+)")


class SlowPlanner:
    def __init__(
        self,
        qwen_path: str | Path,
        slow_ckpt: Optional[Path],
        skeleton_vocab: SkeletonVocab,
        device: torch.device | str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_new_tokens: int = 32,
        do_sample: bool = False,
    ):
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.skeleton_vocab = skeleton_vocab

        backbone, processor = load_qwen3vl_backbone(
            str(qwen_path), dtype=dtype, device_map=self.device,
            apply_lora=True, gradient_checkpointing=False,
        )
        self.processor = processor
        self.model = GameActorCritic(
            backbone=backbone, hidden_size=backbone.hidden_size, numeric_dim=4,
        ).to(self.device).eval()
        for m in (self.model.numeric_head, self.model.value_head):
            m.to(torch.float32)

        if slow_ckpt is not None:
            from idv_agent.training.utils import load_checkpoint
            ckpt = load_checkpoint(slow_ckpt, self.model, map_location=self.device)
            print(f"[slow] 加载 {slow_ckpt} step={ckpt.get('step', '?')}")
        else:
            print("[slow] 未指定 ckpt，使用未微调权重")

        self._last_intent = IntentCategory.IDLE
        self._last_skeleton_idx = 0
        self._last_text = ""

    @torch.no_grad()
    def plan(self, pil_image) -> tuple[np.ndarray, int, str, str]:
        """对单帧规划，返回 (intent_vector[16], skeleton_idx, intent_name, raw_text)。

        可结合 MatchMemory.summary() 把对局记忆拼入 prompt（未在此默认注入，留给调用方）。
        """
        msgs = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": HUMAN_PROMPT},
            ],
        }]
        prompt_text = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        encoded = self.processor(
            text=[prompt_text], images=[pil_image], return_tensors="pt", padding=True,
        ).to(self.device)

        inner = self.model.backbone.model
        gen_ids = inner.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            pixel_values=encoded["pixel_values"],
            image_grid_thw=encoded.get("image_grid_thw"),
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            num_beams=1,
        )
        new_tokens = gen_ids[0, encoded["input_ids"].shape[1]:]
        text = self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        self._last_text = text

        intent = self._last_intent
        m = _INTENT_RE.search(text)
        if m is not None:
            name = m.group(1).strip().upper()
            try:
                intent = IntentCategory[name]
                self._last_intent = intent
            except KeyError:
                pass

        skeleton_idx = self._last_skeleton_idx
        m = _SKELETON_RE.search(text)
        if m is not None:
            sk = m.group(1).strip()
            if sk and sk != "(无)":
                try:
                    skeleton_idx = self.skeleton_vocab.encode(sk)
                    self._last_skeleton_idx = skeleton_idx
                except Exception:
                    pass

        intent_vec = intent_to_vector(intent)
        return intent_vec, skeleton_idx, intent.name.lower(), text
