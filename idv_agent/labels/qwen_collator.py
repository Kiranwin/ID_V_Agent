"""慢系统 Qwen 路径 collator：用 Qwen processor 把 PIL 图 + 对话编码成多模态输入。

供 bc_slow --use-qwen 使用。返回张量字段（pixel_values/input_ids/labels/attention_mask/
image_grid_thw/mm_token_type_ids），labels 对 gpt 回复部分监督、输入部分 mask。
"""

from __future__ import annotations

import torch
from torch.utils.data import default_collate

from idv_agent.labels.build_jsonl import HUMAN_PROMPT


class QwenSlowCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        images = [b["image"] for b in batch]           # PIL 图
        # 构造多模态对话
        msgs = []
        for b in batch:
            msgs.append([{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": HUMAN_PROMPT},
                ],
            }, {
                "role": "assistant",
                "content": [{"type": "text", "text": b["text"]}],
            }])

        enc = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
            return_tensors=None,
        )
        # 用 vision 分支编码图像
        prompt_texts = [self._user_text(b) for b in batch]
        processed = self.processor(
            text=[self.processor.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": HUMAN_PROMPT},
                ]}], tokenize=False, add_generation_prompt=True)
                for _ in batch],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return {
            "pixel_values": processed["pixel_values"],
            "input_ids": processed["input_ids"],
            "attention_mask": processed.get("attention_mask"),
            "image_grid_thw": processed.get("image_grid_thw"),
            "mm_token_type_ids": processed.get("mm_token_type_ids"),
            "numeric_action": torch.as_tensor([b["numeric_action"] for b in batch], dtype=torch.float32),
            "labels": None,  # 简易路径：不做 label-len 对齐，委托调用方填
        }

    def _user_text(self, b: dict) -> str:
        return HUMAN_PROMPT
