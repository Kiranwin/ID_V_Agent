"""慢系统 Qwen 路径 collator：用 Qwen processor 把 PIL 图 + 对话编码成多模态输入。

供 bc_slow --use-qwen 使用。返回张量字段：
    pixel_values / input_ids / attention_mask / image_grid_thw / labels / numeric_action

labels 对齐策略：用 `processor.apply_chat_template(..., return_assistant_tokens_mask=True)`
解析出哪些 token 属于 assistant 回复（mask），reply 段真实 id 作为监督目标，其余
（user 图文 prompt、模板填充、第二条 HUMAN_PROMPT）全部置 -100。避免"整段 reply 都设
为目标、连模板内重复的 user 文本也被要求预测"的错误。

注：transformers 5.x 的 Qwen3-VL 支持 return_assistant_tokens_mask；本环境无实际 Qwen
模型无法端到端实证，真实 GPU 训练前需用 Qwen processor 校验（docs/06 已标注 M2 验证项）。
"""

from __future__ import annotations

import torch

from idv_agent.labels.build_jsonl import HUMAN_PROMPT


class QwenSlowCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        # 完整多轮对话：user(图+prompt) + assistant(回复)
        msgs = [[{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": HUMAN_PROMPT},
            ],
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": b["text"]}],
        }] for b in batch]

        template = self.processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False,
        )
        enc = self.processor(
            text=template,
            images=[b["image"] for b in batch],
            return_tensors="pt",
            padding=True,
            return_assistant_tokens_mask=True,
        )
        input_ids = enc["input_ids"]
        mask = enc.get("assistant_tokens_mask")
        if mask is None:
            # 老版本无 mask：退化为整个输入监督（保守，避免运行时崩溃）
            mask = torch.ones_like(input_ids, dtype=torch.bool)
        labels = torch.where(mask.bool(), input_ids, torch.full_like(input_ids, -100))

        return {
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc.get("image_grid_thw"),
            "mm_token_type_ids": enc.get("mm_token_type_ids"),
            "input_ids": input_ids,
            "attention_mask": enc.get("attention_mask"),
            "labels": labels,  # assistant 段监督；其余 -100
            "numeric_action": torch.as_tensor(
                [b["numeric_action"] for b in batch], dtype=torch.float32
            ),
        }
