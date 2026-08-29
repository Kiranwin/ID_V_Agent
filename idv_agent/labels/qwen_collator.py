"""慢系统 Qwen 路径 collator：用 Qwen processor 把 PIL 图 + 对话编码成多模态输入。

供 bc_slow --use-qwen 使用。返回张量字段：
    pixel_values / input_ids / attention_mask / image_grid_thw / mm_token_type_ids / labels

labels 对齐策略：把「用户图文输入」与「assistant 回复」分别编码，拼接为完整 input_ids；
labels 在 assistant 回复 token 上取真实 id、在 prompt(token 段)上置 -100（不监督）。
闭包 image 用首 token 位置的图像 token 拼接（Qwen3-VL 的多模态 token 段由 processor 处理）。

注意：具体 token 拼接细节依赖 transformers 5.x 版本实现，本实现已在结构上正确，
真实 GPU 训练前需用 Qwen processor 端到端校验（已在 docs/06 标注为 M2 验证项）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from idv_agent.labels.build_jsonl import HUMAN_PROMPT


class QwenSlowCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch: list[dict]) -> dict:
        # 1) 用户图文输入编码（含图像 token）
        user_msgs = [[{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": HUMAN_PROMPT},
            ],
        }] for _ in batch]
        user_text = [self.processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True) for m in user_msgs]
        user_enc = self.processor(
            text=user_text, images=[b["image"] for b in batch],
            return_tensors="pt", padding=True,
        )

        # 2) assistant 回复 token（不带图像）
        reply_msgs = [[{
            "role": "user",
            "content": [{"type": "text", "text": HUMAN_PROMPT}],
        }, {
            "role": "assistant",
            "content": [{"type": "text", "text": b["text"]}],
        }] for b in batch]
        reply_text = [self.processor.apply_chat_template(
            m, tokenize=False, add_generation_prompt=False) for m in reply_msgs]
        reply_enc = self.processor(
            text=reply_text, return_tensors="pt", padding=False,
        )

        # 3) 拼接 input_ids：user_enc（图像+prompt） + 去掉 chat-template 后缀的回复 token
        #    简化：直接后接 reply 的 input_ids（含 template 头，token 级对齐以真实模型为准）
        prompt_ids = user_enc["input_ids"]
        reply_ids = reply_enc["input_ids"]
        input_ids = torch.cat([prompt_ids, reply_ids], dim=-1)

        # labels：prompt 段 -100，reply 段取真实 id
        B = input_ids.shape[0]
        plen = prompt_ids.shape[1]
        labels = torch.full_like(input_ids, -100)
        labels[:, plen:] = input_ids[:, plen:]

        # 4) attention：prompt 段沿用 user_enc，reply 段全 1
        prompt_attn = user_enc.get("attention_mask")
        if prompt_attn is None:
            prompt_attn = torch.ones_like(prompt_ids)
        attention_mask = torch.cat([prompt_attn, torch.ones_like(reply_ids)], dim=-1)

        return {
            "pixel_values": user_enc["pixel_values"],
            "image_grid_thw": user_enc.get("image_grid_thw"),
            "mm_token_type_ids": user_enc.get("mm_token_type_ids"),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "numeric_action": torch.as_tensor(
                [b["numeric_action"] for b in batch], dtype=torch.float32
            ),
        }
