"""把 Qwen3-VL-4B-Instruct 包装成 BackboneOutput 接口，与 dummy_backbone 鸭子兼容。

关键事实（transformers 5.x）：
- 模型类 Qwen3VLForConditionalGeneration；hidden_size 2560（接 GameActorCritic heads）
- vocab_size 151936；pixel_values 为 [num_patches, 1536]
- 必传：input_ids, attention_mask, pixel_values, image_grid_thw

LoRA 默认配置：target=q/k/v/o_proj, r=8, alpha=32。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from idv_agent.model.dummy_backbone import BackboneOutput


def default_qwen_lora_config():
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_qwen3vl_backbone(
    model_path: str | Path,
    dtype: torch.dtype = torch.float16,
    device_map: str | dict | None = "cuda",
    apply_lora: bool = True,
    lora_config=None,
    gradient_checkpointing: bool = False,
):
    """加载 Qwen3-VL-4B-Instruct。返回 (backbone_adapter, processor)。"""
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import get_peft_model

    processor = AutoProcessor.from_pretrained(str(model_path))
    base_model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=dtype, device_map=device_map,
    )
    if gradient_checkpointing:
        base_model.gradient_checkpointing_enable()

    if apply_lora:
        cfg = lora_config or default_qwen_lora_config()
        base_model = get_peft_model(base_model, cfg)

    return Qwen3VLBackboneAdapter(base_model), processor


class Qwen3VLBackboneAdapter(nn.Module):
    """把 Qwen3VLForConditionalGeneration 适配到 BackboneOutput 接口。"""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        cfg = self._underlying_config()
        self.hidden_size = cfg.text_config.hidden_size
        self.vocab_size = cfg.text_config.vocab_size

    def _underlying_config(self):
        m = self.model
        for attr in ("model",):
            if hasattr(m, "config"):
                return m.config
            if hasattr(m, attr):
                m = getattr(m, attr)
        return m.config

    def forward(
        self,
        *,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        attention_mask: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BackboneOutput:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )
        return BackboneOutput(
            loss=out.loss if hasattr(out, "loss") else None,
            logits=out.logits,
            hidden_states=list(out.hidden_states) if output_hidden_states else [],
        )

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()
