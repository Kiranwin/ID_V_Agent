"""把 Qwen3-VL-4B-Instruct 包装成统一 BackboneOutput 接口。

关键事实（transformers 5.x）：
- 模型类 Qwen3VLForConditionalGeneration；hidden_size 2560（供 VLA 主干使用）
- vocab_size 151936；pixel_values 为 [num_patches, 1536]
- 必传：input_ids, attention_mask, pixel_values, image_grid_thw

LoRA 默认配置：target=q/k/v/o_proj, r=8, alpha=32。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

from idv_agent.model.backbone_types import BackboneOutput
from idv_agent.model.temporal import TaskConditionCache


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

    return Qwen3VLBackboneAdapter(base_model, processor=processor), processor


class Qwen3VLBackboneAdapter(nn.Module):
    """把 Qwen3VLForConditionalGeneration 适配到 BackboneOutput 接口。"""

    def __init__(self, model: nn.Module, processor: Any | None = None):
        super().__init__()
        self.model = model
        self.processor = processor
        cfg = self._underlying_config()
        self.hidden_size = cfg.text_config.hidden_size
        self.vocab_size = cfg.text_config.vocab_size
        self.visual_projection = nn.LazyLinear(self.hidden_size, bias=False)
        self.condition_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)

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

    @property
    def device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def encode_task_once(self, instruction: str, mode: str, *, task_id: str = "episode") -> TaskConditionCache:
        """Compute episode-constant text embeddings exactly once."""
        if self.processor is None:
            raise RuntimeError("encode_task_once 需要 processor")
        if not instruction or not mode:
            raise ValueError("instruction/mode 不能为空")
        embedding = self.model.get_input_embeddings()

        def embed(text: str) -> torch.Tensor:
            encoded = self.processor(text=text, return_tensors="pt", add_special_tokens=True)
            ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            with torch.no_grad():
                return embedding(ids.to(self.device)).mean(dim=1).squeeze(0).detach()

        return TaskConditionCache(embed(instruction), embed(f"<mode:{mode}>"),
                                  task_id=task_id, mode=mode)

    def _move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
                for k, v in inputs.items()}

    def _prepare_image(self, image: Any) -> dict[str, Any]:
        if isinstance(image, dict):
            if "pixel_values" not in image:
                raise ValueError("预处理 image dict 必须包含 pixel_values")
            return self._move_inputs(dict(image))
        if self.processor is None:
            raise RuntimeError("encode_frame 需要 processor 或预处理后的 image dict")
        image_processor = getattr(self.processor, "image_processor", self.processor)
        return self._move_inputs(dict(image_processor(images=image, return_tensors="pt")))

    def _visual_forward(self, inputs: dict[str, Any]) -> torch.Tensor:
        # Qwen3-VL exposes a public image-only helper that performs the
        # vision tower + multimodal projector without invoking the decoder.
        image_features = getattr(self.model, "get_image_features", None)
        if image_features is None and hasattr(self.model, "model"):
            image_features = getattr(self.model.model, "get_image_features", None)
        if image_features is not None and "image_grid_thw" in inputs:
            out = image_features(pixel_values=inputs["pixel_values"],
                                 image_grid_thw=inputs["image_grid_thw"])
            if hasattr(out, "last_hidden_state"):
                out = out.last_hidden_state
            elif isinstance(out, (tuple, list)):
                out = next(x for x in out if isinstance(x, torch.Tensor))
            if out.ndim == 2:
                out = out.unsqueeze(0)
            if out.ndim != 3:
                raise ValueError("Qwen get_image_features 输出必须为 [B,L,D]")
            pooled = out.mean(dim=1)
            # LazyLinear may not have been materialized when the adapter was
            # moved to CUDA; materialize/move it on the feature device.
            self.visual_projection.to(device=pooled.device)
            pooled = pooled.to(dtype=self.visual_projection.weight.dtype)
            return self.visual_projection(pooled)
        visual = getattr(self.model, "visual", None)
        if visual is None and hasattr(self.model, "model"):
            visual = getattr(self.model.model, "visual", None)
        if visual is None:
            raise RuntimeError("Qwen 模型未暴露 visual 视觉塔")
        # Transformers 5.x Qwen3-VL names these arguments hidden_states/grid_thw
        # (the processor already converts pixels to patch embeddings).  Keep a
        # fallback for older/custom visual towers using pixel_values.
        if "image_grid_thw" in inputs:
            kwargs = {"hidden_states": inputs["pixel_values"],
                      "grid_thw": inputs["image_grid_thw"]}
        else:
            kwargs = {"pixel_values": inputs["pixel_values"]}
        try:
            out = visual(**kwargs)
        except TypeError:
            try:
                out = visual(pixel_values=inputs.get("pixel_values"),
                             image_grid_thw=inputs.get("image_grid_thw"))
            except TypeError:
                out = visual(inputs.get("pixel_values"), inputs.get("image_grid_thw"))
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            out = next(x for x in out if isinstance(x, torch.Tensor))
        if not isinstance(out, torch.Tensor) or out.ndim not in (2, 3):
            raise ValueError("Qwen visual 输出必须为 [B,L,D] Tensor")
        if out.ndim == 2:
            out = out.unsqueeze(0)
        pooled = out.mean(dim=1)
        self.visual_projection.to(device=pooled.device)
        pooled = pooled.to(dtype=self.visual_projection.weight.dtype)
        return self.visual_projection(pooled)

    def encode_frame(self, image: Any, task_cache: TaskConditionCache | None = None,
                     *, instruction_embedding: torch.Tensor | None = None,
                     mode_embedding: torch.Tensor | None = None) -> torch.Tensor:
        """Single-frame image encoding; never tokenizes or runs the text tower."""
        if task_cache is not None:
            instruction_embedding = task_cache.instruction_embedding
            mode_embedding = task_cache.mode_embedding
        if instruction_embedding is None or mode_embedding is None:
            raise ValueError("必须提供 task_cache 或 instruction/mode embedding")
        visual = self._visual_forward(self._prepare_image(image))
        ins = instruction_embedding.to(visual).reshape(-1)
        mode = mode_embedding.to(visual).reshape(-1)
        if ins.numel() != self.hidden_size or mode.numel() != self.hidden_size:
            raise ValueError("task embedding 维度必须等于 Qwen text hidden_size")
        self.condition_projection.to(device=visual.device)
        cond_input = torch.cat((ins, mode)).unsqueeze(0).to(self.condition_projection.weight.dtype)
        condition = self.condition_projection(cond_input).to(visual.dtype)
        return (visual + condition).squeeze(0)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()
