"""把 Qwen3-VL-4B-Instruct 包装成统一 BackboneOutput 接口。

关键事实（transformers 5.x）：
- 模型类 Qwen3VLForConditionalGeneration；hidden_size 2560（供 VLA 主干使用）
- vocab_size 151936；pixel_values 为 [num_patches, 1536]
- 必传：input_ids, attention_mask, pixel_values, image_grid_thw

LoRA 默认配置：target=q/k/v/o_proj, r=8, alpha=32。
"""

from __future__ import annotations

from pathlib import Path
from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn

from idv_agent.model.backbone_types import BackboneOutput
from idv_agent.model.temporal import TaskConditionCache


ACT_RAW_GRID_SIZE = 2
ACT_RAW_GRID_K = 8
ACT_FEATURE_DIM = ACT_RAW_GRID_SIZE * ACT_RAW_GRID_SIZE * 1024


def pool_raw_spatial_grid(raw_features: torch.Tensor, *, input_k: int = ACT_RAW_GRID_K,
                          output_k: int = ACT_RAW_GRID_SIZE) -> torch.Tensor:
    """Deterministically pool a frozen ``[B,k*k,D]`` raster to ``[B,4*D]``.

    This is deliberately not an ``nn.Module``: ACT must retain the spatial
    signal exposed by the frozen vision tower without a learned raster
    bottleneck.  The output order is row-major ``2x2`` cells, matching the
    raw Qwen raster order.
    """
    if raw_features.ndim != 3:
        raise ValueError("raw_features 必须是 [B,k*k,D]")
    input_k = int(input_k)
    output_k = int(output_k)
    if input_k < 1 or output_k < 1 or input_k % output_k:
        raise ValueError("input_k 必须是 output_k 的整数倍")
    if raw_features.shape[1] != input_k * input_k:
        raise ValueError(f"raw_features 必须有 {input_k * input_k} 个 spatial cells")
    block = input_k // output_k
    pooled = raw_features.reshape(raw_features.shape[0], input_k, input_k, raw_features.shape[-1])
    pooled = pooled.reshape(raw_features.shape[0], output_k, block, output_k, block,
                            raw_features.shape[-1]).mean(dim=(2, 4))
    return pooled.reshape(raw_features.shape[0], output_k * output_k * raw_features.shape[-1])


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


def _spatial_grid_forward(images, adapter, k: int, merge_size: int):
    """在 adapter 上用 SpatialGridEncoder 取 k×k 保胞 token（不进 nn 图）。

    SpatialGridEncoder 持有 adapter 引用仅用于取 per-token；此处每次现场构造，
    避免把它作为 nn.Module 挂进 adapter 造成 Adapter↔Grid 环引用(影响 training
    开关递归)。
    """
    from idv_agent.model.spatial_grid_feature import SpatialGridEncoder

    return SpatialGridEncoder(adapter, k=int(k), merge_size=int(merge_size))(images)


class Qwen3VLBackboneAdapter(nn.Module):
    """把 Qwen3VLForConditionalGeneration 适配到 BackboneOutput 接口。"""

    def __init__(self, model: nn.Module, processor: Any | None = None,
                 act_feature_dim: int = ACT_FEATURE_DIM):
        super().__init__()
        self.model = model
        self.processor = processor
        cfg = self._underlying_config()
        self.hidden_size = cfg.text_config.hidden_size
        self.vocab_size = cfg.text_config.vocab_size
        self.act_feature_dim = int(act_feature_dim)
        if self.act_feature_dim < 1:
            raise ValueError("act_feature_dim 必须为正数")
        self.visual_projection = nn.LazyLinear(self.hidden_size, bias=False)
        self.condition_projection = nn.Linear(self.hidden_size * 2, self.act_feature_dim)
        # ACT projects every k×k cell before fixed-raster fusion.  It is lazy
        # because small unit-test adapters and the real Qwen tower expose
        # different raw widths; the training warm-up materializes it before
        # optimizer/checkpoint construction.
        self.spatial_cell_projector: Optional[nn.Module] = None
        # Spatial cells are the only raw visual representation.  The aggregator
        # is materialized lazily from the vision feature width, or loaded from
        # the VG checkpoint before ACT training.
        self.spatial_agg: Optional[nn.Module] = None
        self.spatial_k: int = 8
        self.spatial_merge_size = int(getattr(getattr(cfg, "vision_config", None),
                                              "spatial_merge_size", 2))
        self._spatial_forward: Any = partial(_spatial_grid_forward, adapter=self,
                                              k=self.spatial_k, merge_size=self.spatial_merge_size)

    def enable_spatial_agg(self, k: int = 8, state: dict | None = None,
                           dim: int = 1024) -> None:
        """打开 k×k 保胞 + 可训 SpatialAgg 通道；state 来自 M2_VG_spatial。

        切到该模式后 spatial_raw_tokens 输出每帧 k² 保胞 token，encode_frames
        统一经 SpatialAgg 聚合为单向量。
        """
        k = int(k)
        if k < 1:
            raise ValueError("spatial k 必须为正数")
        from idv_agent.model.spatial_agg import SpatialAgg

        self.spatial_agg = SpatialAgg(dim=int(dim), n_query=1, k=k)
        if state is not None:
            self.spatial_agg.load_state_dict(state)
        self.spatial_k = k
        # 对齐 adapter 视觉侧设备即可；dtype 交给 autocast / 显式 float in 调用点。
        try:
            devref = next(self.model.parameters()).device
        except StopIteration:
            devref = torch.device("cpu")
        self.spatial_agg.to(device=devref)
        # 不把 SpatialGridEncoder 挂成 nn.Module 子层（会与 adapter 环引用导致
        # .train()/.eval() 递归栈溢出），用普通函数捕获 self 即可。
        self._spatial_forward = partial(_spatial_grid_forward, adapter=self, k=self.spatial_k,
                                        merge_size=self.spatial_merge_size)

    def _ensure_spatial_agg(self, dim: int) -> nn.Module:
        if self.spatial_agg is None:
            from idv_agent.model.spatial_agg import SpatialAgg
            self.spatial_agg = SpatialAgg(dim=int(dim), n_query=1, k=self.spatial_k)
            self.spatial_agg.to(device=self.device)
        elif self.spatial_agg.dim != int(dim) or self.spatial_agg.k != self.spatial_k:
            raise ValueError("SpatialAgg 与 spatial token shape 不匹配")
        return self.spatial_agg

    def _ensure_spatial_cell_projector(self, dim: int) -> nn.Module:
        if self.spatial_cell_projector is None:
            from idv_agent.model.spatial_cell_projector import SpatialCellProjector
            self.spatial_cell_projector = SpatialCellProjector(
                dim=int(dim), k=self.spatial_k, output_dim=self.hidden_size)
            self.spatial_cell_projector.to(device=self.device)
        elif (self.spatial_cell_projector.dim != int(dim)
              or self.spatial_cell_projector.k != self.spatial_k
              or self.spatial_cell_projector.output_dim != self.hidden_size):
            raise ValueError("SpatialCellProjector 与 spatial token shape 不匹配")
        return self.spatial_cell_projector

    def spatial_raw_tokens(self, images: list[Any]) -> torch.Tensor:
        """返回 [N, k*k, dim] 每帧保胞 token（冻结视觉塔，无 agg）。"""
        return self._spatial_forward(images).float()

    def _spatial_raw(self, images: list[Any]) -> torch.Tensor:
        """把每帧 k×k 保胞 token 经可训 agg 压成 [N, dim]。"""
        cells = self.spatial_raw_tokens(images)          # [N, k2, dim]
        return self._aggregate_spatial_tokens(cells)

    def _aggregate_spatial_tokens(self, cells: torch.Tensor) -> torch.Tensor:
        if cells.ndim != 3:
            raise ValueError("spatial cells 必须是 [N,C,D]")
        return self._ensure_spatial_agg(cells.shape[-1])(cells)

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

    def _raw_visual_forward_batch(self, inputs: list[dict[str, Any]]) -> torch.Tensor:
        """Run the frozen vision tower and return ordered ``[N,C,D]`` cells."""
        if not inputs:
            return torch.empty((0, self.spatial_k ** 2, 0), device=self.device)
        return self._spatial_forward(inputs)

    def _visual_forward_batch(self, inputs: list[dict[str, Any]]) -> torch.Tensor:
        """Run the image tower once for several independently processed images."""
        return self.project_raw_features(self._raw_visual_forward_batch(inputs),
                                         self._dummy_task_cache(inputs[0]))

    def _dummy_task_cache(self, _inputs: dict[str, Any]) -> TaskConditionCache:
        zeros = torch.zeros(self.hidden_size, device=self.device)
        return TaskConditionCache(zeros, zeros, task_id="internal")

    def encode_raw_frames(self, images: list[Any], *, micro_batch_size: int | None = None) -> torch.Tensor:
        """Return deterministic frozen-tower k×k grid features for ``images``."""
        if not images:
            return torch.empty((0, self.spatial_k ** 2, 0), device=self.device)
        limit = len(images) if micro_batch_size is None else max(1, int(micro_batch_size))
        was_training = self.training
        self.eval()
        try:
            with torch.inference_mode():
                outputs = []
                for start in range(0, len(images), limit):
                    prepared = [self._prepare_image(image) for image in images[start:start + limit]]
                    outputs.append(self._raw_visual_forward_batch(prepared))
                return torch.cat(outputs, dim=0)
        finally:
            self.train(was_training)

    def project_raw_features(self, raw_features: torch.Tensor,
                             task_cache: TaskConditionCache) -> torch.Tensor:
        """Return the conditioned ACT feature used by the temporal prior path."""
        visual = self.project_raw_visual_features(raw_features)
        condition = self._condition_vector(task_cache, visual.shape[0], visual)
        return visual + condition

    def project_raw_visual_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Return pure deterministic raw-grid features, with no task condition."""
        if raw_features.ndim == 3:
            visual = pool_raw_spatial_grid(raw_features.clone(), input_k=self.spatial_k)
        elif raw_features.ndim == 2:
            visual = raw_features.clone()
        else:
            raise ValueError("raw_features 必须是 [N,C,D] spatial tokens 或 [N,D] 向量")
        if visual.shape[-1] != self.act_feature_dim:
            raise ValueError(
                f"ACT raw grid 输出维度 {visual.shape[-1]} 与 act_feature_dim "
                f"{self.act_feature_dim} 不匹配；Qwen 预期为 {ACT_FEATURE_DIM}"
            )
        return visual

    def _condition_vector(self, task_cache: TaskConditionCache, count: int,
                          reference: torch.Tensor) -> torch.Tensor:
        ins = task_cache.instruction_embedding.to(reference).reshape(-1)
        mode = task_cache.mode_embedding.to(reference).reshape(-1)
        if ins.numel() != self.hidden_size or mode.numel() != self.hidden_size:
            raise ValueError("task embedding 维度必须等于 Qwen text hidden_size")
        self.condition_projection.to(device=reference.device)
        cond_input = torch.cat((ins, mode)).unsqueeze(0).expand(int(count), -1)
        condition = self.condition_projection(
            cond_input.to(self.condition_projection.weight.dtype)
        ).to(reference.dtype)
        return condition

    def uncondition_features(self, conditioned_features: torch.Tensor,
                             task_cache: TaskConditionCache) -> torch.Tensor:
        """Recover pure visual features from an exact conditioned cache entry."""
        if conditioned_features.ndim not in (2, 3) or conditioned_features.shape[-1] != self.act_feature_dim:
            raise ValueError("conditioned_features 必须是 [...,act_feature_dim]")
        reference = conditioned_features.reshape(-1, conditioned_features.shape[-1])
        condition = self._condition_vector(task_cache, reference.shape[0], reference)
        return (reference - condition).reshape_as(conditioned_features)

    def encode_frames(self, images: list[Any], task_cache: TaskConditionCache | None = None,
                      *, micro_batch_size: int | None = None) -> torch.Tensor:
        """Encode multiple images with one visual-tower call per micro-batch."""
        if task_cache is None:
            raise ValueError("必须提供 task_cache")
        if not images:
            return torch.empty((0, self.act_feature_dim), device=self.device)
        limit = len(images) if micro_batch_size is None else max(1, int(micro_batch_size))
        outputs = []
        for start in range(0, len(images), limit):
            chunk = images[start:start + limit]
            cells = self.spatial_raw_tokens(chunk)
            outputs.append(self.project_raw_features(cells, task_cache))
        return torch.cat(outputs, dim=0)

    def encode_frames_with_visual(self, images: list[Any], task_cache: TaskConditionCache,
                                  *, micro_batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode once and return ``(conditioned_prior, pure_visual)`` features."""
        if task_cache is None:
            raise ValueError("必须提供 task_cache")
        if not images:
            empty = torch.empty((0, self.act_feature_dim), device=self.device)
            return empty, empty.clone()
        limit = len(images) if micro_batch_size is None else max(1, int(micro_batch_size))
        conditioned_outputs = []
        visual_outputs = []
        for start in range(0, len(images), limit):
            cells = self.spatial_raw_tokens(images[start:start + limit])
            visual = self.project_raw_visual_features(cells)
            conditioned_outputs.append(visual + self._condition_vector(task_cache, visual.shape[0], visual))
            visual_outputs.append(visual)
        return torch.cat(conditioned_outputs, dim=0), torch.cat(visual_outputs, dim=0)

    def encode_frame(self, image: Any, task_cache: TaskConditionCache | None = None,
                     *, instruction_embedding: torch.Tensor | None = None,
                     mode_embedding: torch.Tensor | None = None) -> torch.Tensor:
        """Single-frame image encoding; never tokenizes or runs the text tower."""
        if task_cache is not None:
            instruction_embedding = task_cache.instruction_embedding
            mode_embedding = task_cache.mode_embedding
        if instruction_embedding is None or mode_embedding is None:
            raise ValueError("必须提供 task_cache 或 instruction/mode embedding")
        return self.encode_frames([image], task_cache)[0]

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing_enable()
