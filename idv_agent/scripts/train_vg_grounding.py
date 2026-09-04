"""Train the first VG grounding head from an ``M1_WK`` checkpoint.

This is the small, safe M2_VG baseline: Qwen + the WK adapter are loaded from
the parent checkpoint and frozen; a trainable visual projection and grounding
head learn presence/class and normalized box coordinates from
``vg_grounding.jsonl``.  The parent manifest is mandatory, so VG cannot
silently restart from the base model.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from idv_agent.model.spatial_agg import SpatialAgg
from idv_agent.model.spatial_grid_feature import grid_spatial_pool, per_image_rows_and_geo
from idv_agent.training.checkpoint_manifest import (
    build_manifest, load_manifest, sha256_file, write_manifest,
)


SOURCE_CLASSES = ("cipher_visible", "cipher_highlight", "interact_prompt")


def read_grounding(path: Path, *, repo_root: Path, split: str | None = None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"VG 数据不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 无效") from exc
            if not isinstance(row, dict) or not str(row.get("image", "")).strip():
                raise ValueError(f"{path}:{line_no} 缺少 image")
            if split and row.get("split") != split:
                continue
            image = Path(str(row["image"]))
            image = image if image.is_absolute() else repo_root / image
            if not image.is_file():
                raise ValueError(f"{path}:{line_no} 图像不存在: {image}")
            target = row.get("target")
            if target is not None:
                if not isinstance(target, dict) or target.get("source_class") not in SOURCE_CLASSES:
                    raise ValueError(f"{path}:{line_no} target.source_class 无效")
                box = target.get("bbox_xyxy_norm")
                if not isinstance(box, list) or len(box) != 4:
                    raise ValueError(f"{path}:{line_no} bbox_xyxy_norm 无效")
            row["_image_path"] = image
            rows.append(row)
    if not rows:
        raise ValueError(f"VG 数据为空: {path} split={split}")
    return rows


class GroundingDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        with Image.open(row["_image_path"]) as image:
            image = image.convert("RGB")
        target = row.get("target")
        if target is None:
            cls, box, positive = 0, [0.0] * 4, 0.0
        else:
            cls = SOURCE_CLASSES.index(target["source_class"]) + 1
            box, positive = [float(v) for v in target["bbox_xyxy_norm"]], 1.0
        return {"image": image, "question": str(row.get("question", "")),
                "class_target": torch.tensor(cls, dtype=torch.long),
                "box_target": torch.tensor(box, dtype=torch.float32),
                "positive": torch.tensor(positive, dtype=torch.float32),
                "id": str(row.get("id", index))}


class GroundingHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(hidden_size * 2), nn.Linear(hidden_size * 2, 512),
                                 nn.GELU(), nn.Linear(512, 256), nn.GELU())
        self.classifier = nn.Linear(256, 4)  # 0=none, 1..3=VG source classes
        self.box = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 4), nn.Sigmoid())

    def forward(self, visual: torch.Tensor, question: torch.Tensor):
        hidden = self.net(torch.cat((visual, question), dim=-1))
        return self.classifier(hidden), self.box(hidden)


def _collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {"images": [x["image"] for x in samples],
            "questions": [x["question"] for x in samples],
            "class_target": torch.stack([x["class_target"] for x in samples]),
            "box_target": torch.stack([x["box_target"] for x in samples]),
            "positive": torch.stack([x["positive"] for x in samples]),
            "ids": [x["id"] for x in samples]}


def _load_parent(init_checkpoint: Path, model_path: str | None, device: torch.device,
                 dtype: torch.dtype):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    parent = load_manifest(init_checkpoint / "manifest.json", require_artifacts=True)
    if parent["stage"] != "M1_WK":
        raise ValueError(f"VG 初始化必须使用 M1_WK，收到 {parent['stage']}")
    base = model_path or parent["base_model"]
    processor = AutoProcessor.from_pretrained(str(base))
    base_model = AutoModelForImageTextToText.from_pretrained(str(base), dtype=dtype, device_map=None).to(device)
    adapter_path = init_checkpoint / parent["artifacts"]["adapter"]
    model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=False).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, processor, parent, str(base)


def _question_embedding(model: nn.Module, processor: Any, question: str, device: torch.device) -> torch.Tensor:
    tokenizer = getattr(processor, "tokenizer", processor)
    encoded = tokenizer(question or "画面中有什么？", return_tensors="pt", add_special_tokens=True)
    ids = encoded["input_ids"].to(device)
    with torch.no_grad():
        return model.get_input_embeddings()(ids).mean(dim=1).squeeze(0).float()


def _visual_embedding(model: nn.Module, processor: Any, image: Image.Image, device: torch.device,
                      agg: SpatialAgg, k: int = 8) -> torch.Tensor:
    image_processor = getattr(processor, "image_processor", processor)
    inputs = {kd: (v.to(device) if isinstance(v, torch.Tensor) else v)
              for kd, v in dict(image_processor(images=image, return_tensors="pt")).items()}
    image_features = getattr(model, "get_image_features", None)
    if image_features is None and hasattr(model, "base_model"):
        image_features = getattr(model.base_model, "get_image_features", None)
    if image_features is None:
        raise RuntimeError("Qwen 模型未暴露 get_image_features，无法运行 VG baseline")
    with torch.no_grad():
        out = image_features(pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"])
        if hasattr(out, "last_hidden_state"):
            out = out.last_hidden_state
        elif isinstance(out, (tuple, list)):
            out = next(x for x in out if isinstance(x, torch.Tensor))
        if out.ndim == 2:
            out = out.unsqueeze(0)
    # 保留 k×k 保胞位置后再用可训 agg 压回单向量。
    grid_thw = inputs["image_grid_thw"]
    grids = [tuple(int(x) for x in grid_thw[0].tolist())]        # (t,h,w)
    L = out.shape[1]
    vision_config = getattr(getattr(model, "config", None), "vision_config", None)
    merge_size = int(getattr(vision_config, "spatial_merge_size", 2))
    rows_per, geo = per_image_rows_and_geo(grids, merge_size, L)
    rows, (gh, gw) = rows_per[0], geo[0]
    tok = out[0]                                             # [L,D]
    cells = grid_spatial_pool(tok.float(), rows, gh, gw, k)  # [k*k, D]
    # cells 由冻结视觉塔 no_grad 产出；agg 是可训（train 模式走出 no_grad 段）。
    return agg(cells.unsqueeze(0)).squeeze(0)


def _split_rows(rows: list[dict[str, Any]], fraction: float, seed: int):
    if fraction <= 0:
        return rows, []
    rng = random.Random(seed)
    by_class: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("target", {}).get("source_class", "none")) if row.get("target") else "none"
        by_class.setdefault(key, []).append(row)
    train, val = [], []
    for group in by_class.values():
        rng.shuffle(group)
        n = min(max(1, round(len(group) * fraction)), len(group) - 1)
        val.extend(group[:n]); train.extend(group[n:])
    return train, val


def train(args: argparse.Namespace) -> dict[str, Any]:
    data = Path(args.data); repo_root = Path(args.repo_root).resolve()
    rows = read_grounding(data, repo_root=repo_root, split=args.split or None)
    train_rows, val_rows = _split_rows(rows, args.val_fraction, args.seed)
    if args.dry_run:
        counts = {"none": 0, **{name: 0 for name in SOURCE_CLASSES}}
        for row in rows:
            key = row["target"]["source_class"] if row.get("target") else "none"
            counts[key] += 1
        return {"stage": "M2_VG", "rows": len(rows), "train": len(train_rows),
                "val": len(val_rows), "classes": counts, "parent": args.init_checkpoint}
    if not args.init_checkpoint:
        raise ValueError("VG 训练必须提供 --init-checkpoint M1_WK")
    device = torch.device(args.device); use_amp = device.type == "cuda"
    dtype = torch.float16 if use_amp else torch.float32
    model, processor, parent, base_model = _load_parent(Path(args.init_checkpoint), args.model_path or None, device, dtype)
    # Raw Qwen vision features are projected by a small trainable layer; the
    # parent WK language adapter and Qwen vision tower remain frozen.
    spatial_k = int(getattr(args, "spatial_k", 8))
    if spatial_k < 1:
        raise ValueError("spatial-k 必须为正数")
    # 阶段二空间保胞通道：k×k 保胞(SpatialGrid 折叠) + 可训 SpatialAgg 压回单向量
    agg = SpatialAgg(dim=1024, n_query=1, k=spatial_k).to(device=device, dtype=torch.float32)
    # Raw Qwen vision features are projected by a small trainable layer; the
    # parent WK language adapter and Qwen vision tower remain frozen.
    first_raw = _visual_embedding(model, processor, train_rows[0]["image"], device, agg=agg, k=spatial_k)
    visual_projection = nn.Linear(first_raw.numel(), 2560, bias=False).to(device=device, dtype=torch.float32)
    head = GroundingHead(2560).to(device=device, dtype=torch.float32)
    params = list(visual_projection.parameters()) + list(head.parameters())
    if agg is not None:
        params += list(agg.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loader = DataLoader(GroundingDataset(train_rows), batch_size=max(1, args.batch_size), shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(GroundingDataset(val_rows), batch_size=max(1, args.batch_size), shuffle=False, collate_fn=_collate) if val_rows else None
    history: list[float] = []; iterator = iter(loader); started = time.perf_counter()
    model.eval(); visual_projection.train(); head.train()
    agg.train()
    def run_batch(batch: dict[str, Any]):
        visual = torch.stack([visual_projection(
            _visual_embedding(model, processor, image, device, agg=agg, k=spatial_k))
            for image in batch["images"]])
        question = torch.stack([_question_embedding(model, processor, q, device) for q in batch["questions"]])
        logits, boxes = head(visual, question)
        cls = batch["class_target"].to(device); target_box = batch["box_target"].to(device); positive = batch["positive"].to(device)
        loss_cls = nn.functional.cross_entropy(logits, cls)
        positive_mask = positive > 0
        loss_box = nn.functional.smooth_l1_loss(boxes[positive_mask], target_box[positive_mask]) if bool(positive_mask.any()) else boxes.sum() * 0
        return loss_cls + args.box_weight * loss_box, loss_cls.detach(), loss_box.detach()
    for _step in range(1, args.steps + 1):
        try: batch = next(iterator)
        except StopIteration: iterator = iter(loader); batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            loss, _, _ = run_batch(batch)
        if use_amp:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(params, 1.0); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        history.append(float(loss.detach().cpu()))
    val_loss = None
    if val_loader:
        visual_projection.eval(); head.eval()
        if agg is not None:
            agg.eval()
        values = []
        with torch.no_grad():
            for batch in val_loader:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp): values.append(float(run_batch(batch)[0].cpu()))
        val_loss = sum(values) / len(values) if values else None
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    state = {"visual_projection": visual_projection.state_dict(), "grounding_head": head.state_dict()}
    if agg is not None:
        state["spatial_agg"] = agg.state_dict()
    torch.save(state, output_dir / "grounding_head.pt")
    processor.save_pretrained(output_dir / "processor")
    trainable = ["visual_projection", "grounding_head"]
    if agg is not None:
        trainable.append("spatial_agg")
    manifest = build_manifest(stage="M2_VG", parent=str(Path(args.init_checkpoint).resolve()), base_model=base_model,
        adapters=["lora_wk", "grounding_head", "spatial_agg"], frozen=["qwen_vision_tower", "lora_wk"],
        trainable=trainable, data={"vg": str(data).replace("\\", "/"), "vg_sha256": sha256_file(data)},
        counts={"train": len(train_rows), "val": len(val_rows)}, precision="fp16+GradScaler" if use_amp else "fp32",
        training={"method": "grounding_sft", "steps": args.steps, "learning_rate": args.lr, "batch_size": args.batch_size, "box_weight": args.box_weight, "spatial_k": spatial_k},
        schema_versions=["vg.grounding.v2"], artifacts={"grounding_head": "grounding_head.pt", "processor": "processor"})
    write_manifest(output_dir / "manifest.json", manifest)
    result = {"stage": "M2_VG", "train_records": len(train_rows), "val_records": len(val_rows), "steps": args.steps,
              "initial_loss": history[0] if history else None, "final_loss": history[-1] if history else None,
              "val_loss": val_loss, "checkpoint": str(output_dir.resolve()), "parent": manifest["parent"], "elapsed_sec": time.perf_counter() - started}
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True); parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--model-path", default=""); parser.add_argument("--output-dir", default="checkpoints/M2_VG")
    parser.add_argument("--repo-root", default="."); parser.add_argument("--split", choices=("train", "val"), default=None)
    parser.add_argument("--val-fraction", type=float, default=0.1); parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--box-weight", type=float, default=2.0); parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--spatial-k", type=int, default=8,
                        help="k×k 保胞数；默认 8。")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--seed", type=int, default=42); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv); torch.manual_seed(args.seed)
    try: result = train(args)
    except (OSError, ValueError, RuntimeError) as exc: parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
