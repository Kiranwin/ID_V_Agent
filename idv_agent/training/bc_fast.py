"""FastController 行为克隆训练（快系统 M2）。

用法（CLI）：
    # 基线（dummy vision，CPU 也能跑）
    python -m idv_agent.training.bc_fast --session data/sessions/<id> --steps 1000 --batch-size 8

    # 真实 SigLIP2 vision（需 GPU）
    python -m idv_agent.training.bc_fast --session data/sessions/<id> --steps 200 \
        --use-siglip --device cuda --fp16

设计：
- FastController 接受预计算 skeleton_embedding，训练时套 FastControllerWithSkeleton
  包装 skeleton_idx -> embedding + 条件 dropout。
- --use-siglip 时 DummyVisionEncoder → SigLIP2-base/16。
- 统一包名：python -m idv_agent.training.bc_fast（cwd=项目根目录）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from idv_agent.configs.intent import INTENT_VECTOR_DIM
from idv_agent.configs.schema import NUM_CATEGORIES, NUM_CONTINUOUS
from idv_agent.labels.collator import FastControllerCollator
from idv_agent.labels.dataset import SessionDataset, compute_class_weights
from idv_agent.labels.tokenizer import SkeletonVocab
from idv_agent.model.dummy_vision import DummyVisionEncoder
from idv_agent.model.fast_controller import FastController
from idv_agent.training.config import TrainingConfig
from idv_agent.training.utils import (
    AverageMeter,
    auto_timestamped_dir,
    format_elapsed,
    load_checkpoint,
    prepare_output_dir,
    save_checkpoint,
    set_seed,
)


class FastControllerWithSkeleton(nn.Module):
    """训练/部署便利 wrapper：内置 skeleton 嵌入表 + 条件 dropout。"""

    def __init__(
        self,
        fast: FastController,
        skeleton_vocab_size: int,
        skeleton_dim: int,
        intent_dropout: float = 0.0,
        skeleton_dropout: float = 0.0,
    ):
        super().__init__()
        self.fast = fast
        self.skeleton_emb = nn.Embedding(skeleton_vocab_size, skeleton_dim)
        self.intent_dropout = intent_dropout
        self.skeleton_dropout = skeleton_dropout

    def forward(self, batch: dict):
        sk_emb = self.skeleton_emb(batch["skeleton_idx"])
        intent = batch["intent_vector"]
        if self.training and self.intent_dropout > 0:
            keep = (torch.rand(intent.shape[0], 1, device=intent.device)
                    > self.intent_dropout).to(intent.dtype)
            intent = intent * keep
        if self.training and self.skeleton_dropout > 0:
            keep = (torch.rand(sk_emb.shape[0], 1, device=sk_emb.device)
                    > self.skeleton_dropout).to(sk_emb.dtype)
            sk_emb = sk_emb * keep
        return self.fast(
            pixel_values=batch["pixel_values"],
            intent_vector=intent,
            skeleton_embedding=sk_emb,
            category_labels=batch["category_labels"],
            continuous_labels=batch["continuous_labels"],
        )


def _infinite_loader(loader: DataLoader) -> Iterator[dict]:
    while True:
        for batch in loader:
            yield batch


def train_fast_controller(
    config: TrainingConfig,
    skeleton_dim: int = 64,
    vision_hidden_size: int = 192,
    use_siglip: bool = False,
    siglip_freeze: bool = False,
    class_balance: str = "none",
    intent_dropout: float = 0.0,
    skeleton_dropout: float = 0.0,
) -> FastControllerWithSkeleton:
    set_seed(config.seed)
    device = torch.device(config.device)
    use_fp16 = bool(config.fp16 and device.type == "cuda")

    # 骨架词表 & 数据
    jsonl_paths = [p / "samples.jsonl" for p in config.session_dirs]
    skeleton_vocab = SkeletonVocab.build_from_jsonls(jsonl_paths)
    print(f"[bc_fast] skeleton vocab size: {len(skeleton_vocab)}")

    if use_siglip:
        norm_mean, norm_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        image_size = 224
    else:
        norm_mean, norm_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        image_size = config.image_size

    dataset = SessionDataset(
        jsonl_paths=[jp for jp in jsonl_paths if jp.exists()],
        image_size=image_size,
        mode="tensor",
        normalize_mean=norm_mean,
        normalize_std=norm_std,
    )
    print(f"[bc_fast] dataset size: {len(dataset)}, image_size={image_size}, "
          f"normalize={'siglip' if use_siglip else 'imagenet'}")
    if len(dataset) == 0:
        raise RuntimeError("dataset 为空：请先运行 build_jsonl 生成 samples.jsonl")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=FastControllerCollator(skeleton_vocab=skeleton_vocab),
        num_workers=config.num_workers,
        drop_last=len(dataset) >= config.batch_size,
    )

    # 视觉编码
    if use_siglip:
        from idv_agent.model.siglip_vision_adapter import load_siglip_vision_encoder
        try:
            from modelscope import snapshot_download
            siglip_path = snapshot_download("google/siglip2-base-patch16-224")
        except Exception:
            siglip_path = "google/siglip2-base-patch16-224"
        vision, _, vhid, _ = load_siglip_vision_encoder(
            siglip_path, dtype=torch.float32, device=device, freeze=siglip_freeze,
        )
        vision_hidden_size = vhid
        print(f"[bc_fast] using SigLIP2 vision (hidden={vhid}, freeze={siglip_freeze})")
    else:
        vision = DummyVisionEncoder(hidden_size=vision_hidden_size)
        print(f"[bc_fast] using dummy vision (hidden={vision_hidden_size})")

    # class weights（P4）
    class_weights: Optional[torch.Tensor] = None
    if class_balance != "none":
        class_weights = compute_class_weights(
            [jp for jp in jsonl_paths if jp.exists()], NUM_CATEGORIES, mode=class_balance
        )
        nz = (class_weights != 1.0).sum().item()
        print(f"[bc_fast] class_balance={class_balance}, nontrivial={nz}/{NUM_CATEGORIES}, "
              f"min={class_weights.min().item():.3f} max={class_weights.max().item():.3f}")
        class_weights = class_weights.to(device)

    fast_inner = FastController(
        vision_encoder=vision,
        vision_hidden_size=vision_hidden_size,
        intent_dim=INTENT_VECTOR_DIM,
        skeleton_dim=skeleton_dim,
        num_categories=NUM_CATEGORIES,
        num_continuous=NUM_CONTINUOUS,
        class_weights=class_weights,
    )
    model = FastControllerWithSkeleton(
        fast=fast_inner,
        skeleton_vocab_size=len(skeleton_vocab),
        skeleton_dim=skeleton_dim,
        intent_dropout=intent_dropout,
        skeleton_dropout=skeleton_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[bc_fast] params: {n_params/1e6:.2f}M (训练 {n_train/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    start_step = 0
    if config.resume_from is not None and Path(config.resume_from).exists():
        ckpt = load_checkpoint(config.resume_from, model, optimizer, map_location=device)
        start_step = ckpt["step"]
        print(f"[bc_fast] 从 {config.resume_from} 恢复，step={start_step}")

    loss_meter = AverageMeter()
    cat_meter = AverageMeter()
    cont_meter = AverageMeter()
    iterator = _infinite_loader(loader)
    t0 = time.perf_counter()
    model.train()
    for step in range(start_step, config.num_steps):
        batch = next(iterator)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        loss_meter.add(loss.item())
        cat_meter.add(out.category_loss.item() if out.category_loss is not None else 0.0)
        cont_meter.add(out.continuous_loss.item() if out.continuous_loss is not None else 0.0)

        if (step + 1) % config.log_interval == 0:
            print(
                f"[bc_fast] step {step+1:>5d}/{config.num_steps} "
                f"loss(recent)={loss_meter.recent_mean:.4f} "
                f"cat={cat_meter.recent_mean:.4f} cont={cont_meter.recent_mean:.4f} "
                f"elapsed={format_elapsed(time.perf_counter() - t0)}"
            )
        if (step + 1) % config.save_interval == 0:
            ckpt_path = config.output_dir / "fast_controller" / f"step_{step+1:06d}.pt"
            save_checkpoint(ckpt_path, step + 1, model, optimizer)
            print(f"[bc_fast] 保存 {ckpt_path}")

    final_path = config.output_dir / "fast_controller" / "final.pt"
    save_checkpoint(final_path, config.num_steps, model, optimizer, extra={
        "skeleton_vocab_size": len(skeleton_vocab),
        "skeleton_dim": skeleton_dim,
        "use_siglip": use_siglip,
        "vision_hidden_size": vision_hidden_size,
    })
    print(f"[bc_fast] 训练完成，最终 checkpoint: {final_path}")
    return model


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, action="append", required=True,
                   help="session 目录，可多次指定")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fp16", action="store_true", help="启用 FP16 + GradScaler（仅 GPU）")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--run-name", default="fast")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--use-siglip", action="store_true", help="用 SigLIP2-base 替换 dummy vision")
    p.add_argument("--siglip-freeze", action="store_true", help="冻结 SigLIP backbone")
    p.add_argument("--class-balance", choices=["none", "inverse", "sqrt_inverse"],
                   default="none", help="类别加权：none/inverse/sqrt_inverse")
    p.add_argument("--intent-dropout", type=float, default=0.0)
    p.add_argument("--skeleton-dropout", type=float, default=0.0)
    args = p.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = auto_timestamped_dir(Path("checkpoints"), args.run_name)
    args.output_dir = prepare_output_dir(args.output_dir)
    print(f"[bc_fast] output_dir: {args.output_dir}")

    cfg = TrainingConfig(
        session_dirs=args.session,
        batch_size=args.batch_size,
        num_steps=args.steps,
        lr=args.lr,
        image_size=args.image_size,
        device=args.device,
        fp16=args.fp16,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )
    train_fast_controller(
        cfg, use_siglip=args.use_siglip, siglip_freeze=args.siglip_freeze,
        class_balance=args.class_balance,
        intent_dropout=args.intent_dropout,
        skeleton_dropout=args.skeleton_dropout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
