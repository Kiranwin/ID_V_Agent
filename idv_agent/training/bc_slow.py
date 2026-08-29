"""GameActorCritic（慢系统）行为克隆训练。

用法：
    # dummy backbone（CPU 也能跑，验证链路）
    python -m idv_agent.training.bc_slow --session data/sessions/<id> --steps 1000 --batch-size 4

    # 真实 Qwen3-VL-4B + LoRA（需要 GPU）
    python -m idv_agent.training.bc_slow --session data/sessions/<id> --steps 200 \
        --use-qwen --device cuda --fp16 --batch-size 1

设计：
- --use-qwen：Qwen3-VL-4B-Instruct + LoRA（qwen_collator 编码多模态输入）。
- 否则：DummyBackbone + SimpleWordTokenizer，便于离线开发/CI。
- value_target 当前默认关闭（数据无 reward 信号），仅 text + numeric 两项（P3 清醒标注）。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader

from idv_agent.configs.schema import NUM_CONTINUOUS
from idv_agent.labels.build_jsonl import HUMAN_PROMPT
from idv_agent.labels.dataset import SessionDataset
from idv_agent.labels.tokenizer import SimpleWordTokenizer, SkeletonVocab
from idv_agent.model.dummy_backbone import DummyBackbone
from idv_agent.model.game_actor_critic import GameActorCritic
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


def _build_simple_tokenizer(jsonl_paths: list[Path]) -> SimpleWordTokenizer:
    toks = SimpleWordTokenizer()
    for p in jsonl_paths:
        if not Path(p).exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for c in rec.get("conversations", []):
                    toks.build_vocab_from_texts([c.get("value", "")])
    return toks


class SlowCollator:
    """dummy 分支 collator：把 SessionDataset 样本（含 img + text + numeric）编码成输入。"""

    def __init__(self, tokenizer: SimpleWordTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict:
        max_len = 0
        enc = []
        prompts = []
        for b in batch:
            prompt = self.tokenizer.encode(HUMAN_PROMPT, add_special=True)
            target = self.tokenizer.encode(b["text"], add_special=True)
            prompts.append(prompt)
            enc.append(target)
            max_len = max(max_len, len(prompt) + len(target))

        input_ids, labels = [], []
        for prompt, target in zip(prompts, enc):
            seq = prompt + target
            pad = max_len - len(seq)
            seq += [self.tokenizer.pad_id] * pad
            lbl = [-100] * len(prompt) + target + [-100] * pad
            input_ids.append(seq)
            labels.append(lbl)

        images = [b["image"] for b in batch]           # tensor (default_collate 已 stack)
        import numpy as np
        import torch
        pixel = torch.stack(images)
        numeric = torch.as_tensor(
            np.array([b["numeric_action"] for b in batch], dtype=np.float32), dtype=torch.float32
        )
        return {
            "pixel_values": pixel,
            "input_ids": torch.as_tensor(input_ids, dtype=torch.long),
            "labels": torch.as_tensor(labels, dtype=torch.long),
            "numeric_action": numeric,
        }


def train_actor_critic(
    config: TrainingConfig,
    hidden_size: int = 64,
    use_qwen: bool = False,
) -> GameActorCritic:
    set_seed(config.seed)
    device = torch.device(config.device)
    use_fp16 = bool(config.fp16 and device.type == "cuda")

    jsonl_paths = [p / "samples.jsonl" for p in config.session_dirs]
    existing = [jp for jp in jsonl_paths if jp.exists()]
    skeleton_vocab = SkeletonVocab.build_from_jsonls(existing)
    print(f"[bc_slow] skeleton vocab: {len(skeleton_vocab)}")

    if use_qwen:
        from idv_agent.labels.qwen_collator import QwenSlowCollator
        from idv_agent.model.qwen_backbone_adapter import load_qwen3vl_backbone
        try:
            from modelscope import snapshot_download
            qwen_path = snapshot_download("Qwen/Qwen3-VL-4B-Instruct")
        except Exception:
            qwen_path = "Qwen/Qwen3-VL-4B-Instruct"
        print(f"[bc_slow] loading Qwen3-VL-4B-Instruct from {qwen_path}")
        dtype = torch.float16 if use_fp16 else torch.float32
        backbone, processor = load_qwen3vl_backbone(
            qwen_path, dtype=dtype, device_map=device,
            apply_lora=True, gradient_checkpointing=False,
        )
        hidden_size = backbone.hidden_size
        print(f"[bc_slow] qwen hidden_size={hidden_size}")
        dataset = SessionDataset(jsonl_paths=existing, image_size=config.image_size, mode="pil")
        collate_fn = QwenSlowCollator(processor)
    else:
        tokenizer = _build_simple_tokenizer(existing)
        print(f"[bc_slow] tokenizer vocab: {tokenizer.vocab_size}")
        dataset = SessionDataset(jsonl_paths=existing, image_size=config.image_size, mode="tensor")
        collate_fn = SlowCollator(tokenizer)
        backbone = DummyBackbone(
            vocab_size=tokenizer.vocab_size + 1,
            hidden_size=hidden_size,
            image_feature=3,
        )

    print(f"[bc_slow] dataset size: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError("dataset 为空：请先 build_jsonl 生成 samples.jsonl")

    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=config.num_workers,
        drop_last=len(dataset) >= config.batch_size,
    )

    model = GameActorCritic(
        backbone=backbone,
        hidden_size=hidden_size,
        numeric_dim=NUM_CONTINUOUS,
        numeric_loss_weight=1.0,
    ).to(device)
    if use_fp16:
        for m in (model.numeric_head, model.value_head):
            m.to(torch.float32)

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[bc_slow] params: 总 {n_params/1e6:.1f}M  可训练 {n_train/1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr, weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    start_step = 0
    if config.resume_from is not None and Path(config.resume_from).exists():
        ckpt = load_checkpoint(config.resume_from, model, optimizer, map_location=device)
        start_step = ckpt["step"]
        print(f"[bc_slow] 从 {config.resume_from} 恢复，step={start_step}")

    loss_meter = AverageMeter()
    text_meter = AverageMeter()
    num_meter = AverageMeter()
    t0 = time.perf_counter()
    model.train()
    for step in range(start_step, config.num_steps):
        batch = next(iter_cycle(loader))
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                    numeric_labels=batch["numeric_action"],
                    attention_mask=batch.get("attention_mask"),
                    image_grid_thw=batch.get("image_grid_thw"),
                )
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                labels=batch["labels"],
                numeric_labels=batch["numeric_action"],
                attention_mask=batch.get("attention_mask"),
                image_grid_thw=batch.get("image_grid_thw"),
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                           config.grad_clip)
            optimizer.step()

        loss_meter.add(loss.item())
        text_meter.add(out.text_loss.item() if out.text_loss is not None else 0.0)
        num_meter.add(out.numeric_loss.item() if out.numeric_loss is not None else 0.0)

        if (step + 1) % config.log_interval == 0:
            print(
                f"[bc_slow] step {step+1:>5d}/{config.num_steps} "
                f"loss(recent)={loss_meter.recent_mean:.4f} "
                f"text={text_meter.recent_mean:.4f} num={num_meter.recent_mean:.4f} "
                f"elapsed={format_elapsed(time.perf_counter() - t0)}"
            )
        if (step + 1) % config.save_interval == 0:
            ckpt_path = config.output_dir / "actor_critic" / f"step_{step+1:06d}.pt"
            save_checkpoint(ckpt_path, step + 1, model, optimizer)
            print(f"[bc_slow] 保存 {ckpt_path}")

    final_path = config.output_dir / "actor_critic" / "final.pt"
    save_checkpoint(final_path, config.num_steps, model, optimizer,
                    extra={"hidden_size": hidden_size, "use_qwen": use_qwen})
    print(f"[bc_slow] 训练完成，最终 checkpoint: {final_path}")
    return model


def iter_cycle(loader: DataLoader) -> Iterator[dict]:
    """无限循环 loader（step 制训练）。"""
    while True:
        for batch in loader:
            yield batch


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, action="append", required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cpu")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--run-name", default="slow")
    p.add_argument("--resume-from", type=Path, default=None)
    p.add_argument("--use-qwen", action="store_true", help="用 Qwen3-VL-4B + LoRA")
    args = p.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = auto_timestamped_dir(Path("checkpoints"), args.run_name)
    args.output_dir = prepare_output_dir(args.output_dir)
    print(f"[bc_slow] output_dir: {args.output_dir}")

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
    train_actor_critic(cfg, use_qwen=args.use_qwen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
