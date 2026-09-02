"""Train the WK (World Knowledge) text adapter with LoRA/SFT.

This stage is deliberately text-only.  It does not consume VG images or ACT
trajectories.  The output directory is an ``M1_WK`` adapter checkpoint which
later VG/ACT stages must load as their parent checkpoint.

Example (RTX 2080 Ti):
    python -m idv_agent.scripts.train_wk_lora \
      --data data/wk_vg/wk/wk_train.jsonl \
      --model-path C:/.../Qwen3-VL-4B-Instruct \
      --output-dir checkpoints/M1_WK --steps 100

Use ``--dry-run`` to validate and preview the formatted examples without
loading model weights.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader, Dataset

from idv_agent.model.qwen_backbone_adapter import default_qwen_lora_config
from idv_agent.training.checkpoint_manifest import build_manifest, sha256_file, write_manifest


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"WK 数据不存在: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} JSON 无效") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} 必须是 JSON 对象")
            for field in ("id", "question", "answer", "mode"):
                if not str(item.get(field, "")).strip():
                    raise ValueError(f"{path}:{line_no} 缺少字段 {field}")
            status = str(item.get("verification", {}).get("status", "reviewed"))
            if status not in {"reviewed", "verified"}:
                raise ValueError(f"{path}:{line_no} 未审核样本不能训练: {status}")
            rows.append(item)
    if not rows:
        raise ValueError(f"WK 数据为空: {path}")
    return rows


def format_example(record: dict[str, Any]) -> tuple[str, str]:
    """Return prompt and answer strings; mode is an explicit condition token."""
    mode = str(record["mode"]).strip()
    question = str(record["question"]).strip()
    answer = str(record["answer"]).strip()
    prompt = f"<mode:{mode}>\n问题：{question}\n答案："
    return prompt, answer


class WKTextDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], tokenizer: Any, max_length: int):
        self.items: list[dict[str, torch.Tensor | str]] = []
        if max_length < 32:
            raise ValueError("max_length 必须 >= 32")
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        for record in records:
            prompt, answer = format_example(record)
            prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            full_ids = tokenizer(prompt + answer, add_special_tokens=True,
                                 truncation=True, max_length=max_length)["input_ids"]
            # Ensure the answer remains supervised when a long sample is cut.
            if len(full_ids) <= len(prompt_ids):
                continue
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
            labels = labels[:len(full_ids)]
            self.items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
                "id": str(record["id"]),
            })
        if not self.items:
            raise ValueError("WK 样本在截断后没有可监督的答案 token")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self.items[index]


def collate_wk(samples: list[dict[str, torch.Tensor | str]], pad_id: int) -> dict[str, Any]:
    max_len = max(int(sample["input_ids"].numel()) for sample in samples)
    batch_size = len(samples)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention = torch.zeros((batch_size, max_len), dtype=torch.long)
    ids = []
    for row, sample in enumerate(samples):
        n = int(sample["input_ids"].numel())
        input_ids[row, :n] = sample["input_ids"]
        labels[row, :n] = sample["labels"]
        attention[row, :n] = 1
        ids.append(sample["id"])
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": attention, "ids": ids}


def _load_text_model(model_path: str | Path, dtype: torch.dtype, device: torch.device,
                     gradient_checkpointing: bool = False):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import get_peft_model

    processor = AutoProcessor.from_pretrained(str(model_path))
    tokenizer = getattr(processor, "tokenizer", processor)
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_path), dtype=dtype, device_map=None,
    ).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, default_qwen_lora_config())
    return model, processor, tokenizer


def _split_records(records: list[dict[str, Any]], val_fraction: float, seed: int):
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction 必须在 [0,1)")
    if val_fraction == 0:
        return records, []
    rng = random.Random(seed)
    by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_topic.setdefault(str(row.get("topic", "unknown")), []).append(row)
    train, val = [], []
    for group in by_topic.values():
        rng.shuffle(group)
        n_val = min(max(1, round(len(group) * val_fraction)), len(group) - 1)
        val.extend(group[:n_val]); train.extend(group[n_val:])
    return train, val


def _write_manifest(output_dir: Path, *, model_path: str | Path, data: Path,
                    train_count: int, val_count: int, steps: int, parent: str | None = None,
                    args: argparse.Namespace | None = None):
    training = {
        "method": "causal_sft",
        "steps": steps,
        "learning_rate": getattr(args, "lr", None),
        "batch_size": getattr(args, "batch_size", None),
        "max_length": getattr(args, "max_length", None),
        "seed": getattr(args, "seed", None),
        "gradient_checkpointing": bool(getattr(args, "gradient_checkpointing", False)),
    }
    manifest = build_manifest(
        stage="M1_WK", parent=parent, base_model=model_path,
        adapters=["lora_wk"], frozen=["vision_tower"],
        trainable=["language_lora:q_proj", "language_lora:k_proj",
                   "language_lora:v_proj", "language_lora:o_proj"],
        data={"wk": str(data).replace("\\", "/"), "wk_sha256": sha256_file(data)},
        counts={"train": train_count, "val": val_count},
        precision="fp16+GradScaler" if torch.cuda.is_available() else "fp32",
        training=training, schema_versions=["wk.sft.v1"],
        artifacts={"adapter": "adapter", "processor": "processor", "metrics": "metrics.json"},
    )
    return write_manifest(output_dir / "manifest.json", manifest)


def train(args: argparse.Namespace) -> dict[str, Any]:
    records = read_jsonl(Path(args.data))
    val_records = read_jsonl(Path(args.val_data)) if args.val_data else []
    train_records, auto_val = _split_records(records, args.val_fraction, args.seed)
    if not val_records:
        val_records = auto_val
    train_ids = {str(row["id"]) for row in train_records}
    overlap = train_ids & {str(row["id"]) for row in val_records}
    if overlap:
        raise ValueError(f"WK train/val ID 重叠: {sorted(overlap)[:5]}")
    if args.dry_run:
        preview = [format_example(row) for row in train_records[:3]]
        return {"stage": "M1_WK", "train_records": len(train_records),
                "val_records": len(val_records), "preview": preview}

    device = torch.device(args.device)
    if not args.model_path:
        raise ValueError("真实训练必须提供 --model-path；仅预览可使用 --dry-run")
    use_amp = device.type == "cuda"
    dtype = torch.float16 if use_amp else torch.float32
    model, processor, tokenizer = _load_text_model(args.model_path, dtype, device,
                                                    args.gradient_checkpointing)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    train_ds = WKTextDataset(train_records, tokenizer, args.max_length)
    val_ds = WKTextDataset(val_records, tokenizer, args.max_length) if val_records else None
    loader = DataLoader(train_ds, batch_size=max(1, args.batch_size), shuffle=True,
                        collate_fn=lambda x: collate_wk(x, pad_id))
    val_loader = (DataLoader(val_ds, batch_size=max(1, args.batch_size), shuffle=False,
                             collate_fn=lambda x: collate_wk(x, pad_id)) if val_ds else None)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[float] = []
    iterator = iter(loader)
    started = time.perf_counter()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader); batch = next(iterator)
        inputs = {key: batch[key].to(device) for key in ("input_ids", "labels", "attention_mask")}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            output = model(**inputs)
            loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step {step}: loss 非有限值")
        if use_amp:
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0); scaler.step(optimizer); scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); optimizer.step()
        history.append(float(loss.detach().cpu()))

    val_loss = None
    if val_loader:
        model.eval(); values = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {key: batch[key].to(device) for key in ("input_ids", "labels", "attention_mask")}
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    values.append(float(model(**inputs).loss.detach().cpu()))
        val_loss = sum(values) / len(values) if values else None
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "adapter"; model.save_pretrained(adapter_dir)
    processor.save_pretrained(output_dir / "processor")
    manifest = _write_manifest(output_dir, model_path=args.model_path, data=Path(args.data),
                               train_count=len(train_ds), val_count=len(val_ds) if val_ds else 0,
                               steps=args.steps, args=args)
    result = {"stage": "M1_WK", "train_records": len(train_ds),
              "val_records": len(val_ds) if val_ds else 0, "steps": args.steps,
              "initial_loss": history[0] if history else None,
              "final_loss": history[-1] if history else None, "val_loss": val_loss,
              "checkpoint": str(output_dir.resolve()),
              "elapsed_sec": time.perf_counter() - started, "manifest": manifest}
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--output-dir", default="checkpoints/M1_WK")
    parser.add_argument("--val-data", default=None)
    parser.add_argument("--val-fraction", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)
    try:
        result = train(args)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
