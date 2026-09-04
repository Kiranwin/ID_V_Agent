"""Validate M1_WK with deterministic semantic transforms of the training set.

This is an in-source consistency gate, not a generalization benchmark. It
loads the trained adapter, rewrites each training question with fixed Chinese
paraphrase wrappers, and checks answer overlap, termination, and repetition.
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from idv_agent.scripts.train_wk_lora import (
    _coerce_token_ids,
    build_semantic_eval_records,
    format_user_content,
    read_jsonl,
)

F1_THRESHOLD = 0.35
OVERALL_F1_PASS_RATE = 0.80
MODE_F1_PASS_RATE = 0.70
EOS_RATE = 0.90
REPEAT_RATE = 0.10


def normalize_text(text: str) -> str:
    punctuation = string.punctuation + "，。！？：；、“”‘’（）【】《》、·"
    return re.sub(r"\s+", "", str(text)).translate(str.maketrans("", "", punctuation))


def char_f1(prediction: str, reference: str) -> float:
    pred = Counter(normalize_text(prediction))
    ref = Counter(normalize_text(reference))
    if not pred or not ref:
        return 0.0
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def repetition_rate(text: str, n: int = 3) -> float:
    normalized = normalize_text(text)
    if len(normalized) < n:
        return 0.0
    grams = [normalized[i:i + n] for i in range(len(normalized) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def summarize_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"pass": False, "reason": "empty semantic evaluation"}
    overall_f1_pass = sum(float(row["char_f1"]) >= F1_THRESHOLD for row in rows) / len(rows)
    modes: dict[str, dict[str, float]] = {}
    for mode in sorted({str(row["mode"]) for row in rows}):
        subset = [row for row in rows if str(row["mode"]) == mode]
        modes[mode] = {
            "records": len(subset),
            "f1_pass_rate": sum(float(row["char_f1"]) >= F1_THRESHOLD for row in subset) / len(subset),
        }
    eos_rate = sum(bool(row["ended_with_eos"]) for row in rows) / len(rows)
    mean_repeat = sum(float(row["repeat_rate"]) for row in rows) / len(rows)
    passed = (
        eos_rate >= EOS_RATE
        and mean_repeat <= REPEAT_RATE
        and overall_f1_pass >= OVERALL_F1_PASS_RATE
        and len(modes) >= 3
        and all(item["f1_pass_rate"] >= MODE_F1_PASS_RATE for item in modes.values())
    )
    return {
        "pass": bool(passed),
        "records": len(rows),
        "eos_rate": round(eos_rate, 4),
        "mean_repeat_rate": round(mean_repeat, 4),
        "f1_threshold": F1_THRESHOLD,
        "f1_pass_rate": round(overall_f1_pass, 4),
        "modes": modes,
        "thresholds": {
            "eos_rate": EOS_RATE,
            "mean_repeat_rate_max": REPEAT_RATE,
            "overall_f1_pass_rate": OVERALL_F1_PASS_RATE,
            "mode_f1_pass_rate": MODE_F1_PASS_RATE,
        },
    }


def run_inference(*, base: str, adapter: str, data: str, variants_per_record: int = 2,
                  max_new_tokens: int = 96, device: str = "cuda", batch_size: int = 4) -> dict[str, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    records = read_jsonl(Path(data))
    eval_records = build_semantic_eval_records(records, variants_per_record)
    target_device = torch.device(device)
    dtype = torch.float16 if target_device.type == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(base)
    tokenizer = getattr(processor, "tokenizer", processor)
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=dtype, device_map=device if target_device.type == "cuda" else None,
    )
    if target_device.type != "cuda":
        model = model.to(target_device)
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    model_device = next(model.parameters()).device
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if batch_size < 1:
        raise ValueError("batch_size 必须 >= 1")
    rows: list[dict[str, Any]] = []
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    with torch.inference_mode():
        for start in range(0, len(eval_records), batch_size):
            batch_records = eval_records[start:start + batch_size]
            encoded = [_coerce_token_ids(processor.apply_chat_template(
                [{"role": "user", "content": format_user_content(record)}],
                tokenize=True, add_generation_prompt=True,
            )) for record in batch_records]
            max_len = max(len(ids) for ids in encoded)
            input_ids = torch.full((len(encoded), max_len), int(pad_id), dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            for index, ids in enumerate(encoded):
                offset = max_len - len(ids)
                input_ids[index, offset:] = torch.tensor(ids, dtype=torch.long)
                attention_mask[index, offset:] = 1
            inputs = {"input_ids": input_ids.to(model_device),
                      "attention_mask": attention_mask.to(model_device)}
            generated = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=eos_id, pad_token_id=eos_id,
            )
            for index, record in enumerate(batch_records):
                new_ids = generated[index, max_len:]
                ended_with_eos = eos_id is not None and eos_id in new_ids.tolist()
                prediction = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                rows.append({
                    "id": record["id"], "source_id": record["source_id"],
                    "mode": record["mode"], "topic": record.get("topic"),
                    "question": record["question"], "reference": record["answer"],
                    "prediction": prediction, "ended_with_eos": bool(ended_with_eos),
                    "char_f1": round(char_f1(prediction, record["answer"]), 4),
                    "repeat_rate": round(repetition_rate(prediction), 4),
                })
    return {
        "stage": "M1_WK",
        "base": str(Path(base).resolve()),
        "adapter": str(Path(adapter).resolve()),
        "data": str(Path(data).resolve()),
        "device": str(model_device), "dtype": str(next(model.parameters()).dtype),
        "variants_per_record": variants_per_record,
        "batch_size": batch_size,
        "source_records": len(records), "eval_records": len(eval_records),
        "rows": rows, "summary": summarize_gate(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants-per-record", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_inference(base=args.base, adapter=args.adapter, data=args.data,
                           variants_per_record=args.variants_per_record,
                           max_new_tokens=args.max_new_tokens, device=args.device,
                           batch_size=args.batch_size)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["summary"]["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
