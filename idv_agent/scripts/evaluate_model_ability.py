"""SGan（原 M29）模型能力离线评估：标签指标 + 图像消融 + 视觉依赖门禁。

三种模式（`--mode`）：

- `metrics`：标签指标（Q / decoding / phase / move / camera / bbox），并对比
  图像特征置零与跨 scenario-group 打乱的影响；
- `gate`：视觉依赖门禁，跑 normal / image_zero / image_shuffle / history_zero
  四种条件，外加 bbox 平移干预与 Q 假触发检查；
- `all`（默认）：两种都跑，写进同一份报告。

全程只读，不发送任何游戏输入。输出路径必须是新路径（不覆盖旧报告）。

用法：

    python -m idv_agent.scripts.evaluate_model_ability `
      --checkpoint checkpoints/<run>/m29.pt `
      --data data/derived/m29/<val>.jsonl `
      --model-path <Qwen3-VL-4B 权重目录> `
      --device cuda --output reports/m29_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from idv_agent.model.state_guided_action import M29_SCHEMA

CONDITIONS = ("normal", "image_zero", "image_shuffle", "history_zero")
MODES = ("metrics", "gate", "all")


def _load(checkpoint: str | Path, args: argparse.Namespace):
    from idv_agent.model.m29_checkpoint import load_m29_checkpoint
    from idv_agent.training.m29_features import load_m29_encoder

    core, manifest = load_m29_checkpoint(checkpoint, device=args.device)
    encoder = load_m29_encoder(args.model_path or manifest["base_model"], args.device)
    return core, manifest, encoder


def evaluate_metrics(core, dataset, encoder, checkpoint: str | Path,
                     args: argparse.Namespace) -> dict[str, Any]:
    """标签指标 + 图像特征置零 + 跨 scenario-group 图像打乱。"""
    from idv_agent.scripts.train_m29 import evaluate

    batch_size = max(1, int(args.batch_size))
    normal = evaluate(core, dataset, encoder, args.device, batch_size)

    def zero(paths):
        return torch.zeros((len(paths), core.feature_dim))

    report = {"schema": "m29.evaluation.metrics.v1", "checkpoint": str(checkpoint),
              "data": str(args.data), "files": dataset.files, "coverage": dataset.coverage(),
              "normal": normal,
              "image_feature_zero": evaluate(core, dataset, zero, args.device, batch_size),
              "acceptance": "diagnostic_only; no automatic deployment authorization"}
    groups = {}
    for sample, row in zip(dataset.samples, dataset.records):
        for path in sample["paths"]:
            groups[path] = row["scenario_group"]
    replacements = {}
    for path, group in groups.items():
        alternatives = sorted(p for p, g in groups.items() if g != group)
        if not alternatives:
            break
        replacements[path] = alternatives[sum(path.encode()) % len(alternatives)]
    if groups and len(replacements) == len(groups):
        report["cross_group_image_shuffle"] = evaluate(
            core, dataset, lambda paths: encoder([replacements[p] for p in paths]),
            args.device, batch_size)
    else:
        report["cross_group_image_shuffle"] = {
            "status": "unavailable", "reason": "requires independent scenario groups"}
    return report


def evaluate_gate(core, dataset, encoder, checkpoint: str | Path,
                  args: argparse.Namespace) -> dict[str, Any]:
    """视觉依赖门禁：三种退化条件 + bbox 干预 + Q 契约。"""
    from idv_agent.scripts.train_m29 import label_distributions
    from idv_agent.training.m29_dataset import collate_m29

    if len(dataset) < 2:
        raise ValueError("视觉依赖门禁至少需要两个样本")
    if args.max_samples > 0 and len(dataset) > args.max_samples:
        indices = torch.linspace(0, len(dataset) - 1, args.max_samples).long().tolist()
        samples = [dataset.samples[i] for i in indices]
    else:
        samples = list(dataset.samples)
    batch_size = max(1, int(args.batch_size))

    def run(condition: str):
        outputs, rows = [], []
        bbox_deltas = []
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start:start + batch_size]
            if condition == "history_zero":
                # SGan 的 history 就是视觉帧窗口：保留形状与时间戳，把每帧换成零特征。
                batch_samples = [dict(s) for s in batch_samples]
            if condition == "image_shuffle":
                batch_samples = [dict(s,
                                      paths=list(samples[(start + i + 1) % len(samples)]["paths"]),
                                      times=list(samples[(start + i + 1) % len(samples)]["times"]))
                                  for i, s in enumerate(batch_samples)]

            def encode(paths):
                if condition in {"image_zero", "history_zero"}:
                    from PIL import Image
                    return torch.cat([encoder.adapter.project_raw_visual_features(
                        encoder.adapter.encode_raw_frames([Image.new("RGB", (224, 224), (0, 0, 0))], micro_batch_size=1)
                    ).float().cpu() for _ in paths])
                return encoder(paths)

            inputs, targets, masks = collate_m29(batch_samples, encode, device=args.device)
            with torch.no_grad():
                out = core(**inputs)
                shifted = out.facts.bbox.clone()
                shifted[:, :2] = (shifted[:, :2] + 0.35).clamp(0, 1)
                shifted_facts = type(out.facts)(out.facts.visibility_logits, shifted,
                                                out.facts.prompt_bbox, out.facts.decoding_logits,
                                                out.facts.prompt_logits)
                shifted_beliefs = torch.cat((shifted_facts.probabilities(), out.decision.probabilities()), -1)
                shifted_nav = core.navigate_from_beliefs(shifted_beliefs, out.top_level_context)
                bbox_deltas.append({
                    "move": float((out.navigation.move_logits - shifted_nav.move_logits).abs().mean().cpu()),
                    "camera_dx": float((out.navigation.camera_dx_logits - shifted_nav.camera_dx_logits).abs().mean().cpu()),
                    "camera_dy": float((out.navigation.camera_dy_logits - shifted_nav.camera_dy_logits).abs().mean().cpu()),
                    "move_flip": float((out.navigation.move_logits.argmax(-1) != shifted_nav.move_logits.argmax(-1)).float().mean().cpu()),
                    "camera_dx_flip": float((out.navigation.camera_dx_logits.argmax(-1) != shifted_nav.camera_dx_logits.argmax(-1)).float().mean().cpu()),
                    "camera_dy_flip": float((out.navigation.camera_dy_logits.argmax(-1) != shifted_nav.camera_dy_logits.argmax(-1)).float().mean().cpu()),
                })
            pred = {"q": (out.facts.prompt_logits > 0).long(), "phase": out.decision.phase_logits.argmax(-1),
                    "move": out.navigation.move_logits.argmax(-1), "camera_dx": out.navigation.camera_dx_logits.argmax(-1),
                    "camera_dy": out.navigation.camera_dy_logits.argmax(-1)}
            rows.extend(zip(batch_samples, pred["q"].cpu().tolist(), pred["phase"].cpu().tolist(),
                             pred["move"].cpu().tolist(), pred["camera_dx"].cpu().tolist(), pred["camera_dy"].cpu().tolist()))
            outputs.append({k: v.detach().cpu() for k, v in {
                "q": out.facts.prompt_logits, "phase": out.decision.phase_logits,
                "move": out.navigation.move_logits, "camera_dx": out.navigation.camera_dx_logits,
                "camera_dy": out.navigation.camera_dy_logits}.items()})
        logits = {k: torch.cat([x[k] for x in outputs]) for k in outputs[0]}
        return logits, rows, bbox_deltas

    normal, normal_rows, normal_bbox = run("normal")
    degraded = {name: run(name)[0] for name in ("image_zero", "image_shuffle", "history_zero")}

    def flips(other):
        return {k: float((normal[k].argmax(-1) != other[k].argmax(-1)).float().mean())
                for k in ("phase", "move", "camera_dx", "camera_dy")}

    q_rows = [r for r in normal_rows if bool(r[0]["masks"]["q"])]
    q_tp = sum(int(q == 1 and bool(s["targets"]["q"])) for s, q, *_ in q_rows)
    q_fp = sum(int(q == 1 and not bool(s["targets"]["q"])) for s, q, *_ in q_rows)
    q_pos = sum(int(bool(s["targets"]["q"])) for s, *_ in q_rows)
    q_neg = len(q_rows) - q_pos
    q_prompt_ok = q_fp == 0
    bbox_summary = {k: sum(x[k] for x in normal_bbox) / max(1, len(normal_bbox))
                    for k in normal_bbox[0]} if normal_bbox else {}
    return {"schema": "m29.visual_dependency_gate.v1", "checkpoint": str(Path(checkpoint).resolve()),
            "checkpoint_schema": M29_SCHEMA, "data": str(args.data), "samples": len(samples),
            "distributions": label_distributions(dataset),
            "output_change": {k: flips(v) for k, v in degraded.items()},
            "q_contract": {"known_q_rows": len(q_rows), "tp": q_tp, "fp": q_fp,
                           "positive_recall": q_tp / q_pos if q_pos else None,
                           "false_trigger_rate": q_fp / q_neg if q_neg else None,
                           "precision": q_tp / (q_tp + q_fp) if q_tp + q_fp else None,
                           "normal_q_matches_label": q_prompt_ok,
                           "prompt_only_contract": "Q target is supervised only when interact_prompt is true"},
            "bbox_intervention": {"status": "measured", "shift": "+0.35 normalized x/y",
                                  "output_change": bbox_summary,
                                  "pass": any(bbox_summary.get(k, 0.0) > 0 for k in ("move", "camera_dx", "camera_dy"))},
            "history_zero": {"output_change": flips(degraded["history_zero"]),
                             "fixed_search_approach": flips(degraded["history_zero"])["phase"] < 0.95},
            "visual_dependency_gate_pass": all(max(flips(v).values()) >= 0.05 for v in degraded.values()) and q_prompt_ok}


def evaluate_checkpoint(checkpoint: str | Path, args: argparse.Namespace) -> dict[str, Any]:
    """按 ``args.mode`` 评估一个 SGan checkpoint。"""
    saved_header = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved_header.get("schema") != M29_SCHEMA:
        raise ValueError(
            f"evaluate_model_ability 只接受 {M29_SCHEMA} checkpoint，收到 {saved_header.get('schema')!r}；"
            "旧 m28/ACT checkpoint 已移除")
    from idv_agent.training.m29_dataset import M29Dataset

    dataset = M29Dataset(args.data)
    core, manifest, encoder = _load(checkpoint, args)
    result = {"schema": "m29.evaluation.v2", "mode": args.mode,
              "checkpoint": str(Path(checkpoint).resolve()),
              "manifest": {"task": manifest.get("task"), "steps": manifest.get("steps"),
                           "deployable": bool(manifest.get("deployable", False))},
              "data": str(args.data)}
    if args.mode in ("metrics", "all"):
        result["metrics"] = evaluate_metrics(core, dataset, encoder, checkpoint, args)
    if args.mode in ("gate", "all"):
        result["gate"] = evaluate_gate(core, dataset, encoder, checkpoint, args)
    return result


def _checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.checkpoint]
    if args.checkpoint_dir:
        directory = Path(args.checkpoint_dir)
        if not directory.is_dir():
            raise ValueError(f"checkpoint 目录不存在: {directory}")
        paths.extend(sorted(directory.glob("*.pt")))
    unique = list(dict.fromkeys(path.resolve() for path in paths))
    if not unique:
        raise ValueError("至少提供一个 --checkpoint 或 --checkpoint-dir")
    return unique


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="all",
                        help="metrics=标签指标；gate=视觉依赖门禁；all=两者（默认）")
    parser.add_argument("--data", required=True, help="M29 v6 JSONL 或逗号/分号分隔的 JSONL")
    parser.add_argument("--model-path", default=None,
                        help="Qwen3-VL 目录；省略时用 checkpoint manifest 的 base_model")
    parser.add_argument("--checkpoint", action="append", default=[], help="m29.pt，可重复")
    parser.add_argument("--checkpoint-dir", default="", help="批量评估目录下的 *.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=256,
                        help="gate 模式最多评估多少样本；0=全部")
    parser.add_argument("--output", type=Path, required=True,
                        help="报告输出路径；必须是不存在的路径")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size 必须为正数")
    if args.output.exists():
        raise ValueError("output exists; use a new report path")
    results = [evaluate_checkpoint(path, args) for path in _checkpoint_paths(args)]
    gate_runs = [result["gate"] for result in results if "gate" in result]
    report = {"schema": "m29.evaluation.v2", "mode": args.mode,
              "conditions": list(CONDITIONS), "checkpoints": results,
              "gate_pass": (all(run["visual_dependency_gate_pass"] for run in gate_runs)
                            if gate_runs else None)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "mode": args.mode, "gate_pass": report["gate_pass"],
                      "checkpoints": [result["checkpoint"] for result in results]},
                     ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())
