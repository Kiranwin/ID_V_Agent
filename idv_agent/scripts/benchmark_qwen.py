"""Benchmark the Qwen single-frame path against the 15 Hz deadline.

The measured path is intentionally limited to ``encode_frame`` + temporal
encoder + fast head, matching docs/15.  With no model path this runs a small
CPU fake Qwen visual tower, which validates plumbing and reports a baseline;
real acceptance should be run with the local Qwen checkpoint on the 2080 Ti.
"""

from __future__ import annotations

import argparse
import statistics
import time
from types import SimpleNamespace

import torch
from torch import nn
from PIL import Image

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.qwen_backbone_adapter import Qwen3VLBackboneAdapter, load_qwen3vl_backbone
from idv_agent.model.temporal import TaskConditionCache


class _FakeProcessor:
    def __call__(self, *, text, return_tensors="pt", add_special_tokens=True):
        return {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)}

    def image_processor(self, *, images, return_tensors="pt"):
        return {"pixel_values": images if isinstance(images, torch.Tensor)
                else torch.zeros(1, 3, 224, 224),
                "image_grid_thw": torch.tensor([[1, 16, 16]], dtype=torch.long)}


class _FakeVisual(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.proj = nn.Linear(3, width)

    def forward(self, pixel_values=None, image_grid_thw=None):
        x = pixel_values.mean(dim=(2, 3))
        return self.proj(x).unsqueeze(1).expand(-1, 16, -1)


class _FakeQwen(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.visual = _FakeVisual(width)
        self.emb = nn.Embedding(32, width)
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=width, vocab_size=32))

    def get_input_embeddings(self):
        return self.emb


def _stats(values):
    values = sorted(values)
    return {"n": len(values), "mean_ms": statistics.mean(values),
            "p50_ms": values[len(values) // 2],
            "p95_ms": values[max(0, int(len(values) * .95) - 1)],
            "hz_capacity": 1000.0 / max(values)}


def run(adapter, cache, *, seconds: float, device: str):
    width = adapter.hidden_size
    try:
        model_dtype = next(adapter.model.parameters()).dtype
    except StopIteration:
        model_dtype = torch.float32
    core = SharedFastSlowVLA(frame_feature_dim=width, temporal_dim=min(256, width)).to(
        device=device, dtype=model_dtype).eval()
    # Pass a PIL image so the real Qwen processor supplies image_grid_thw;
    # fake mode accepts it as well.
    image = Image.new("RGB", (224, 224), color=(0, 0, 0))
    # Prime lazy projections and kernels before collecting samples.
    with torch.inference_mode():
        for _ in range(3):
            feat = adapter.encode_frame(image, cache)
            out = core.temporal(feat.view(1, 1, -1))
            core.fast_head(out)
        samples = []
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline or len(samples) < 10:
            t0 = time.perf_counter()
            feat = adapter.encode_frame(image, cache)
            temporal = core.temporal(feat.view(1, 1, -1))
            core.fast_head(temporal)
            samples.append((time.perf_counter() - t0) * 1000.0)
    return _stats(samples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", help="本地 Qwen3-VL checkpoint；省略则运行 fake CPU 基线")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--target-hz", type=float, default=15.0)
    args = parser.parse_args(argv)
    if args.model_path:
        adapter, _ = load_qwen3vl_backbone(args.model_path, device_map=args.device,
                                            apply_lora=False)
        device = str(adapter.device)
        cache = adapter.encode_task_once("找到密码机并破译", "standard", task_id="benchmark")
    else:
        adapter = Qwen3VLBackboneAdapter(_FakeQwen(), processor=_FakeProcessor()).to(
            args.device, dtype=torch.float32)
        device = args.device
        cache = adapter.encode_task_once("benchmark", "standard", task_id="benchmark")
    result = run(adapter, cache, seconds=max(0.1, args.seconds), device=device)
    budget_ms = 1000.0 / args.target_hz
    result["target_hz"] = args.target_hz
    result["budget_ms"] = budget_ms
    result["meets_15hz"] = result["p95_ms"] <= budget_ms
    print("=== Qwen encode_frame latency benchmark ===")
    print(f"samples={result['n']} mean={result['mean_ms']:.2f}ms "
          f"p50={result['p50_ms']:.2f}ms p95={result['p95_ms']:.2f}ms "
          f"capacity={result['hz_capacity']:.2f}Hz target={args.target_hz:.1f}Hz "
          f"budget={budget_ms:.2f}ms meets={result['meets_15hz']}")
    return 0 if result["meets_15hz"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
