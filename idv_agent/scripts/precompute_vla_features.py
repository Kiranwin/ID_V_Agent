"""Incrementally precompute frozen Qwen pooled frame features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from idv_agent.model.qwen_backbone_adapter import load_qwen3vl_backbone
from idv_agent.training.raw_feature_cache import RawFeatureCache, collect_unique_frame_paths


def verify_feature_cache(adapter, cache, groups, *, device="cuda", micro_batch_size=8) -> dict:
    checked = 0
    for _session_id, paths in groups.items():
        for start in range(0, len(paths), max(1, int(micro_batch_size))):
            chunk = paths[start:start + max(1, int(micro_batch_size))]
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            live = adapter.encode_raw_frames(images, micro_batch_size=len(chunk)).cpu()
            loaded = torch.stack([cache.get(path) for path in chunk])
            if not torch.equal(live, loaded):
                raise AssertionError(f"spatial cache 与实时 encode 不一致: {chunk[0]}")
            checked += len(chunk)
    return {"checked_frames": checked, "torch_equal": True}


def update_feature_cache(data_paths, model_path, output_dir, *, device="cuda", micro_batch_size=8,
                         spatial_k: int = 8, verify: bool = False):
    adapter, _ = load_qwen3vl_backbone(model_path, dtype=torch.float16 if str(device) != "cpu" else torch.float32,
                                       device_map=device, apply_lora=False)
    cache = RawFeatureCache(output_dir, spatial_k=spatial_k)
    groups = collect_unique_frame_paths(data_paths)
    encoded = 0
    for session_id, paths in groups.items():
        new_paths = [path for path in paths if path not in cache]
        if not new_paths:
            continue
        images = []
        for path in new_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        features = adapter.encode_raw_frames(images, micro_batch_size=micro_batch_size)
        encoded += cache.add_shard(session_id, new_paths, features)
    cache.save()
    result = {"unique_frames": cache.count, "new_frames_encoded": encoded,
              "shards": len(cache.index["shards"]), "raw_dim": cache.index["raw_dim"],
              "dtype": cache.index["dtype"], "pooling": cache.index["pooling"]}
    if verify:
        result.update(verify_feature_cache(adapter, cache, groups,
                                           device=device, micro_batch_size=micro_batch_size))
    (Path(output_dir) / "stats.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--spatial-k", type=int, default=8)
    parser.add_argument("--verify", action="store_true",
                        help="重编码所有帧并与磁盘 cache 做 torch.equal 校验")
    args = parser.parse_args(argv)
    print(json.dumps(update_feature_cache(args.data.split(","), args.model_path, args.output,
                                           device=args.device, micro_batch_size=args.micro_batch_size,
                                           spatial_k=args.spatial_k, verify=args.verify), indent=2))


if __name__ == "__main__":
    main()
