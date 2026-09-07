"""Audit whether same-frame visual grounding labels explain ACT camera replay.

Read-only: this command joins canonical v5 chunks with the completed grounding
sidecar by ``episode_id + observation_end_frame`` and reports camera bucket
distributions conditioned on target side, prompt, reachability and visibility.
It does not load a model or write data.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from idv_agent.scripts.train_vla import _dataset_paths
from idv_agent.training.vla_dataset import GroundingAnnotationIndex, VLASequenceDataset
from idv_agent.vla.action_chunk import CAMERA_BUCKETS


def _counts() -> dict[str, Counter[str]]:
    return {"dx": Counter(), "dy": Counter()}


def _distribution_summary(counter: Counter[str]) -> dict[str, object]:
    """Summarize label ambiguity without treating a majority as causality."""
    total = sum(counter.values())
    if not total:
        return {"count": 0, "entropy_bits": None, "dominant_bucket": None,
                "dominant_rate": None, "counts": {}}
    probabilities = [value / total for value in counter.values() if value]
    entropy = -sum(value * math.log2(value) for value in probabilities)
    bucket, count = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0]
    return {"count": total, "entropy_bits": entropy, "dominant_bucket": bucket,
            "dominant_rate": count / total, "counts": dict(counter)}


def audit(data: str | list[str], annotations: str | Path) -> dict:
    dataset = VLASequenceDataset(_dataset_paths(data), verify_images=False,
                                 grounding_annotations=annotations)
    grouped: dict[str, dict[str, Counter[str]]] = defaultdict(_counts)
    # A current observation can only be directly causal for its first action
    # step.  Preserve the existing aggregate but expose every horizon so the
    # data audit can measure how rapidly that relationship decays.
    horizon_grouped: dict[int, dict[str, dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(_counts))
    total = 0
    for index in range(len(dataset)):
        item = dataset[index]
        if not bool(item["grounding_mask"].item()):
            continue
        total += 1
        present = int(item["grounding_present_target"].item())
        prompt = int(item["grounding_prompt_target"].item())
        reachable = int(item["grounding_reachable_target"].item())
        side_index = int(item["grounding_side_target"].item())
        side = ("none", "left", "center", "right")[side_index]
        last_history_dx_index = int(item["history_actions"][-9 + 1].item())
        last_history_dy_index = int(item["history_actions"][-9 + 2].item())
        last_history_dx = CAMERA_BUCKETS[last_history_dx_index]
        last_history_dy = CAMERA_BUCKETS[last_history_dy_index]
        groups = ("all", f"present={present}", f"prompt={prompt}",
                  f"reachable={reachable}", f"side={side}",
                  f"previous_dx={last_history_dx}", f"previous_dy={last_history_dy}",
                  f"side={side}|previous_dx={last_history_dx}")
        for group in groups:
            for horizon, bucket in enumerate(item["camera_dx_target"].tolist()):
                grouped[group]["dx"][str(CAMERA_BUCKETS[int(bucket)])] += 1
                horizon_grouped[horizon][group]["dx"][str(CAMERA_BUCKETS[int(bucket)])] += 1
            for horizon, bucket in enumerate(item["camera_dy_target"].tolist()):
                grouped[group]["dy"][str(CAMERA_BUCKETS[int(bucket)])] += 1
                horizon_grouped[horizon][group]["dy"][str(CAMERA_BUCKETS[int(bucket)])] += 1
    if not total:
        raise ValueError("没有 canonical chunk 与 grounding 标注在 observation_end_frame 对齐")
    return {
        "schema": "act.grounding_camera_alignment.v1",
        "aligned_chunks": total,
        "groups": {name: {axis: dict(counter) for axis, counter in axes.items()}
                   for name, axes in sorted(grouped.items())},
        "horizons": {
            str(horizon): {"groups": {
                name: {axis: dict(counter) for axis, counter in axes.items()}
                for name, axes in sorted(groups.items())}}
            for horizon, groups in sorted(horizon_grouped.items())
        },
        # Only h=0 is causally consumed by m25/m26.  These summaries quantify
        # whether the observation labels make that executed replay bucket
        # determinate; they do not claim a condition is itself sufficient.
        "causal_horizon_summary": {
            name: {axis: _distribution_summary(counter) for axis, counter in axes.items()}
            for name, axes in sorted(horizon_grouped[0].items())
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="canonical v5 JSONL，支持逗号/分号分隔")
    parser.add_argument("--grounding-annotations", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = audit(args.data, args.grounding_annotations)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
