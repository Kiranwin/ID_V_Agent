"""M29 consumes v6 reviewed navigation or independent prompt-Q exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from idv_agent.model.state_guided_action import PHASES, PATHS, STEERING, VISIBILITY, CAMERA_COMMANDS
from idv_agent.vla.prompt_q import Q_CONTRACT


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_feedback(observed_ns: int, sent_ns: int | None) -> list[float]:
    """Only a known past, actually submitted Q edge is feedback (not success)."""
    if sent_ns is None:
        return [0.0, 0.0]
    if type(sent_ns) is not int or not 0 <= sent_ns <= observed_ns:
        raise ValueError("Q feedback must be a past execution timestamp")
    return [1.0, min((observed_ns-sent_ns)/1e9, 5.0)/5.0]


def _is_deprecated_record(row: dict) -> bool:
    """Exclude endpoints explicitly marked 废弃 in the review provenance."""
    review = row.get("audit_only", {}).get("review", {})
    action = row.get("targets", {}).get("action", {})
    text = " ".join(str(value or "") for value in (
        action.get("reason"), review.get("reason"), review.get("notes")))
    return "废弃" in text


class M29Dataset(Dataset):
    def __init__(self, data: str | Path, *, expected_split: str | None = None):
        paths = [Path(item.strip()) for item in str(data).split(",") if item.strip()]
        self.records, self.samples, self.files = [], [], []
        self.ids, self.sessions, self.groups, self.splits = set(), set(), set(), set()
        positions = {}
        verified = set()
        for path in paths:
            if not path.is_file():
                raise ValueError(f"M29 expects explicit JSONL export files: {path}")
            self.files.append({"path": str(path.resolve()), "sha256": _hash(path)})
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if _is_deprecated_record(row):
                    continue
                if row.get("schema") not in {"idv.mvp_training.v6", "idv.prompt_q_training.v2"}:
                    raise ValueError("M29 requires v6 or prompt-Q exports; legacy v5 is incompatible")
                if not row.get("id") or not row.get("scenario_group"):
                    raise ValueError("missing ID or scenario_group")
                if row.get("split") not in {"train", "val", "test"} or expected_split and row["split"] != expected_split:
                    raise ValueError("dataset split mismatch")
                self.sessions.add(row.get("session_id", row["id"].rsplit(":", 1)[0]))
                self.groups.add(row["scenario_group"])
                self.splits.add(row["split"])
                inputs = row["model_input"]
                q_only = row["schema"] == "idv.prompt_q_training.v2"
                contract = row.get("q_contract") if q_only else row.get("protocol", {}).get("q_contract")
                if contract != Q_CONTRACT:
                    raise ValueError("prompt-Q contract mismatch")
                observed = inputs["observed_ns"]
                if type(observed) is not int or observed < 0:
                    raise ValueError("observed_ns must be nonnegative integer")
                if q_only:
                    image_paths, times = [Path(inputs["image_path"])], [observed]
                    source_hashes = {str(image_paths[0]): row["audit_only"]["image_sha256"]}
                else:
                    from idv_agent.scripts.prepare_mvp_v6 import TRAIN_PROTOCOL
                    if row["protocol"] != TRAIN_PROTOCOL or inputs.get("task") != "decipher":
                        raise ValueError("M29 full sample temporal/task protocol mismatch")
                    frame_ids = inputs["history_frames"]
                    times = inputs["history_timestamps_ns"]
                    if not 1 <= len(frame_ids) <= 8 or len(times) != len(frame_ids) or frame_ids != sorted(set(frame_ids)):
                        raise ValueError("invalid visual history IDs")
                    raw_root = Path(inputs["raw_root"])
                    image_paths = [raw_root / "frames" / f"{i:08d}.jpg" for i in frame_ids]
                    source_hashes = {str(p): row["audit_only"]["raw_sha256"][f"frames/{i:08d}.jpg"]
                                     for p, i in zip(image_paths, frame_ids)}
                if times != sorted(set(times)) or times[-1] != observed or any(type(t) is not int or t < 0 for t in times):
                    raise ValueError("visual history timestamps must strictly increase to observed time")
                for p in image_paths:
                    key = (str(p.resolve()), source_hashes[str(p)])
                    if key not in verified:
                        if not p.is_file() or _hash(p) != source_hashes[str(p)]:
                            raise ValueError(f"image missing or changed: {p}")
                        verified.add(key)
                targets = {key: 0 for key in ("q", "decoding", "visibility", "phase", "steering", "path", "move", "camera_dx", "camera_dy")}
                targets["bbox"] = [0., 0., 0., 0.]
                targets["prompt_bbox"] = [0., 0., 0., 0.]
                masks = {key: False for key in ("q", "decoding", "visibility", "phase", "steering", "path", "bbox", "prompt_bbox", "navigation")}
                q = row["targets"]["q"] if q_only else row["targets"]["action"]["q"]
                prompt = row["targets"]["interact_prompt"] if q_only else row["targets"]["facts"]["interact_prompt"]
                if type(row["masks"].get("q")) is not bool or q != prompt:
                    raise ValueError("Q mask or target disagrees with prompt annotation")
                masks["q"] = row["masks"]["q"]
                if masks["q"]:
                    if type(q) is not bool:
                        raise ValueError("known Q target must be boolean")
                    targets["q"] = float(q)
                prompt_box = row["targets"].get("prompt_bbox") if q_only else row["targets"]["facts"].get("prompt_bbox")
                prompt_box_mask = row["masks"].get("prompt_bbox")
                if type(prompt_box_mask) is not bool:
                    raise ValueError("missing/invalid mask: prompt_bbox")
                masks["prompt_bbox"] = prompt_box_mask
                if prompt_box_mask:
                    if (not isinstance(prompt_box, list) or len(prompt_box) != 4 or
                            not all(type(value) in (int, float) and 0 <= value <= 1 for value in prompt_box) or
                            not (prompt_box[0] < prompt_box[2] and prompt_box[1] < prompt_box[3])):
                        raise ValueError("invalid prompt box")
                    targets["prompt_bbox"] = prompt_box
                feedback = execution_feedback(observed, inputs.get("last_executed_q_ns"))
                if not q_only:
                    facts, decision, action = (row["targets"][key] for key in ("facts", "decision", "action"))
                    for key in ("navigation", "phase", "visibility", "bbox", "decoding", "steering", "path"):
                        if type(row["masks"].get(key)) is not bool:
                            raise ValueError(f"missing/invalid mask: {key}")
                        masks[key] = row["masks"][key]
                    if masks["visibility"]:
                        targets["visibility"] = VISIBILITY.index(facts["cipher_visibility"])
                    if masks["decoding"]:
                        if type(facts["decoding"]) is not bool:
                            raise ValueError("decoding target must be boolean when known")
                        targets["decoding"] = float(facts["decoding"])
                    if masks["bbox"]:
                        box = facts["target_bbox"]
                        if not isinstance(box, list) or len(box) != 4 or not all(type(v) in (int, float) and 0 <= v <= 1 for v in box) or not (box[0] < box[2] and box[1] < box[3]):
                            raise ValueError("invalid target box")
                        targets["bbox"] = box
                    if masks["phase"]:
                        targets["phase"] = PHASES.index(decision["phase"])
                        if decision["phase"] == "maintain_decode" and facts["decoding"] is not True:
                            raise ValueError("maintain_decode requires current decoding=true")
                    if masks["steering"]:
                        targets["steering"] = STEERING.index(decision["steering"])
                    if masks["path"]:
                        targets["path"] = PATHS.index(facts["path"])
                    if masks["navigation"]:
                        if len(times) < 3 or action["source"] not in {"accept_replay", "correction"}:
                            raise ValueError("navigation requires reviewed complete action and visual history")
                        if type(action["move"]) is not int or not 0 <= action["move"] <= 8:
                            raise ValueError("invalid movement target")
                        if (not isinstance(action["camera_command"], list) or len(action["camera_command"]) != 2
                                or any(type(value) is not int for value in action["camera_command"])):
                            raise ValueError("camera command must have two integer codebook values")
                        targets["move"] = action["move"]
                        targets["camera_dx"], targets["camera_dy"] = [CAMERA_COMMANDS.index(value) for value in action["camera_command"]]
                if not any(masks.values()):
                    continue
                sample = {"id": row["id"], "paths": [str(p.resolve()) for p in image_paths],
                          "times": [(t-observed)/1e9 for t in times],
                          "feedback": feedback, "targets": targets, "masks": masks,
                          "q_only": q_only}
                previous = positions.get(row["id"])
                if previous is not None:
                    old = self.samples[previous]
                    if old["q_only"] == q_only:
                        raise ValueError("duplicate ID within the same supervision kind")
                    if (old["targets"]["q"] != sample["targets"]["q"] or
                            old["masks"]["q"] != sample["masks"]["q"] or
                            old["paths"][-1] != sample["paths"][-1]):
                        raise ValueError("full and Q-only records disagree at duplicate endpoint")
                    if q_only:
                        continue
                    self.records[previous] = row
                    self.samples[previous] = sample
                    continue
                positions[row["id"]] = len(self.samples)
                self.ids.add(row["id"])
                self.records.append(row)
                self.samples.append(sample)
        if not self.samples:
            raise ValueError("M29 dataset has no supervised samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def coverage(self):
        return {key: sum(sample["masks"][key] for sample in self.samples) for key in self.samples[0]["masks"]}


def collate_m29(samples, encode, *, device):
    """encode(paths)->detached pure visual features; both runtime and train use identical pooling."""
    unique = list(dict.fromkeys(path for sample in samples for path in sample["paths"]))
    encoded = encode(unique).to(device=device, dtype=torch.float32)
    lookup = {path: encoded[i] for i, path in enumerate(unique)}
    length = max(len(sample["paths"]) for sample in samples)
    features = encoded.new_zeros((len(samples), length, encoded.shape[-1]))
    valid = torch.zeros((len(samples), length), device=device, dtype=torch.bool)
    times = encoded.new_zeros((len(samples), length))
    for i, sample in enumerate(samples):
        n = len(sample["paths"])
        features[i, :n] = torch.stack([lookup[path] for path in sample["paths"]])
        valid[i, :n] = True
        times[i, :n] = torch.tensor(sample["times"], device=device)
    inputs = {"features": features, "valid_mask": valid, "relative_times_s": times,
              "execution_feedback": torch.tensor([s["feedback"] for s in samples], device=device)}
    targets = {key: torch.tensor([s["targets"][key] for s in samples], device=device) for key in samples[0]["targets"]}
    masks = {key: torch.tensor([s["masks"][key] for s in samples], device=device, dtype=torch.bool) for key in samples[0]["masks"]}
    return inputs, targets, masks
