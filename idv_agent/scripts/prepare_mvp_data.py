"""Prepare the MVP ACT dataset from recorded v4 sessions.

Only the two skills needed by the MVP are retained: ``travel`` (finding a
machine) and ``decipher`` (approaching/starting decoding).  Source JSONL files
are never modified.  Sessions are split as whole units to prevent frame
leakage; a deterministic greedy split keeps both intents on each side.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

KEEP = {"travel", "decipher"}
V4_CHUNK_FILENAMES = ("vla_chunks_v4.jsonl", "train_vla_chunks_v4.jsonl")


def _find_v4_chunks(session: Path) -> Path | None:
    """Return the canonical v4 file, with a transitional filename fallback.

    Some already-generated sessions used the early ``train_vla_chunks_v4``
    name.  Raw-session compatibility is intentionally not broadened; this
    narrow alias only prevents a valid, current-schema chunk file from being
    silently omitted by the MVP collector.
    """
    for filename in V4_CHUNK_FILENAMES:
        path = session / filename
        if path.is_file():
            return path
    return None


def _read(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from exc
                if isinstance(value, dict):
                    yield value


def _session_records(path: Path, travel_scope: str) -> list[dict]:
    rows = list(_read(path))
    if travel_scope == "pre_decipher":
        starts = [int(r.get("anchor_frame", 0)) for r in rows
                  if r.get("slow_label", {}).get("intent") == "decipher"]
        first = min(starts) if starts else None
    else:
        first = None
    out = []
    for row in rows:
        intent = row.get("slow_label", {}).get("intent", "")
        if intent not in KEEP:
            continue
        if first is not None and intent == "travel" and int(row.get("anchor_frame", 0)) >= first:
            continue
        # Keep protocol-relative paths and record the original session root;
        # the dataset loader uses this hint when the filtered JSONL is moved.
        row = json.loads(json.dumps(row))
        row["source_root"] = str(path.parent.resolve())
        out.append(row)
    return out


def _split(sessions: list[tuple[str, list[dict]]], val_ratio: float, seed: int,
           explicit_val: set[str] | None) -> tuple[list, list]:
    if explicit_val:
        val = [x for x in sessions if x[0] in explicit_val]
        train = [x for x in sessions if x[0] not in explicit_val]
    else:
        rng = random.Random(seed)
        shuffled = sessions[:]
        rng.shuffle(shuffled)
        # Keep at least one whole session in train.  For tiny datasets the
        # intent-coverage seed below can otherwise consume every session.
        target = min(len(shuffled) - 1, max(1, round(len(shuffled) * val_ratio)))
        # Start with one session for each intent, then fill to the target.
        val: list[tuple[str, list[dict]]] = []
        for intent in sorted(KEEP):
            if len(val) >= target:
                break
            candidate = next((item for item in shuffled if any(r["slow_label"].get("intent") == intent for r in item[1])
                              and item not in val), None)
            if candidate:
                val.append(candidate)
        for item in shuffled:
            if len(val) >= target:
                break
            if item not in val:
                val.append(item)
        train = [item for item in shuffled if item not in val]
        # Keep both intents in train as well; swap a suitable session if needed.
        for intent in KEEP - {r["slow_label"]["intent"] for _, rs in train for r in rs}:
            candidate = next((item for item in val if any(r["slow_label"].get("intent") == intent for r in item[1])
                              and any(r["slow_label"].get("intent") != intent for r in item[1])), None)
            if candidate:
                val.remove(candidate); train.append(candidate)
    return sorted(train), sorted(val)


def _stats(items: list[tuple[str, list[dict]]]) -> dict:
    intents, moves, cams, buttons = Counter(), Counter(), Counter(), Counter()
    n = 0
    for _, rows in items:
        for row in rows:
            n += 1
            intent = row.get("slow_label", {}).get("intent", "")
            intents[intent] += 1
            for action in row.get("action_chunk", []):
                moves[str(action.get("move_dir", 0))] += 1
                cams[f"dx:{action.get('camera_dx', 0)}"] += 1
                cams[f"dy:{action.get('camera_dy', 0)}"] += 1
                for i, value in enumerate(action.get("buttons", [])):
                    if value:
                        buttons[str(i)] += 1
    return {"chunks": n, "intents": dict(intents), "move_dir": dict(moves),
            "camera_buckets": dict(cams), "button_nonzero": dict(buttons),
            "sessions": [name for name, _ in items]}


def prepare(root: Path, output: Path, *, val_ratio: float, seed: int,
            val_sessions: set[str] | None, travel_scope: str,
            max_session_chunks: int | None = None) -> dict:
    sessions = []
    for session_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name) \
            if root.is_dir() else []:
        path = _find_v4_chunks(session_dir)
        if path is None:
            continue
        rows = _session_records(path, travel_scope)
        if max_session_chunks is not None and len(rows) > max_session_chunks:
            continue
        if rows:
            sessions.append((path.parent.name, rows))
    train, val = _split(sessions, val_ratio, seed, val_sessions)
    if not train or not val:
        raise ValueError("train/val session split is empty")
    for name, rows in train + val:
        target = output / ("train" if (name, rows) in train else "val") / f"{name}.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    train_stats, val_stats = _stats(train), _stats(val)
    missing = {side: sorted(KEEP - set(stats["intents"])) for side, stats in (("train", train_stats), ("val", val_stats))}
    report = {"schema": "mvp.v1", "keep_intents": sorted(KEEP), "travel_scope": travel_scope,
              "max_session_chunks": max_session_chunks,
              "seed": seed, "train": train_stats, "val": val_stats,
              "missing_intents": missing, "session_overlap": sorted(set(train_stats["sessions"]) & set(val_stats["sessions"])),
              "gate_pass": not any(missing.values()) and not (set(train_stats["sessions"]) & set(val_stats["sessions"]))}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Candidate starts are deterministic and reproducible.  They are not
    # claimed as successful sandbox points until the same start state is
    # recreated in the game; this file is the hand-off checklist for that
    # recording/evaluation step.
    candidates = []
    for index, (name, rows) in enumerate(val):
        travel = [r for r in rows if r.get("slow_label", {}).get("intent") == "travel"]
        if not travel:
            continue
        row = min(travel, key=lambda r: int(r.get("anchor_frame", 0)))
        frame = row.get("observations", {}).get("frames", [{}])[-1]
        candidates.append({"id": f"mvp_val_{index:02d}", "session": name,
                           "anchor_frame": row.get("anchor_frame"),
                           "image": str((Path(row["source_root"]) / frame.get("path", "")).resolve()),
                           "target": "cipher_machine", "expected_flow": ["travel", "decipher"],
                           "status": "candidate_recreate_in_sandbox"})
    (output / "eval_start_points.json").write_text(
        json.dumps({"schema": "mvp.eval_start_points.v1", "points": candidates},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sessions-root", type=Path, default=Path("data/new_vla_raw_sessions"))
    p.add_argument("--output", type=Path, default=Path("data/mvp_vla"))
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260902)
    p.add_argument("--val-sessions", default="", help="comma separated session ids")
    p.add_argument("--travel-scope", choices=("all", "pre_decipher"), default="pre_decipher")
    p.add_argument("--max-session-chunks", type=int, default=500,
                   help="只纳入筛选后不超过该数量的短 session；0 表示不限制")
    args = p.parse_args(argv)
    report = prepare(args.sessions_root, args.output, val_ratio=args.val_ratio, seed=args.seed,
                     val_sessions={s.strip() for s in args.val_sessions.split(",") if s.strip()} or None,
                     travel_scope=args.travel_scope,
                     max_session_chunks=(args.max_session_chunks or None))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
