"""Validate a WK/VG/ACT checkpoint manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idv_agent.training.checkpoint_manifest import load_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="checkpoint 目录或 manifest.json")
    parser.add_argument("--require-artifacts", action="store_true")
    args = parser.parse_args(argv)
    path = args.checkpoint if args.checkpoint.name == "manifest.json" else args.checkpoint / "manifest.json"
    try:
        manifest = load_manifest(path, require_artifacts=args.require_artifacts)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"valid": True, "path": str(path.resolve()),
                      "stage": manifest["stage"], "parent": manifest.get("parent"),
                      "adapters": manifest["adapters"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
