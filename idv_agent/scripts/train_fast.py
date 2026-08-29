"""快层 BC 训练 CLI 入口（薄封装 bc_fast.train_fast_controller）。

用法：
    python -m idv_agent.scripts.train_fast --session data/sessions/<id> --steps 200 [--use-siglip --fp16 --device cuda]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from idv_agent.training.bc_fast import main as bc_fast_main


def main(argv=None) -> int:
    return bc_fast_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
