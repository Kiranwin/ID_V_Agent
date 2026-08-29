"""慢层 BC 训练 CLI 入口（薄封装 bc_slow.train_actor_critic）。

用法：
    python -m idv_agent.scripts.train_slow --session data/sessions/<id> --steps 200 [--use-qwen --fp16 --device cuda]
"""

from __future__ import annotations

import argparse
from idv_agent.training.bc_slow import main as bc_slow_main


def main(argv=None) -> int:
    return bc_slow_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
