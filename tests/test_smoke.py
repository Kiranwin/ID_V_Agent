"""核心组件冒烟测试（无 GPU / 无游戏可跑）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idv_agent.scripts.smoke_test import (
    test_intent_orthogonality,
    test_action_decoder,
    test_vla_contract,
)


def test_all():
    test_intent_orthogonality()
    test_action_decoder()
    test_vla_contract()
