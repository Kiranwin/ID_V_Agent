"""configs 模块导入 + 常量校验。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idv_agent.configs.schema import NUM_CATEGORIES, NUM_CONTINUOUS, ActionCategory  # noqa: F401
from idv_agent.configs.intent import NUM_INTENT_CATEGORIES, INTENT_VECTOR_DIM  # noqa: F401
from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP  # noqa: F401


def test_schema_consistency():
    assert NUM_CATEGORIES == len(ActionCategory) == 20
    assert NUM_CONTINUOUS == 4
    assert INTENT_VECTOR_DIM == 16
    assert NUM_INTENT_CATEGORIES == 10
