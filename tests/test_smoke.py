"""核心组件冒烟测试（无 GPU / 无游戏可跑）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idv_agent.scripts.smoke_test import (
    test_intent_orthogonality,
    test_action_decoder,
    test_decode_state_machine_calibration,
    test_fast_model_forward,
    test_slow_model_forward,
    test_dataset_collator,
)


def test_all():
    test_intent_orthogonality()
    test_action_decoder()
    test_decode_state_machine_calibration()
    test_fast_model_forward()
    test_slow_model_forward()
    test_dataset_collator()


def test_decode_exit_not_rerun():
    """破译态被退出键(如 W)退出后，只要该退出键仍按住，就不应立即重新进入破译态。

    回归：早期版本在退出后下一帧因 Q 仍被 held 而重新进入，导致退出被抵消。
    """
    import pandas as pd
    from idv_agent.labels.state_machine import DecodeStateTracker, StateReplay

    end = 5_000_000_000
    events = pd.DataFrame([
        {"timestamp_ns": 0, "kind": "key_down", "code": "key:q", "value": 1},
        {"timestamp_ns": 500_000_000, "kind": "key_down", "code": "key:w", "value": 1},
    ])
    pos = pd.DataFrame(columns=["timestamp_ns", "x", "y"])
    replay = StateReplay(events, pos, end_ts_ns=end)
    tracker = DecodeStateTracker()

    s0 = tracker.on_frame(0, replay.frame_state(0, 0))
    assert s0.active, "Q press 应进入破译态"
    s1 = tracker.on_frame(600_000_000, replay.frame_state(600_000_000, 500_000_000))
    assert not s1.active, "按住 W 应退出破译态"
    s2 = tracker.on_frame(700_000_000, replay.frame_state(700_000_000, 600_000_000))
    assert not s2.active, "W 仍按住时不应重新进入破译态"


def test_decode_reenter_after_exit_key_released():
    """退出键松开且 Q 释放后再按 Q 才能重新进入破译态（防止"Q 一直按住"的重入）。"""
    import pandas as pd
    from idv_agent.labels.state_machine import DecodeStateTracker, StateReplay

    end = 5_000_000_000
    events = pd.DataFrame([
        {"timestamp_ns": 0, "kind": "key_down", "code": "key:q", "value": 1},
        {"timestamp_ns": 500_000_000, "kind": "key_down", "code": "key:w", "value": 1},
        {"timestamp_ns": 600_000_000, "kind": "key_up", "code": "key:w", "value": 0},
        # Q 保持按住（未释放）：即便 W 已松开也不应重入
        {"timestamp_ns": 700_000_000, "kind": "key_down", "code": "key:q", "value": 1},  # 无实质效果
    ])
    pos = pd.DataFrame(columns=["timestamp_ns", "x", "y"])
    replay = StateReplay(events, pos, end_ts_ns=end)
    tracker = DecodeStateTracker()

    tracker.on_frame(0, replay.frame_state(0, 0))
    tracker.on_frame(500_000_000, replay.frame_state(500_000_000, 0))
    assert not tracker.status().active, "W 使破译退出"
    # Q 一直按住：不应重入（防跨帧重入）
    st = tracker.on_frame(700_000_000, replay.frame_state(700_000_000, 600_000_000))
    assert not st.active, "Q 未释放前不应重新进入破译态"


def test_decode_reenter_after_q_released():
    """Q 完全释放后再按 Q → 重新进入破译态。"""
    import pandas as pd
    from idv_agent.labels.state_machine import DecodeStateTracker, StateReplay

    end = 5_000_000_000
    events = pd.DataFrame([
        {"timestamp_ns": 0, "kind": "key_down", "code": "key:q", "value": 1},
        {"timestamp_ns": 500_000_000, "kind": "key_down", "code": "key:w", "value": 1},
        {"timestamp_ns": 600_000_000, "kind": "key_up", "code": "key:w", "value": 0},
        {"timestamp_ns": 700_000_000, "kind": "key_up", "code": "key:q", "value": 0},      # Q 释放
        {"timestamp_ns": 800_000_000, "kind": "key_down", "code": "key:q", "value": 1},    # 再按 Q
    ])
    pos = pd.DataFrame(columns=["timestamp_ns", "x", "y"])
    replay = StateReplay(events, pos, end_ts_ns=end)
    tracker = DecodeStateTracker()

    tracker.on_frame(0, replay.frame_state(0, 0))
    tracker.on_frame(500_000_000, replay.frame_state(500_000_000, 0))
    assert not tracker.status().active, "W 使破译退出"
    # 中间帧(750ms)：Q 已释放且不再按住 → 清除等待释放
    tracker.on_frame(750_000_000, replay.frame_state(750_000_000, 500_000_000))
    # 再按 Q(800ms) → 应重入
    st = tracker.on_frame(800_000_000, replay.frame_state(800_000_000, 750_000_000))
    assert st.active, "Q 释放后再按 Q 应重新进入破译态"


def test_decode_single_q_tap_enters():
    """第五人格按一次 Q 后自动破译：Q 在两帧间完成 tap 也必须进入。"""
    import pandas as pd
    from idv_agent.labels.state_machine import DecodeStateTracker, StateReplay

    events = pd.DataFrame([
        {"timestamp_ns": 10_000_000, "kind": "key_down", "code": "key:q", "value": 1},
        {"timestamp_ns": 50_000_000, "kind": "key_up", "code": "key:q", "value": 0},
    ])
    pos = pd.DataFrame(columns=["timestamp_ns", "x", "y"])
    replay = StateReplay(events, pos, end_ts_ns=2_000_000_000)
    tracker = DecodeStateTracker()
    st = tracker.on_frame(100_000_000, replay.frame_state(100_000_000, 0))
    assert st.active, "一次 Q tap（按下+抬起均发生在帧间）应进入破译态"
