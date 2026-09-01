"""无 GPU / 无游戏冒烟测试：验证 VLA 核心与规则安全兜底。

覆盖：
- intent 正交性
- ActionDecoder 差分逻辑（press/release/mouse）
- 破译状态机校准标记（P2）
- VLA action-chunk schema

用法：
    python -m idv_agent.scripts.smoke_test
"""

from __future__ import annotations

import sys


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        print(f"  [FAIL] {msg}")
        raise SystemExit(1)
    print(f"  [ok] {msg}")


def test_intent_orthogonality() -> None:
    from idv_agent.configs.intent import get_prototype_matrix
    M = get_prototype_matrix()
    n = M.shape[0]
    _expect(M.shape[1] == 16, "intent 维度=16")
    for i in range(n):
        for j in range(n):
            if i != j:
                _expect(abs(M[i] @ M[j]) < 1e-4, f"intent {i},{j} 正交")
    _expect(abs(M[0] @ M[0] - 1.0) < 1e-4, "intent 单位长度")


def test_action_decoder() -> None:
    from idv_agent.agent.action_decoder import ActionDecoder
    from idv_agent.configs.schema import ActionCategory
    dec = ActionDecoder()
    # 按 D 移动
    cmds = dec.decode(int(ActionCategory.MOVE), [1.0, 0.0, 0.0, 0.0])
    _expect(any(c.kind == "press" and c.code == "key:d" for c in cmds), "MOVE 按下 D")
    # 释放（无按键）
    cmds2 = dec.decode(int(ActionCategory.NOOP), [0, 0, 0, 0])
    _expect(any(c.kind == "release" and c.code == "key:d" for c in cmds2), "NOOP 释放 D")
    # 鼠标移动
    cmds3 = dec.decode(int(ActionCategory.LOOK), [0, 0, 0.5, 0])
    _expect(any(c.kind == "mouse_move" for c in cmds3), "LOOK 产生鼠标移动")


def test_decode_state_machine_calibration() -> None:
    import pandas as pd
    from idv_agent.labels.state_machine import DecodeStateTracker, StateReplay
    # 合成事件：按 Q 进入破译（0.0s），0.5s 按 Space（校准，未抬起）
    # 录制末端设置足够大，保证 Space 在校准帧仍处于 held 态。
    end_ns = 5_000_000_000   # 5s
    events = pd.DataFrame([
        {"timestamp_ns": 0, "kind": "key_down", "code": "key:q", "value": 1},
        {"timestamp_ns": 500_000_000, "kind": "key_down", "code": "key:space", "value": 1},
    ])
    positions = pd.DataFrame(columns=["timestamp_ns", "x", "y"])
    replay = StateReplay(events, positions, end_ts_ns=end_ns)
    tracker = DecodeStateTracker()
    # 帧 0：Q 按下 → 进入破译
    st0 = tracker.on_frame(0, replay.frame_state(0, 0))
    _expect(st0.active, "Q press 进入破译态")
    # 帧 0.6s：Space 仍在 held → 校准
    st1 = tracker.on_frame(600_000_000, replay.frame_state(600_000_000, 500_000_000))
    _expect(st1.active and st1.calibration, "破译中按 Space 标记校准（P2）")
    # 帧 0.7s：Space 已抬起（无释放事件则 held 仍持续——用单独事件验证非校准场景略过）
    st2 = tracker.on_frame(700_000_000, replay.frame_state(700_000_000, 600_000_000))
    _expect(st2.active, "Space 校准后仍保持破译态")


def test_vla_contract() -> None:
    from idv_agent.configs.game_mode import mode_token
    from idv_agent.vla.action_chunk import validate_record

    record = {
        "schema_version": "vla.action_chunk.v3",
        "episode_id": "smoke",
        "anchor_frame": 2,
        "mode": "standard",
        "task": {
            "name": "find_cipher_and_decode",
            "instruction": "找到密码机，靠近并进入破译",
            "mode_token": mode_token("standard"),
        },
        "observations": {
            "frames": [
                {"path": "frames/00000000.jpg", "frame_index": 0, "timestamp_ns": 0},
                {"path": "frames/00000001.jpg", "frame_index": 1, "timestamp_ns": 33_000_000},
                {"path": "frames/00000002.jpg", "frame_index": 2, "timestamp_ns": 66_000_000},
            ],
            "fps": 30,
            "history_actions": [],
        },
        "action_chunk": [
            {"move_dir": 1, "camera_dx": 0, "camera_dy": 0,
             "buttons": [0, 0, 0, 0, 0, 0], "duration_frames": 6}
        ] * 4,
        "alignment": {
            "mode": "causal_future",
            "action_delay_frames": 1,
            "observation_end_frame": 2,
            "action_start_frame": 4,
            "observation_end_timestamp_ns": 66_000_000,
            "action_start_timestamp_ns": 99_000_000,
            "action_end_timestamp_ns": 759_000_000,
        },
        "quality": {"source": "human", "outcome": "unknown"},
    }
    validate_record(record)
    _expect(True, "VLA action-chunk contract OK")


def main() -> int:
    print("=== ID_V_Agent smoke_test ===")
    print("\n[1] intent 正交性"); test_intent_orthogonality()
    print("\n[2] ActionDecoder"); test_action_decoder()
    print("\n[3] 破译状态机（P2 校准）"); test_decode_state_machine_calibration()
    print("\n[4] VLA action-chunk contract"); test_vla_contract()
    print("\n=== 全部通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
