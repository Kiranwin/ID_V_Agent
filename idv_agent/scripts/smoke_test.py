"""无 GPU / 无游戏冒烟测试：验证核心管线组件可运行。

覆盖：
- intent 正交性
- ActionDecoder 差分逻辑（press/release/mouse）
- 破译状态机校准标记（P2）
- 动作提取（合成事件）
- 模型 forward（dummy fast + dummy slow）
- Dataset + Collator

用法：
    python -m idv_agent.scripts.smoke_test
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


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


def test_fast_model_forward() -> None:
    import torch
    from idv_agent.configs.intent import INTENT_VECTOR_DIM
    from idv_agent.model.dummy_vision import DummyVisionEncoder
    from idv_agent.model.fast_controller import FastController
    from idv_agent.configs.schema import NUM_CATEGORIES, NUM_CONTINUOUS
    vision = DummyVisionEncoder(hidden_size=96)
    model = FastController(vision_encoder=vision, vision_hidden_size=96,
                           intent_dim=INTENT_VECTOR_DIM, skeleton_dim=32,
                           num_categories=NUM_CATEGORIES, num_continuous=NUM_CONTINUOUS)
    model.eval()
    with torch.no_grad():
        out = model(
            pixel_values=torch.randn(2, 3, 64, 64),
            intent_vector=torch.randn(2, INTENT_VECTOR_DIM),
            skeleton_embedding=torch.randn(2, 32),
            category_labels=torch.tensor([0, 1]),
            continuous_labels=torch.randn(2, NUM_CONTINUOUS),
        )
    _expect(out.loss is not None and out.category_logits.shape == (2, NUM_CATEGORIES),
            "FastController forward OK")


def test_slow_model_forward() -> None:
    import torch
    from idv_agent.model.dummy_backbone import DummyBackbone
    from idv_agent.model.game_actor_critic import GameActorCritic
    from idv_agent.configs.schema import NUM_CONTINUOUS
    backbone = DummyBackbone(vocab_size=50, hidden_size=32, image_feature=3)
    model = GameActorCritic(backbone=backbone, hidden_size=32, numeric_dim=NUM_CONTINUOUS)
    model.eval()
    with torch.no_grad():
        out = model(
            pixel_values=torch.randn(1, 3, 32, 32),
            input_ids=torch.randint(0, 50, (1, 8)),
            labels=torch.randint(0, 50, (1, 8)),
            numeric_labels=torch.randn(1, NUM_CONTINUOUS),
        )
    _expect(out.loss is not None and out.numeric_mean.shape == (1, NUM_CONTINUOUS),
            "GameActorCritic forward OK")


def test_dataset_collator() -> None:
    import json
    import torch
    from idv_agent.labels.dataset import SessionDataset
    from idv_agent.labels.collator import FastControllerCollator
    from idv_agent.labels.tokenizer import SkeletonVocab
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "frames").mkdir()
        # 生成占位图
        import numpy as np
        from PIL import Image
        for i in range(2):
            Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(td / "frames" / f"{i:08d}.jpg")
        recs = [{
            "image": f"frames/{i:08d}.jpg",
            "conversations": [{"from": "human", "value": "q"}, {"from": "gpt", "value": "意图: decode\n动作骨架: F_HOLD\n完整操作: F_HOLD"}],
            "slow_system_output": {"intent_id": 2, "intent_name": "decode", "intent_vector": [0.0]*16, "action_skeleton": "F_HOLD"},
            "fast_system_input": {"intent_vector": [0.0]*16, "action_skeleton": "F_HOLD"},
            "fast_system_output": {"category_id": 5, "category_name": "INTERACT_HOLD", "text_action": "F_HOLD", "numeric_action": [0,0,0,0]},
        } for i in range(2)]
        with (td / "samples.jsonl").open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ds = SessionDataset([td / "samples.jsonl"], image_size=224, mode="tensor")
        vocab = SkeletonVocab.build_from_jsonls([td / "samples.jsonl"])
        coll = FastControllerCollator(skeleton_vocab=vocab)
        batch = coll([ds[0], ds[1]])
        _expect(batch["pixel_values"].shape == (2, 3, 224, 224), "Dataset/collator 像素 batch OK")
        _expect(torch.is_tensor(batch["category_labels"]) and batch["category_labels"].shape == (2,),
                "Dataset/collator 标签 OK")


def main() -> int:
    print("=== ID_V_Agent smoke_test ===")
    print("\n[1] intent 正交性"); test_intent_orthogonality()
    print("\n[2] ActionDecoder"); test_action_decoder()
    print("\n[3] 破译状态机（P2 校准）"); test_decode_state_machine_calibration()
    print("\n[4] FastController forward"); test_fast_model_forward()
    print("\n[5] GameActorCritic forward"); test_slow_model_forward()
    print("\n[6] Dataset + collator"); test_dataset_collator()
    print("\n=== 全部通过 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
