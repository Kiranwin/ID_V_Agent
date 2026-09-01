"""configs 模块导入 + 常量校验。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from idv_agent.configs.schema import NUM_CATEGORIES, NUM_CONTINUOUS, ActionCategory  # noqa: F401
from idv_agent.configs.intent import NUM_INTENT_CATEGORIES, INTENT_VECTOR_DIM  # noqa: F401
from idv_agent.configs.keymap import DEFAULT_SURVIVOR_KEYMAP  # noqa: F401
from idv_agent.configs.subgoal import (  # noqa: F401
    INTENT_SUBGOALS,
    SUBGOAL_NAMES,
)


def test_schema_consistency():
    assert NUM_CATEGORIES == len(ActionCategory) == 20
    assert NUM_CONTINUOUS == 4
    assert INTENT_VECTOR_DIM == 16
    assert NUM_INTENT_CATEGORIES == 10


def test_survivor_keymap_heal_binding():
    assert DEFAULT_SURVIVOR_KEYMAP.heal == "key:e"
    assert DEFAULT_SURVIVOR_KEYMAP.skill_2 is None


def test_subgoal_v1_aligns_all_vla_intents():
    from idv_agent.vla.action_chunk import INTENTS

    assert set(INTENT_SUBGOALS) == set(INTENTS)
    assert all(name in SUBGOAL_NAMES
               for candidates in INTENT_SUBGOALS.values()
               for name in candidates)


def test_frame_feature_cache_and_temporal_encoder():
    import torch
    from idv_agent.model.temporal import FrameFeatureCache, SharedTemporalEncoder, TaskConditionCache

    cache = FrameFeatureCache(max_length=8)
    for i in range(8):
        cache.append(i, (i + 1) * 100, torch.ones(4) * i)
    features, timestamps, valid = cache.window(8)
    assert features.shape == (8, 4)
    assert timestamps.tolist() == [100, 200, 300, 400, 500, 600, 700, 800]
    assert valid.all()
    encoder = SharedTemporalEncoder(input_dim=4, hidden_dim=6)
    output = encoder(features, valid_mask=valid,
                     time_deltas=(timestamps - timestamps[0]).float() / 1e9)
    assert output.shape == (1, 6)
    condition = TaskConditionCache(torch.zeros(8), torch.zeros(4), task_id="episode-1")
    assert condition.task_id == "episode-1"


def test_build_vla_v4_adds_slow_labels_and_timestamp_mask(tmp_path):
    import csv
    import json
    from idv_agent.scripts.build_vla_chunks import build
    from idv_agent.vla.action_chunk import VLA_SCHEMA_VERSION_V4

    session = tmp_path / "episode"
    frames = session / "frames"
    frames.mkdir(parents=True)
    for i in range(45):
        (frames / f"{i:08d}.jpg").write_bytes(b"jpeg")
    with (session / "per_frame_actions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("frame_idx", "timestamp_ns", "category_name",
                                                    "move_x", "move_y", "cam_dx", "cam_dy", "held_keys"))
        writer.writeheader()
        for i in range(45):
            writer.writerow({"frame_idx": i, "timestamp_ns": i * 100_000_000,
                             "category_name": "MOVE", "move_x": 0, "move_y": 1,
                             "cam_dx": 0, "cam_dy": 0, "held_keys": ""})
    (session / "intent_segments.jsonl").write_text(
        json.dumps({"start_frame": 0, "end_frame": 44, "intent": "search"}) + "\n",
        encoding="utf-8")
    output = session / "v4.jsonl"
    count = build(session, output, history=3, schema_version=VLA_SCHEMA_VERSION_V4,
                  slow_period_s=1.0)
    assert count > 0
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records[0]["schema_version"] == VLA_SCHEMA_VERSION_V4
    assert records[0]["slow_label"]["subgoal"] == "find_cipher"
    assert records[0]["loss_mask"] == {"slow": 1, "fast": 1}
    assert all("slow_label" in frame for frame in records[0]["observations"]["frames"])


def test_vla_heads_and_film_contract():
    import torch
    from idv_agent.configs.subgoal import SubgoalCategory
    from idv_agent.model.vla_heads import (
        CONTEXT_EMBEDDING_DIM, FiLMConditioner, FastVLAHead, SlowVLAHead,
    )

    batch, steps, feature_dim = 2, 8, 16
    frames = torch.randn(batch, steps, feature_dim)
    conditioner = FiLMConditioner(feature_dim)
    conditioned = conditioner(
        frames,
        intent_id=torch.tensor([0, 2]),
        subgoal_id=torch.tensor([1, int(SubgoalCategory.FIND_CIPHER)]),
        context_embedding=torch.randn(batch, CONTEXT_EMBEDDING_DIM),
        mode_id=torch.tensor([0, 0]),
    )
    assert conditioned.shape == frames.shape
    pooled = conditioned[:, -1]
    slow = SlowVLAHead(feature_dim)(pooled)
    assert slow.intent_logits.shape == (batch, 8)
    assert slow.subgoal_logits.shape[0] == batch
    assert slow.context_embedding.shape == (batch, CONTEXT_EMBEDDING_DIM)
    fast = FastVLAHead(feature_dim)(pooled)
    assert fast.move_logits.shape == (batch, 4, 9)
    assert fast.camera_dx_logits.shape == (batch, 4, 5)
    assert fast.button_logits.shape == (batch, 4, 6)
    assert fast.duration.shape == (batch, 4)
    assert torch.all((fast.duration >= 1) & (fast.duration <= 30))


def test_shared_vla_and_capture_fast_rate_decoupling():
    import torch
    from idv_agent.model.fast_slow_vla import FixedRateTrigger, SharedFastSlowVLA
    from idv_agent.model.temporal import FrameFeatureCache

    model = SharedFastSlowVLA(frame_feature_dim=8, temporal_dim=12)
    condition = model.initial_condition(batch_size=1)
    cache = FrameFeatureCache(max_length=8)
    fast_trigger = FixedRateTrigger(hz=15, start_timestamp_ns=0)
    fast_ticks = 0
    # 30 capture ticks over one second produce about 15 fast ticks. Appending a
    # feature does not itself call the model; the trigger owns consumption.
    for frame_idx in range(30):
        timestamp_ns = frame_idx * 33_333_333
        cache.append(frame_idx, timestamp_ns + 1, torch.randn(8))
        if fast_trigger.due(timestamp_ns):
            features, timestamps, valid = cache.window(8)
            deltas = (timestamps - timestamps[0]).float() / 1e9
            output = model(features, condition, valid_mask=valid,
                           time_deltas=deltas, run_slow=(fast_ticks == 0))
            assert output.fast.move_logits.shape == (1, 4, 9)
            if fast_ticks == 0:
                assert output.slow is not None
            fast_ticks += 1
    assert 14 <= fast_ticks <= 16


def test_vla_joint_loss_is_finite():
    import torch
    from idv_agent.model.vla_heads import FastVLAHead, SlowVLAHead
    from idv_agent.training.vla_loss import compute_vla_loss

    hidden = torch.randn(2, 12)
    fast = FastVLAHead(12)(hidden)
    slow = SlowVLAHead(12)(hidden)
    batch = {
        "fast_loss_mask": torch.ones(2),
        "slow_loss_mask": torch.ones(2),
        "intent_target": torch.tensor([0, 2]),
        "subgoal_target": torch.tensor([1, 3]),
        "subgoal_weight": torch.ones(2),
        "move_target": torch.zeros(2, 4, dtype=torch.long),
        "camera_dx_target": torch.full((2, 4), 2, dtype=torch.long),
        "camera_dy_target": torch.full((2, 4), 2, dtype=torch.long),
        "button_target": torch.zeros(2, 4, 6),
        "duration_target": torch.full((2, 4), 6.0),
    }
    losses = compute_vla_loss(fast, slow, batch)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_xanylabeling_json_to_yolo_handles_missing_negative_sidecars(tmp_path):
    import json
    from PIL import Image
    from idv_agent.scripts.convert_anylabeling_to_yolo import convert

    dataset = tmp_path / "yolo"
    image_dir = dataset / "images" / "train"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (100, 50), color="black").save(image_dir / "positive.jpg")
    Image.new("RGB", (100, 50), color="black").save(image_dir / "negative.jpg")
    (image_dir / "positive.json").write_text(json.dumps({
        "imageWidth": 100, "imageHeight": 50,
        "shapes": [{"label": "cipher_highlight", "shape_type": "rectangle",
                    "points": [[10, 5], [30, 5], [30, 25], [10, 25]]}],
    }), encoding="utf-8")
    summary = convert(dataset, split="train")
    assert summary["images"] == 2
    assert summary["json"] == 1
    assert summary["missing_json_as_empty"] == 1
    assert summary["boxes"] == 1
    assert (dataset / "labels/train/negative.txt").read_text(encoding="utf-8") == ""
    assert (dataset / "labels/train/positive.txt").read_text(encoding="utf-8") == "1 0.200000 0.300000 0.200000 0.400000\n"


def test_rule_policy_passes_spatial_state():
    from idv_agent.model.policy import RulePolicy

    class StubRule:
        def decide(self, spatial):
            assert spatial == {"visible": "yes", "distance": "near", "position": "center"}
            return 4, (0.0, 0.0, 0.0, 0.0)

    out = RulePolicy(StubRule()).decide(None, {
        "spatial": {"visible": "yes", "distance": "near", "position": "center"},
        "memory": {},
    })
    assert out.category_id == 4


def test_match_memory_summary_has_real_collections():
    from idv_agent.agent.memory import MatchMemory, MemoryEvent

    mem = MatchMemory()
    mem.add_event(MemoryEvent(kind="teammate_hooked", position="chair_A"))
    summary = mem.summary()
    assert summary["teammates_hooked"] == ["chair_A"]
    assert summary["n_events"] == 1


def test_match_memory_records_cipher_detection():
    from idv_agent.agent.memory import MatchMemory, MemoryEvent

    memory = MatchMemory()
    memory.add_event(MemoryEvent(
        kind="cipher_detection", position="right",
        detail='{"visible":"yes","source":"highlight"}',
    ))
    summary = memory.summary()
    assert summary["cipher_detection_count"] == 1
    assert summary["cipher_last_detection"]["position"] == "right"


def test_prepare_yolo_dataset_splits_by_session(tmp_path):
    from idv_agent.scripts.prepare_yolo_dataset import CLASSES, prepare

    root = tmp_path / "sessions"
    for sid in ("s1", "s2"):
        frames = root / sid / "frames"
        frames.mkdir(parents=True)
        for i in range(3):
            (frames / f"{i:08d}.jpg").write_bytes(b"jpeg")
    out = tmp_path / "yolo"
    copied = prepare(sessions_root=root, session_names=None, output=out,
                     stride=2, val_ratio=0.5, seed=1, max_per_session=None)
    assert copied == 4
    assert (out / "dataset.yaml").is_file()
    assert CLASSES == ("cipher_visible", "cipher_highlight", "interact_prompt")
    assert len(list((out / "images" / "train").glob("*.jpg"))) == 2
    assert len(list((out / "images" / "val").glob("*.jpg"))) == 2
    assert all(p.stat().st_size == 0 for p in (out / "labels" / "train").glob("*.txt"))


def test_run_agent_send_input_requires_admin(monkeypatch):
    """安全回归：非管理员进程不得进入真发送路径。"""
    import ctypes
    from idv_agent.scripts import run_agent

    class _Shell:
        @staticmethod
        def IsUserAnAdmin():
            return 0

    monkeypatch.setattr(ctypes, "windll", type("W", (), {"shell32": _Shell})(), raising=False)
    try:
        run_agent.main(["--mode", "rule", "--send-input", "--duration", "0"])
    except RuntimeError as exc:
        assert "管理员权限" in str(exc)
    else:
        raise AssertionError("非管理员 --send-input 应被拒绝")


def test_cipher_detector_safe_empty_and_spatial_shape():
    import numpy as np
    from idv_agent.agent.perception import CipherMachineDetector

    detector = CipherMachineDetector()
    result = detector.detect(np.zeros((120, 160, 3), dtype=np.uint8))
    spatial = result.as_spatial()
    assert set(("visible", "position", "distance", "confidence", "interact_prompt")) <= set(spatial)
    assert spatial["visible"] == "no"
    assert 0.0 <= spatial["confidence"] <= 1.0
    assert spatial["center_x"] is None
    assert spatial["bottom_y"] is None


def test_visual_servo_aligns_then_moves_forward_without_area_dependency():
    from idv_agent.agent.rule_agent import CipherVisualServo
    from idv_agent.configs.schema import ActionCategory

    servo = CipherVisualServo(prompt_confirm_frames=2)
    # Far left: turn only, no forward key.
    cat, cont = servo.decide({"visible": "yes", "center_x": 0.2,
                              "bbox_area_ratio": 0.001})
    assert cat == int(ActionCategory.LOOK)
    assert cont[1] == 0 and cont[2] < 0
    # In the centre corridor: advance.  A sudden area drop must not stop us.
    cat, cont = servo.decide({"visible": "yes", "center_x": 0.51,
                              "bbox_area_ratio": 0.0001})
    assert cat == int(ActionCategory.MOVE)
    assert cont[1] == 1.0


def test_visual_servo_requires_consecutive_prompt_and_taps_once():
    from idv_agent.agent.rule_agent import CipherVisualServo
    from idv_agent.configs.schema import ActionCategory

    servo = CipherVisualServo(prompt_confirm_frames=2)
    spatial = {"visible": "yes", "center_x": 0.5, "interact_prompt": "yes"}
    cat, _ = servo.decide(spatial)
    assert cat != int(ActionCategory.INTERACT_TAP)
    cat, _ = servo.decide(spatial)
    assert cat == int(ActionCategory.INTERACT_TAP)
    cat, _ = servo.decide(spatial)
    assert cat != int(ActionCategory.INTERACT_TAP)
    # A frame-level decoding state takes over and stays in hold semantics.
    cat, _ = servo.decide({"visible": "yes", "center_x": 0.5,
                           "frame_state": "decoding"})
    assert cat == int(ActionCategory.INTERACT_HOLD)


def test_cipher_detector_template_match(tmp_path):
    import cv2
    import numpy as np
    from idv_agent.agent.perception import CipherMachineDetector

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    patch = np.full((18, 14, 3), (20, 180, 220), dtype=np.uint8)
    frame[55:73, 72:86] = patch
    template_path = tmp_path / "cipher.jpg"
    cv2.imwrite(str(template_path), patch)
    result = CipherMachineDetector(template_path=template_path).detect(frame)
    assert result.source == "template"
    assert result.visible == "yes"
    assert result.position == "center"


def test_cipher_detector_through_wall_highlight():
    import numpy as np
    from idv_agent.agent.perception import CipherMachineDetector

    # Synthetic through-wall beacon: tall, saturated yellow marker below HUD.
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    frame[55:125, 112:120] = (0, 255, 255)  # BGR yellow
    result = CipherMachineDetector().detect(frame)
    assert result.source == "highlight"
    assert result.visible == "yes"
    assert result.position == "center"
    assert result.confidence >= 0.78


def test_run_agent_offline_image_mode(tmp_path, capsys):
    import cv2
    import numpy as np
    from idv_agent.scripts import run_agent

    image = tmp_path / "frame.jpg"
    frame = np.zeros((180, 240, 3), dtype=np.uint8)
    frame[55:125, 112:120] = (0, 255, 255)
    cv2.imwrite(str(image), frame)
    assert run_agent.main(["--mode", "rule", "--test-image", str(image)]) == 0
    output = capsys.readouterr().out
    assert "[test] offline" in output
    assert "action=MOVE_LOOK" in output or "action=MOVE" in output


def test_realtime_agent_rejects_invalid_camera_scale():
    import torch
    from idv_agent.agent.action_executor import ActionExecutor
    from idv_agent.agent.realtime_agent import RealtimeAgent
    from idv_agent.model.policy import RulePolicy
    from idv_agent.agent.rule_agent import RuleAgent
    from idv_agent.capture.screen_capture import CaptureConfig
    try:
        RealtimeAgent(RulePolicy(RuleAgent()), ActionExecutor(dry_run=True),
                      CaptureConfig(), cam_pixel_scale=0)
    except ValueError as exc:
        assert "cam_pixel_scale" in str(exc)
    else:
        raise AssertionError("invalid camera scale should fail")
