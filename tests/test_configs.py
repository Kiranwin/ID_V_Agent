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


def test_survivor_keymap_heal_binding():
    assert DEFAULT_SURVIVOR_KEYMAP.heal == "key:e"
    assert DEFAULT_SURVIVOR_KEYMAP.skill_2 is None


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
    assert CLASSES == ("cipher_visible", "cipher_highlight", "interact_prompt", "decoding_state")
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
    assert set(("visible", "position", "distance", "confidence", "interact_prompt", "decoding_state")) <= set(spatial)
    assert spatial["visible"] == "no"
    assert 0.0 <= spatial["confidence"] <= 1.0


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
