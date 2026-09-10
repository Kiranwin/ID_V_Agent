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


def test_subgoal_v1_matches_frozen_intent_table():
    expected = {
        "decipher": ("approach_cipher", "start_decoding", "handle_qte"),
        "kite": ("locate_safe_point", "maintain_distance", "vault_window", "drop_pallet", "disengage"),
        "rescue": ("approach_chair", "search_area", "rescue_teammate", "heal_teammate", "wait_opportunity"),
        "rotate": ("disengage", "move_to_zone"),
        "travel": ("find_cipher", "move_to_target", "avoid_obstacle"),
        "search": ("find_chest", "open_chest", "pickup_item"),
        "gate": ("move_to_gate", "open_gate", "escape"),
        "idle": ("observe", "hide"),
    }
    assert INTENT_SUBGOALS == expected
    assert set(name for names in expected.values() for name in names) <= set(SUBGOAL_NAMES)


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


def test_run_agent_only_accepts_sgan_mode():
    """回归：旧 rule/act 模式已从 run_agent 移除，只保留 sgan。"""
    import pytest

    from idv_agent.scripts import run_agent

    for removed in ("rule", "act", "m29"):
        with pytest.raises(SystemExit):
            run_agent.main(["--mode", removed])


def test_run_agent_sgan_requires_existing_checkpoint_and_fresh_trace(tmp_path):
    """回归：SGan 必须给出存在的 checkpoint 与全新的 trace 路径。"""
    import pytest

    from idv_agent.scripts import run_agent

    with pytest.raises(SystemExit):
        run_agent.main(["--mode", "sgan"])
    checkpoint = tmp_path / "m29.pt"
    checkpoint.write_bytes(b"stub")
    with pytest.raises(SystemExit):
        run_agent.main(["--mode", "sgan", "--checkpoint", str(checkpoint),
                        "--trace", str(tmp_path / "trace.jsonl"),
                        "--allow-undeployed-send-input"])
    trace = tmp_path / "trace.jsonl"
    trace.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        run_agent.main(["--mode", "sgan", "--checkpoint", str(checkpoint),
                        "--trace", str(trace)])
    with pytest.raises(SystemExit):
        run_agent.main(["--mode", "sgan", "--checkpoint", str(tmp_path / "missing.pt"),
                        "--trace", str(tmp_path / "fresh.jsonl")])
