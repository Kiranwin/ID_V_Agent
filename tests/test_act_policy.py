import pytest
import torch

from idv_agent.agent.action_executor import ActionExecutor
from idv_agent.agent.act_policy import ACTPolicy
from idv_agent.model.fast_slow_vla import SharedFastSlowVLA
from idv_agent.model.temporal import TaskConditionCache
from idv_agent.vla.action_chunk import INTENTS


class _Adapter:
    hidden_size = 4

    def encode_task_once(self, instruction, mode, *, task_id):
        return TaskConditionCache(torch.zeros(4), torch.zeros(4), task_id=task_id, mode=mode)

    def encode_frame(self, image, task_cache):
        return torch.ones(4) * float(image)


def test_act_policy_cold_start_uses_travel_and_waits_for_full_strided_window():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", send=lambda _commands: None)
    assert int(policy.condition.intent_id[0]) == INTENTS.index("travel")
    assert torch.count_nonzero(policy.history_actions).item() == 0
    for i in range(21):
        assert policy.observe(1, frame_index=i, timestamp_ns=i + 1) is False
    assert policy.observe(1, frame_index=21, timestamp_ns=22) is True


def test_act_policy_alignment_controls_window_and_time_deltas():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", history_frames=8, history_stride=3,
                       use_time_deltas=False)
    assert policy.history_frames == 8
    assert policy.history_stride == 3
    assert policy.use_time_deltas is False
    for i in range(22):
        policy.observe(i, frame_index=i, timestamp_ns=i + 1)
    features, deltas, valid = policy._temporal_inputs()
    assert features.shape == (8, 4)
    assert features[:, 0].tolist() == [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0]
    assert deltas is None
    assert valid.shape == (8,)


def test_act_policy_rejects_untrained_time_deltas():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    try:
        ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                  device="cpu", use_time_deltas=True)
    except ValueError as exc:
        assert "time_deltas" in str(exc)
    else:
        raise AssertionError("v5 ACT must reject time_deltas until retrained")


def test_act_policy_rejects_noncanonical_capture_fps():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    with pytest.raises(ValueError, match="capture_fps"):
        ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                  device="cpu", capture_fps=20.0)


def test_act_policy_emits_fixed_v5_duration():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu")
    for i in range(22):
        policy.observe(1, frame_index=i, timestamp_ns=i + 1)
    steps = policy._predict_chunk(None)

    # Runtime must execute only the causal first macro-step then re-observe.
    assert [step.duration_frames for step in steps] == [6]


def test_act_policy_zero_history_diagnostic_does_not_consume_executor_history():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", zero_history=True)
    assert policy.zero_history is True
    assert torch.count_nonzero(policy.history_actions).item() == 0


def test_act_policy_logs_action_chunk_metadata(capsys):
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", send=lambda _commands: None)
    for i in range(22):
        policy.observe(1, frame_index=i, timestamp_ns=i + 1)
    policy.tick(now=0.0)
    output = capsys.readouterr().out
    assert "[act]" in output
    assert "pred=" in output
    assert "frame=" in output
    assert "intent=" in output
    assert "move=" in output
    assert "camera=" in output
    assert "buttons=" in output
    assert "duration=" in output
    assert "feature_mean=" in output
    assert "chunk=" in output


def test_act_policy_uses_real_history_after_executor_starts_a_step():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    sent = []
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", send=lambda commands: sent.extend(commands))
    for i in range(22):
        policy.observe(1, frame_index=i, timestamp_ns=i + 1)
    policy.tick(now=0.0)
    assert policy.history_actions.shape == (1, 72)
    # The first predicted block is applied immediately; its encoded history
    # is then available to the next slow/fast pass.
    policy.executor.tick(dt_s=0.001)
    policy.history_actions = policy.executor.history_tensor(device="cpu")
    assert policy.history_actions.shape == (1, 72)
