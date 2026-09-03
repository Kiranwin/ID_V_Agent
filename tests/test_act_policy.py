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


def test_act_policy_cold_start_uses_travel_and_waits_for_three_frames():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", send=lambda _commands: None)
    assert int(policy.condition.intent_id[0]) == INTENTS.index("travel")
    assert torch.count_nonzero(policy.history_actions).item() == 0
    assert policy.observe(1, frame_index=0, timestamp_ns=1) is False
    assert policy.observe(1, frame_index=1, timestamp_ns=2) is False


def test_act_policy_uses_real_history_after_executor_starts_a_step():
    core = SharedFastSlowVLA(4, temporal_dim=8, history_action_dim=72)
    sent = []
    policy = ACTPolicy(_Adapter(), core, instruction="find", mode="standard",
                       device="cpu", send=lambda commands: sent.extend(commands))
    for i in range(3):
        policy.observe(1, frame_index=i, timestamp_ns=i + 1)
    policy.tick(now=0.0)
    assert policy.history_actions.shape == (1, 72)
    # The first predicted block is applied immediately; its encoded history
    # is then available to the next slow/fast pass.
    policy.executor.tick(dt_s=0.001)
    policy.history_actions = policy.executor.history_tensor(device="cpu")
    assert policy.history_actions.shape == (1, 72)
