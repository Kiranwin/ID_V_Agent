"""Test separated temporal encoders for gradient independence."""

import torch

from idv_agent.model.fast_slow_vla import SharedFastSlowVLA


def test_separated_temporal_encoders_have_independent_parameters():
    """Verify slow and fast heads use different GRU parameters."""
    core = SharedFastSlowVLA(frame_feature_dim=128, temporal_dim=256, history_action_dim=72)

    # Check that slow_temporal and fast_temporal are separate modules
    assert hasattr(core, "slow_temporal")
    assert hasattr(core, "fast_temporal")
    assert core.slow_temporal is not core.fast_temporal

    # Verify parameter independence
    slow_params = set(core.slow_temporal.parameters())
    fast_params = set(core.fast_temporal.parameters())
    assert len(slow_params & fast_params) == 0, "Slow and fast temporal encoders share parameters"

    # Check total parameter count increase (~2x GRU params)
    total_params = sum(p.numel() for p in core.parameters())
    slow_temporal_params = sum(p.numel() for p in core.slow_temporal.parameters())
    fast_temporal_params = sum(p.numel() for p in core.fast_temporal.parameters())

    assert slow_temporal_params > 0
    assert fast_temporal_params > 0
    print(f"Slow temporal: {slow_temporal_params:,} params")
    print(f"Fast temporal: {fast_temporal_params:,} params")
    print(f"Total model: {total_params:,} params")


def test_separated_forward_produces_valid_outputs():
    """Verify separated architecture still produces valid slow/fast outputs."""
    core = SharedFastSlowVLA(frame_feature_dim=128, temporal_dim=256, history_action_dim=72)

    batch_size = 2
    seq_len = 8
    frame_features = torch.randn(batch_size, seq_len, 128)

    condition = core.initial_condition(batch_size, device="cpu")
    history_actions = torch.zeros(batch_size, 72)

    # Run slow pass
    output_slow = core(
        frame_features, condition,
        history_actions=history_actions,
        run_slow=True, detach_slow_condition=False
    )

    assert output_slow.slow is not None
    assert output_slow.slow.intent_logits.shape == (batch_size, 8)  # 8 intents
    assert output_slow.slow.context_embedding.shape == (batch_size, 256)
    assert output_slow.fast.move_logits.shape == (batch_size, 4, 9)  # 4 steps, 9 directions

    # Run fast-only pass
    next_condition = core.condition_from_slow_output(
        output_slow.slow, mode_id=torch.zeros(batch_size, dtype=torch.long)
    )
    output_fast = core(
        frame_features, next_condition,
        history_actions=history_actions,
        run_slow=False
    )

    # m28 always exposes its visual state intent/subgoal in the same planner
    # pass.  Only the legacy temporal diagnostic slow branch is absent here.
    assert output_fast.slow is not None
    assert output_fast.prior_slow is None
    assert output_fast.fast.move_logits.shape == (batch_size, 4, 9)


def test_single_pass_gradient_flow_is_independent_when_condition_is_fixed():
    """A single forward call with a fixed (non-graph) condition keeps the two
    GRUs fully independent: fast loss touches only fast_temporal, slow loss
    touches only slow_temporal.  This is the cold-start case (condition comes
    from ``initial_condition``, a fresh constant tensor with no autograd
    history) -- it does NOT cover the two-pass training scheme, see the next
    test for that.
    """
    core = SharedFastSlowVLA(frame_feature_dim=64, temporal_dim=128, history_action_dim=72)

    batch_size = 1
    seq_len = 8
    frame_features = torch.randn(batch_size, seq_len, 64)

    condition = core.initial_condition(batch_size, device="cpu")
    history_actions = torch.zeros(batch_size, 72)

    output = core(
        frame_features, condition,
        history_actions=history_actions,
        run_slow=True, detach_slow_condition=False
    )

    fast_loss = output.prior_fast.move_logits.sum()
    fast_loss.backward(retain_graph=True)

    fast_temporal_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.fast_temporal.parameters()
    )
    slow_temporal_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.slow_temporal.parameters()
    )

    assert fast_temporal_has_grad, "Fast temporal should have gradients from fast loss"
    assert not slow_temporal_has_grad, "Slow temporal should NOT have gradients from fast loss"

    core.zero_grad()

    slow_loss = output.prior_slow.intent_logits.sum()
    slow_loss.backward()

    fast_temporal_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.fast_temporal.parameters()
    )
    slow_temporal_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.slow_temporal.parameters()
    )

    assert not fast_temporal_has_grad, "Fast temporal should NOT have gradients from slow loss"
    assert slow_temporal_has_grad, "Slow temporal should have gradients from slow loss"


def test_two_pass_training_scheme_still_couples_fast_loss_into_slow_head():
    """Document the real (non-independent) behavior of train_vla.py's scheme.

    Pass 1 runs the slow head; pass 2 builds ``next_condition`` from that
    slow output via ``condition_from_slow_output`` and feeds it back with
    ``detach_slow_condition=False``.  FiLM conditioning is applied *before*
    the slow/fast split, so ``context_embedding`` -- and therefore
    ``slow_head``/``slow_temporal`` -- still receives gradient from the fast
    loss in this scheme.  Separating the GRUs removed competition over the
    *recurrent* weights, but did not remove this FiLM coupling; that is
    intentional (it is how the fast head learns to use slow context) and
    remains controllable via ``detach_slow_condition``.
    """
    core = SharedFastSlowVLA(frame_feature_dim=64, temporal_dim=128, history_action_dim=72)

    batch_size = 1
    frame_features = torch.randn(batch_size, 8, 64)
    condition = core.initial_condition(batch_size, device="cpu")
    history_actions = torch.zeros(batch_size, 72)

    slow_pass = core(frame_features, condition, history_actions=history_actions, run_slow=True)
    next_condition = core.condition_from_slow_output(
        slow_pass.slow, mode_id=torch.zeros(batch_size, dtype=torch.long)
    )
    fast_pass = core(frame_features, next_condition, history_actions=history_actions,
                     run_slow=False, detach_slow_condition=False)

    fast_pass.fast.move_logits.sum().backward()

    slow_temporal_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.slow_temporal.parameters()
    )
    slow_head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.slow_head.parameters()
    )
    assert not slow_temporal_has_grad, "visual-only deployment output must not couple into slow_temporal"
    assert not slow_head_has_grad, "visual-only deployment output must not couple into slow_head"

    # detach_slow_condition=True on the second pass removes this coupling.
    core.zero_grad()
    fast_pass_detached = core(frame_features, next_condition, history_actions=history_actions,
                              run_slow=False, detach_slow_condition=True)
    fast_pass_detached.fast.move_logits.sum().backward()
    slow_temporal_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in core.slow_temporal.parameters()
    )
    assert not slow_temporal_has_grad, "visual-only deployment output should stay independent"


if __name__ == "__main__":
    test_separated_temporal_encoders_have_independent_parameters()
    test_separated_forward_produces_valid_outputs()
    test_single_pass_gradient_flow_is_independent_when_condition_is_fixed()
    test_two_pass_training_scheme_still_couples_fast_loss_into_slow_head()
    print("All tests passed!")
