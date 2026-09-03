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

    assert output_fast.slow is None
    assert output_fast.fast.move_logits.shape == (batch_size, 4, 9)


def test_separated_temporal_gradient_flow_is_independent():
    """Verify fast loss does not propagate to slow temporal encoder."""
    core = SharedFastSlowVLA(frame_feature_dim=64, temporal_dim=128, history_action_dim=72)

    batch_size = 1
    seq_len = 8
    frame_features = torch.randn(batch_size, seq_len, 64)

    condition = core.initial_condition(batch_size, device="cpu")
    history_actions = torch.zeros(batch_size, 72)

    # Forward pass with detach_slow_condition=False (allow gradient flow to context_embedding)
    output = core(
        frame_features, condition,
        history_actions=history_actions,
        run_slow=True, detach_slow_condition=False
    )

    # Compute a dummy fast loss (only affects fast head)
    fast_loss = output.fast.move_logits.sum()
    fast_loss.backward(retain_graph=True)

    # Check gradients: fast_temporal should have gradients, slow_temporal should not
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

    # Clear gradients
    core.zero_grad()

    # Compute a dummy slow loss (only affects slow head)
    slow_loss = output.slow.intent_logits.sum()
    slow_loss.backward()

    # Now only slow_temporal should have gradients
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


if __name__ == "__main__":
    test_separated_temporal_encoders_have_independent_parameters()
    test_separated_forward_produces_valid_outputs()
    test_separated_temporal_gradient_flow_is_independent()
    print("All tests passed!")
