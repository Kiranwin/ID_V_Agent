"""Model-independent supervision for the learned interact-prompt -> Q task.

Used for constructing labels, never as a runtime annotation lookup. The
deployed visual model must predict the token from image evidence itself.
"""
from __future__ import annotations

Q_CONTRACT = {
    "id": "interact_prompt_to_q.v1",
    "target_source": "reviewed_current_frame_interact_prompt",
    "positive_token": "Q",
    "negative_token": "NO_Q",
    "short_repeats_allowed": True,
    "requires_replay_q": False,
    "requires_decoding_outcome": False,
    "requires_navigation_phase": False,
}


def prompt_q_target(interact_prompt: bool | None) -> dict:
    """Known presence/absence is a label; unknown is masked, never negative."""
    if interact_prompt is not None and type(interact_prompt) is not bool:
        raise ValueError("interact_prompt must be true/false/null")
    return {"q": interact_prompt, "token": ("Q" if interact_prompt else "NO_Q")
            if interact_prompt is not None else None, "mask": interact_prompt is not None}
