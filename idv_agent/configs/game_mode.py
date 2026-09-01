"""Supported Identity V game modes for VLA conditioning and adapters."""

from __future__ import annotations

GAME_MODE_CHOICES = ("standard", "joint_hunt", "blackjack")
DEFAULT_GAME_MODE = "standard"


def mode_token(mode: str) -> str:
    """Return the canonical language-conditioning token for a game mode."""
    if mode not in GAME_MODE_CHOICES:
        raise ValueError(f"未知游戏模式: {mode!r}，可选 {GAME_MODE_CHOICES}")
    return f"<mode:{mode}>"
