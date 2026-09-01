"""VLA v1 subgoal vocabulary and intent alignment.

Subgoals are derived labels in v1. Annotators provide the top-level intent;
the builder derives a subgoal from frame state, visual detections and input
events. Keeping the vocabulary here prevents free-form strings from leaking
into the training schema.
"""

from __future__ import annotations

from enum import IntEnum


SUBGOAL_VOCAB_VERSION = "subgoal.v1"


class SubgoalCategory(IntEnum):
    NONE = 0
    OBSERVE = 1
    SEARCH_AREA = 2
    FIND_CIPHER = 3
    APPROACH_CIPHER = 4
    FACE_CIPHER = 5
    START_DECODING = 6
    MAINTAIN_DECODING = 7
    HANDLE_QTE = 8
    RECOVER_TARGET = 9
    CHOOSE_DESTINATION = 10
    MOVE_TO_TARGET = 11
    AVOID_OBSTACLE = 12
    LOCATE_SAFE_POINT = 13
    MAINTAIN_DISTANCE = 14
    RESCUE_TEAMMATE = 15
    OPEN_GATE = 16
    ESCAPE = 17


SUBGOAL_NAMES = tuple(item.name.lower() for item in SubgoalCategory)

# Every top-level intent from intents.md has at least one legal subgoal.  The
# mapping is intentionally many-to-one: v1 is meant to be rule-derived, not a
# second annotation task.
INTENT_SUBGOALS: dict[str, tuple[str, ...]] = {
    "decipher": (
        "find_cipher", "approach_cipher", "face_cipher", "start_decoding",
        "maintain_decoding", "handle_qte", "recover_target",
    ),
    "kite": ("locate_safe_point", "maintain_distance", "avoid_obstacle", "escape"),
    "rescue": ("search_area", "move_to_target", "rescue_teammate", "escape"),
    "rotate": ("choose_destination", "move_to_target", "avoid_obstacle"),
    "travel": ("move_to_target", "avoid_obstacle", "recover_target"),
    "search": ("search_area", "find_cipher", "recover_target"),
    "gate": ("search_area", "move_to_target", "open_gate", "escape"),
    "idle": ("observe", "none"),
}


def is_valid_subgoal(value: str) -> bool:
    return value in SUBGOAL_NAMES


def subgoals_for_intent(intent: str) -> tuple[str, ...]:
    """Return the v1 candidate set for one top-level intent."""
    return INTENT_SUBGOALS.get(intent, ("none",))

