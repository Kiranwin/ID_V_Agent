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
    APPROACH_CIPHER = 1
    START_DECODING = 2
    HANDLE_QTE = 3
    LOCATE_SAFE_POINT = 4
    MAINTAIN_DISTANCE = 5
    VAULT_WINDOW = 6
    DROP_PALLET = 7
    DISENGAGE = 8
    APPROACH_CHAIR = 9
    SEARCH_AREA = 10
    RESCUE_TEAMMATE = 11
    HEAL_TEAMMATE = 12
    WAIT_OPPORTUNITY = 13
    MOVE_TO_ZONE = 14
    FIND_CIPHER = 15
    MOVE_TO_TARGET = 16
    AVOID_OBSTACLE = 17
    FIND_CHEST = 18
    OPEN_CHEST = 19
    PICKUP_ITEM = 20
    MOVE_TO_GATE = 21
    OPEN_GATE = 22
    ESCAPE = 23
    OBSERVE = 24
    HIDE = 25


SUBGOAL_NAMES = tuple(item.name.lower() for item in SubgoalCategory)

# Every top-level intent from intents.md has at least one legal subgoal.  The
# mapping is intentionally many-to-one: v1 is meant to be rule-derived, not a
# second annotation task.
INTENT_SUBGOALS: dict[str, tuple[str, ...]] = {
    "decipher": ("approach_cipher", "start_decoding", "handle_qte"),
    "kite": ("locate_safe_point", "maintain_distance", "vault_window", "drop_pallet", "disengage"),
    "rescue": ("approach_chair", "search_area", "rescue_teammate", "heal_teammate", "wait_opportunity"),
    "rotate": ("disengage", "move_to_zone"),
    "travel": ("find_cipher", "move_to_target", "avoid_obstacle"),
    "search": ("find_chest", "open_chest", "pickup_item"),
    "gate": ("move_to_gate", "open_gate", "escape"),
    "idle": ("observe", "hide"),
}


def is_valid_subgoal(value: str) -> bool:
    return value in SUBGOAL_NAMES


def subgoals_for_intent(intent: str) -> tuple[str, ...]:
    """Return the v1 candidate set for one top-level intent."""
    return INTENT_SUBGOALS.get(intent, ("none",))
