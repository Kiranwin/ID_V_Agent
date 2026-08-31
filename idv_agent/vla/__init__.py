"""Native VLA data contracts and lightweight utilities.

The VLA path intentionally does not depend on the legacy 20-way action schema.
"""

from .action_chunk import (
    ACTION_CHUNK_HORIZON,
    DEFAULT_ACTION_DELAY_FRAMES,
    HISTORY_FRAMES,
    MACRO_FRAMES,
    NAV_ACTIONS,
    INTERACTION_ACTIONS,
    INTENTS,
    VLA_SCHEMA_VERSION,
    validate_record,
)

__all__ = [
    "ACTION_CHUNK_HORIZON", "HISTORY_FRAMES", "MACRO_FRAMES",
    "DEFAULT_ACTION_DELAY_FRAMES", "NAV_ACTIONS", "INTERACTION_ACTIONS",
    "INTENTS", "VLA_SCHEMA_VERSION",
    "validate_record",
]
