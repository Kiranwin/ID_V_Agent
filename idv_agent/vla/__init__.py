"""Native VLA data contracts and lightweight utilities.

The VLA path intentionally does not depend on the legacy 20-way action schema.
"""

from .action_chunk import (
    ACTION_CHUNK_HORIZON,
    DEFAULT_ACTION_DELAY_FRAMES,
    HISTORY_FRAMES,
    MACRO_FRAMES,
    MOVE_DIRECTIONS,
    CAMERA_BUCKETS,
    BUTTON_NAMES,
    INTENTS,
    VLA_SCHEMA_VERSION,
    VLA_SCHEMA_VERSION_V4,
    validate_record,
    validate_v4_record,
)

__all__ = [
    "ACTION_CHUNK_HORIZON", "HISTORY_FRAMES", "MACRO_FRAMES",
    "DEFAULT_ACTION_DELAY_FRAMES", "MOVE_DIRECTIONS", "CAMERA_BUCKETS",
    "BUTTON_NAMES", "INTENTS", "VLA_SCHEMA_VERSION", "VLA_SCHEMA_VERSION_V4",
    "validate_record", "validate_v4_record",
]
