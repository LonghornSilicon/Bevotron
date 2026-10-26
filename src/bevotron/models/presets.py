"""Named defaults for the architecture knobs. The table itself lives in config.PRESETS
so a config can be validated without importing torch; this module re-exports it."""

from bevotron.config import PRESETS

__all__ = ["PRESETS"]
