"""Independent extrusion compensation for already-mapped source NPZ jobs."""

from .compensator import ExtrusionCompensationResult, compensate_extrusion

__all__ = ["ExtrusionCompensationResult", "compensate_extrusion"]
