"""Optional post-Prusa honeycomb centreline planning.

The package deliberately owns no PrusaSlicer bindings.  It converts a sliced
STL cross-section into a centreline graph after the native backend has
finished, so the native perimeter/infill and avoid-crossing implementation is
left untouched.
"""

from .config import HoneycombPathingConfig, HoneycombTopology
from .planner import apply_honeycomb_centerline_pathing, solid_geometry_at_z
from .travel_router import HoleSafeTravelRouter

__all__ = [
    "HoneycombPathingConfig",
    "HoneycombTopology",
    "HoleSafeTravelRouter",
    "apply_honeycomb_centerline_pathing",
    "solid_geometry_at_z",
]
