"""Public configuration for the independent honeycomb path planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HoneycombTopology = Literal[
    "native_single_motion",
    "closed_minimum_trails",
    "closed_hexagon_double_wall",
]


@dataclass(frozen=True, slots=True)
class HoneycombPathingConfig:
    """Enable wall-centreline planning after the native Prusa slice.

    Native mode preserves every Prusa-produced honeycomb wall exactly as it
    lies in the source STL section.  It keeps only genuine shared endpoints
    continuous and emits all inter-region motion as separate hole-safe travel
    paths; it never fabricates a material bridge merely to obtain one global
    path.  The closed-hexagon double-wall mode remains an explicit topology-
    changing alternative.
    """

    enabled: bool = False
    topology: HoneycombTopology = "closed_minimum_trails"

    def __post_init__(self) -> None:
        if self.topology not in (
            "native_single_motion",
            "closed_minimum_trails",
            "closed_hexagon_double_wall",
        ):
            raise ValueError("unsupported honeycomb topology")
