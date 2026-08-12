"""Public configuration for the independent honeycomb path planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


HoneycombTopology = Literal[
    "macro_partition_zero_e",
]


@dataclass(frozen=True, slots=True)
class HoneycombPathingConfig:
    """Enable wall-centreline planning after the native Prusa slice.

    The STL section defines the honeycomb wall graph.  Every wall edge is
    deposited once, and bounded multi-start ordering packs those trails into
    the fewest tested macro execution partitions.  Safe transitions inside a
    partition remain in the material path with unchanged cumulative E rather
    than being classified as source travel paths.
    """

    enabled: bool = False
    topology: HoneycombTopology = "macro_partition_zero_e"

    def __post_init__(self) -> None:
        if self.topology != "macro_partition_zero_e":
            raise ValueError("unsupported honeycomb topology")
