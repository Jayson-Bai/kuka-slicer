"""Material-neutral policy for placing fiber between flat resin layers.

The policy deliberately knows nothing about PrusaSlicer, conformal geometry,
path generation, or Core export.  Callers provide their ordered resin-layer
identifiers and adapt the returned interfaces to their own path representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


FLAT_RESIN_INTERLAYER_POLICY_SOURCE = "flat_resin_interlayer_policy_v2"
DEFAULT_INITIAL_RESIN_ONLY_LAYERS = 1


@dataclass(frozen=True, slots=True)
class FiberInterlayerSchedule:
    """Selected interfaces after resin layers, excluding the top cap."""

    resin_layer_indices: tuple[int, ...]
    after_resin_layer_indices: tuple[int, ...]
    skipped_initial_interface_count: int
    source: str = FLAT_RESIN_INTERLAYER_POLICY_SOURCE

    def insertion_count_before(self, resin_layer_index: int) -> int:
        """Return how many selected fiber layers sit below a resin layer."""

        return sum(
            interface_index < int(resin_layer_index)
            for interface_index in self.after_resin_layer_indices
        )

    @property
    def physical_interface_window(self) -> tuple[int, int] | None:
        """Return the inclusive one-based interface window used by Core paths."""

        if not self.after_resin_layer_indices:
            return None
        return (
            self.after_resin_layer_indices[0] + 1,
            self.after_resin_layer_indices[-1] + 1,
        )


def plan_flat_resin_interlayers(
    resin_layer_indices: Iterable[int],
    *,
    skip_initial_interfaces: int = DEFAULT_INITIAL_RESIN_ONLY_LAYERS,
) -> FiberInterlayerSchedule:
    """Select flat resin interfaces after a resin-only first layer and before a top cap.

    By default physical layer 1 is resin-only: the first fiber is emitted
    after physical resin layer 2. ``skip_initial_interfaces`` can reserve a
    longer process-owned prefix without referring to any slicing backend.
    """

    ordered = tuple(sorted(int(index) for index in resin_layer_indices))
    if len(set(ordered)) != len(ordered):
        raise ValueError("resin layer indices must be unique")
    if skip_initial_interfaces < 0:
        raise ValueError("skip_initial_interfaces must be non-negative")

    eligible = ordered[:-1]
    selected = eligible[int(skip_initial_interfaces) :]
    return FiberInterlayerSchedule(
        resin_layer_indices=ordered,
        after_resin_layer_indices=selected,
        skipped_initial_interface_count=min(int(skip_initial_interfaces), len(eligible)),
    )
