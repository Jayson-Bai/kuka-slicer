"""Symmetric logical-layer progression owned by the surface mapper."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LayerProgression:
    """A flat → curved → flat profile inferred from one user-selected start layer.

    ``surface_return_layer`` is mirrored from ``surface_start_layer`` about the
    last logical layer.  One central layer (odd layer count) or the two central
    layers (even layer count) therefore reach the complete target curvature.
    """

    surface_start_layer: int
    final_logical_layer: int

    def __post_init__(self) -> None:
        if self.surface_start_layer < 0 or self.final_logical_layer < 0:
            raise ValueError("logical layer indices must be non-negative")
        if self.surface_start_layer > self.final_logical_layer // 2:
            raise ValueError(
                "surface_start_layer must leave a symmetric curved region around the middle layer"
            )

    @property
    def surface_return_layer(self) -> int:
        """Last layer of the mirrored curved region; later layers are flat."""

        return self.final_logical_layer - self.surface_start_layer

    @property
    def peak_layers(self) -> tuple[int, ...]:
        """One middle peak layer for odd counts, two for even counts."""

        first = self.surface_start_layer
        last = self.surface_return_layer
        count = last - first + 1
        if count <= 2:
            return tuple(range(first, last + 1))
        midpoint = (first + last) / 2.0
        if count % 2:
            return (int(midpoint),)
        return (int(midpoint - 0.5), int(midpoint + 0.5))

    def alpha(self, logical_layer: int) -> float:
        """Return a smooth, symmetric completion ratio without reading path Z."""

        first = self.surface_start_layer
        last = self.surface_return_layer
        if logical_layer < first or logical_layer > last:
            return 0.0
        active_count = last - first + 1
        if active_count <= 2:
            return 1.0
        distance_from_edge = min(logical_layer - first, last - logical_layer)
        half_transition_steps = (active_count - 1) // 2
        raw = distance_from_edge / half_transition_steps
        return raw * raw * (3.0 - 2.0 * raw)
