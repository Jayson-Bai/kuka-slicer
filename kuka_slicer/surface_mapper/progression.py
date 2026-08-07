"""Logical-layer progression functions owned by the surface mapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProgressionCurve = Literal["linear", "smoothstep"]


@dataclass(frozen=True, slots=True)
class LayerProgression:
    """Map a logical layer key to the target-surface completion ratio ``s_k``."""

    start_logical_layer: int
    end_logical_layer: int
    curve: ProgressionCurve = "smoothstep"

    def __post_init__(self) -> None:
        if self.start_logical_layer < 0 or self.end_logical_layer < 0:
            raise ValueError("logical layer indices must be non-negative")
        if self.end_logical_layer < self.start_logical_layer:
            raise ValueError("end_logical_layer must not precede start_logical_layer")
        if self.curve not in ("linear", "smoothstep"):
            raise ValueError("curve must be linear or smoothstep")

    def alpha(self, logical_layer: int) -> float:
        """Return a deterministic ratio without inspecting geometric Z values."""

        if logical_layer < self.start_logical_layer:
            return 0.0
        if self.start_logical_layer == self.end_logical_layer:
            return 1.0
        if logical_layer >= self.end_logical_layer:
            return 1.0
        raw = (logical_layer - self.start_logical_layer) / (
            self.end_logical_layer - self.start_logical_layer
        )
        if self.curve == "linear":
            return raw
        return raw * raw * (3.0 - 2.0 * raw)
