"""Material process parameters used to derive extrusion."""

from __future__ import annotations

from dataclasses import dataclass, field
import math


RESIN_FIXED_BEAD_WIDTH_MM = 2.0
RESIN_FILAMENT_DIAMETER_MM = 1.75
RESIN_FILAMENT_LENGTH_PER_MM3 = 1.0 / (
    math.pi * (RESIN_FILAMENT_DIAMETER_MM / 2.0) ** 2
)


@dataclass(frozen=True)
class ResinProcessParams:
    layer_height_mm: float = 0.5
    extrusion_scale: float = 1.0
    feed_mm_s: float = 10.0
    first_layer_feed_mm_s: float = field(default=10.0, kw_only=True)
    temperature_c: float = 250.0
    fan_enabled: bool = True
    prime_length_mm: float = 18.0
    prime_speed_mm_s: float = 15.0
    retract_length_mm: float = 15.0
    retract_speed_mm_s: float = 30.0
    e_per_mm_override: float | None = None

    def e_per_mm(self) -> float:
        if self.e_per_mm_override is not None:
            return float(self.e_per_mm_override)
        return (
            RESIN_FIXED_BEAD_WIDTH_MM
            * float(self.layer_height_mm)
            * float(self.extrusion_scale)
            * RESIN_FILAMENT_LENGTH_PER_MM3
        )


@dataclass(frozen=True)
class FiberProcessParams:
    layer_height_mm: float = 0.1
    extrusion_scale: float = 1.0
    feed_mm_s: float = 10.0
    first_layer_feed_mm_s: float = field(default=10.0, kw_only=True)
    temperature_c: float = 250.0
    fan_enabled: bool = True
    prime_length_mm: float = 12.0
    prime_speed_mm_s: float = 5.0
    retract_length_mm: float = 10.0
    retract_speed_mm_s: float = 5.0
    start_accel_s: float = 2.0

    def e_per_mm(self) -> float:
        # Base 1.0 means fiber feed speed equals TCP movement speed.
        return float(self.extrusion_scale)


@dataclass(frozen=True)
class ProcessParams:
    resin: ResinProcessParams = field(default_factory=ResinProcessParams)
    fiber: FiberProcessParams = field(default_factory=FiberProcessParams)
    travel_feed_mm_s: float = 10.0
    first_layer_travel_feed_mm_s: float = field(default=10.0, kw_only=True)
    default_a: float = 0.0
    default_b: float = 0.0
    default_c: float = 0.0
    start_x_mm: float = 0.0
    start_y_mm: float = 0.0
    primeline_x_mm: float = 0.0
    primeline_y_mm: float = -10.0
    primeline_length_mm: float = 100.0
    prime_settle_s: float = field(default=0.5, kw_only=True)
    dt: float = 0.004
    corner_angle_deg: float = 45.0
    corner_retreat_ratio: float = 0.65
    spline_max_error_mm: float = 0.1
    spline_max_angle_deg: float = 45.0
    source_merge_distance_mm: float = 0.04
    corner_retreat_max_mm: float = 0.4
    corner_blend_segments: int = 8
    density: int = 0
    degree: int = 3
    max_fit_points_per_segment: int = 20000

    def __post_init__(self) -> None:
        first_layer_speeds = {
            "resin.first_layer_feed_mm_s": self.resin.first_layer_feed_mm_s,
            "fiber.first_layer_feed_mm_s": self.fiber.first_layer_feed_mm_s,
            "first_layer_travel_feed_mm_s": self.first_layer_travel_feed_mm_s,
        }
        for name, value in first_layer_speeds.items():
            speed = float(value)
            if not math.isfinite(speed) or speed <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        prime_settle_s = float(self.prime_settle_s)
        if not math.isfinite(prime_settle_s) or prime_settle_s < 0.0:
            raise ValueError("prime_settle_s must be >= 0")

    @property
    def default_abc(self) -> tuple[float, float, float]:
        return (
            float(self.default_a),
            float(self.default_b),
            float(self.default_c),
        )
