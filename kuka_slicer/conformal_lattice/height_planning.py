"""Small shared rounding policy for target physical part heights."""

import math


def nearest_resin_layer_count(target_mm: float, layer_mm: float) -> int:
    """Choose whole layers, with at least one layer and lower-stack ties."""
    if not math.isfinite(target_mm) or target_mm <= 0:
        raise ValueError("target height must be positive and finite")
    if not math.isfinite(layer_mm) or layer_mm <= 0:
        raise ValueError("resin layer height must be positive and finite")
    return max(1, math.ceil(target_mm / layer_mm - 0.5 - 1e-12))
