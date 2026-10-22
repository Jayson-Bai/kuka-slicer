"""Distance-based resin coasting on final samples, independent of motion planning.

Keep only a trailing window so a stroke may span several fitted curves without
coasting at their internal boundaries. Coordinates, clocks and row schema are
never changed. The exporter carries removed E forward until the next E reset.
"""

from collections import deque
import math


def validate_coasting_distance(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("resin_coast_distance_mm must be finite and >= 0")
    return value


class ResinCoasting:
    """Hold the last L mm; paths <= 2L retain their original extrusion."""

    def __init__(self, distance_mm: float):
        self.distance_mm = validate_coasting_distance(distance_mm)
        self.removed_e = 0.0
        self._tail = deque()
        self._length = 0.0
        self._previous_xyz = None
        self._last_e = None

    def push(self, row):
        xyz = (row.x, row.y, row.z)
        if self._previous_xyz is not None:
            self._length += math.dist(self._previous_xyz, xyz)
        if self._last_e is not None and row.e < self._last_e - 1e-8:
            raise ValueError("resin coasting requires nondecreasing print E")
        self._previous_xyz = xyz
        self._last_e = row.e
        self._tail.append((self._length, row))
        if self._length > 2.0 * self.distance_mm:
            cutoff = self._length - self.distance_mm
            # Keep the left interpolation anchor until the true end is known.
            while len(self._tail) > 1 and self._tail[1][0] <= cutoff:
                yield self._tail.popleft()[1]

    def finish(self):
        if self._length <= 2.0 * self.distance_mm or self.distance_mm == 0.0:
            while self._tail:
                yield self._tail.popleft()[1]
            return
        cutoff = self._length - self.distance_mm
        while len(self._tail) > 1 and self._tail[1][0] <= cutoff:
            yield self._tail.popleft()[1]
        left_s, left = self._tail[0]
        right_s, right = self._tail[1]
        ratio = (cutoff - left_s) / (right_s - left_s)
        stop_e = left.e + (right.e - left.e) * ratio
        self.removed_e = max(0.0, self._last_e - stop_e)
        while self._tail:
            _, row = self._tail.popleft()
            row.e = min(row.e, stop_e)
            yield row
