"""Private course ordering and travel on the established safe boundary network.

No new deposited geometry or Core commands are created here. Production
blocks enter at the nearest corner-adjacent course endpoint, then optimize
the remaining visits including the connection to the next process record.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np


_EPS = 1e-7
_MAX_FREE_COURSES = 18


class BoundaryRouter:
    """Shortest 3-D perimeter arc, retaining authored surface waypoints.

The open working rectangle is excluded: its pores must never be shortcut.
Approach segments are permitted only outside it (the solid grip regions).
"""

    def __init__(self, bounds, boundary_sources, pose_at):
        self.bounds = tuple(float(v) for v in bounds)
        left, lower, right, upper = self.bounds
        self.width, self.height = right - left, upper - lower
        if self.width <= 0 or self.height <= 0:
            raise ValueError("continuous-course travel requires positive work bounds")
        self.perimeter = 2 * (self.width + self.height)
        self.pose_at = pose_at
        parameters = [0., self.width, self.width + self.height, 2 * self.width + self.height]
        for path in boundary_sources:
            for point in path:
                parameter = self.parameter(point[:2])
                if parameter is not None:
                    parameters.append(parameter)
        self.parameters = np.asarray(sorted(set(parameters)), dtype=float)
        self.poses = np.asarray([pose_at(self.point(t)) for t in self.parameters])

    def point(self, t):
        left, lower, right, upper = self.bounds
        t %= self.perimeter
        if t <= self.width:
            return np.array([left + t, lower])
        if t <= self.width + self.height:
            return np.array([right, lower + t - self.width])
        if t <= 2 * self.width + self.height:
            return np.array([right - (t - self.width - self.height), upper])
        return np.array([left, upper - (t - 2 * self.width - self.height)])

    def parameter(self, xy):
        x, y = map(float, xy)
        left, lower, right, upper = self.bounds
        if not left - _EPS <= x <= right + _EPS or not lower - _EPS <= y <= upper + _EPS:
            return None
        if abs(y - lower) <= _EPS:
            return min(self.width, max(0., x - left))
        if abs(x - right) <= _EPS:
            return self.width + min(self.height, max(0., y - lower))
        if abs(y - upper) <= _EPS:
            return self.width + self.height + min(self.width, max(0., right - x))
        if abs(x - left) <= _EPS:
            return (2 * self.width + self.height + min(self.height, max(0., upper - y))) % self.perimeter
        return None

    def crosses_interior(self, first, second):
        """Segment/open-rectangle intersection, including interior endpoints."""
        left, lower, right, upper = self.bounds
        lo, hi = 0., 1.
        for a, b, minimum, maximum in zip(first[:2], second[:2], (left, lower), (right, upper)):
            minimum += _EPS
            maximum -= _EPS
            delta = float(b - a)
            if abs(delta) <= 1e-14:
                if not minimum < a < maximum:
                    return False
            else:
                t0, t1 = sorted(((minimum - a) / delta, (maximum - a) / delta))
                lo, hi = max(lo, t0), min(hi, t1)
                if lo >= hi:
                    return False
        return lo < hi

    def candidates(self, pose):
        parameter = self.parameter(pose[:2])
        if parameter is not None:
            return [(parameter, pose)]
        left, lower, right, upper = self.bounds
        x, y = np.clip(pose[:2], [left, lower], [right, upper])
        candidates = []
        for xy in ([x, lower], [right, y], [x, upper], [left, y]):
            if not self.crosses_interior(pose, xy):
                candidates.append((self.parameter(xy), self.pose_at(np.asarray(xy))))
        if not candidates:
            raise ValueError("continuous-course endpoint has no hole-safe boundary approach")
        return candidates

    def arc(self, start_t, start_pose, end_t, end_pose):
        span = (end_t - start_t) % self.perimeter
        distances = (self.parameters - start_t) % self.perimeter
        indexes = np.flatnonzero((distances > _EPS) & (distances < span - _EPS))
        indexes = indexes[np.argsort(distances[indexes], kind="stable")]
        return [start_pose, *self.poses[indexes], end_pose]

    def route(self, start, end):
        start, end = np.asarray(start), np.asarray(end)
        if start.shape != (6,) or end.shape != (6,) or not np.isfinite([start, end]).all():
            raise ValueError("continuous-course travel requires finite XYZABC endpoints")
        best = None
        for a, pa in self.candidates(start):
            for b, pb in self.candidates(end):
                for arc in (self.arc(a, pa, b, pb), list(reversed(self.arc(b, pb, a, pa)))):
                    points = [start]
                    for pose in [*arc, end]:
                        # Preserve orientation-only changes as well as motion.
                        if not np.array_equal(pose, points[-1]):
                            points.append(pose)
                    if len(points) == 1:
                        points.append(end)
                    route = np.asarray(points, dtype=float)
                    if any(self.crosses_interior(p, q) for p, q in zip(route, route[1:])):
                        continue
                    length = float(np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1).sum())
                    if best is None or length < best[0] - 1e-10:
                        best = length, route
        if best is None:
            raise ValueError("cannot connect continuous courses without crossing the working-region pores")
        return best[1]


def route_length(route):
    return float(np.linalg.norm(np.diff(route[:, :3], axis=0), axis=1).sum())


def _cost(order, costs):
    return sum(costs[a, b] for a, b in zip(order, order[1:]))


@lru_cache(maxsize=16)
def _solve_cached(count, cost_bytes, first_state=0, exit_bytes=None, fixed_last=True):
    costs = np.frombuffer(cost_bytes, dtype=np.float64).reshape(2 * count, 2 * count)
    exits = np.zeros(2 * count) if exit_bytes is None else np.frombuffer(exit_bytes, dtype=np.float64)
    last = (2 * (count - 1),) if fixed_last and count > 1 else ()
    free_indices = [i for i in range(count) if i != first_state // 2 and 2 * i not in last]
    baseline = (first_state, *(2 * i for i in free_indices), *last)
    free = len(free_indices)
    def total(order):
        return _cost(order, costs) + exits[order[-1]]
    if free == 0:
        return baseline, True
    if free > _MAX_FREE_COURSES:
        # A bounded fallback must not claim proof of optimality. Preserve the
        # original route unless an actual-cost nearest-neighbour tour improves it.
        order, remaining = [first_state], set(free_indices)
        while remaining:
            next_state = min((2 * i + d for i in remaining for d in (0, 1)), key=lambda s: (costs[order[-1], s], s))
            order.append(next_state)
            remaining.remove(next_state // 2)
        order.extend(last)
        return (tuple(order) if total(order) < total(baseline) - 1e-9 else baseline), False
    # Held-Karp over (visited courses, last course, last direction). A reversed
    # state changes endpoints only; the deposited stroke is never re-sampled.
    states = np.asarray([2 * i + d for i in free_indices for d in (0, 1)])
    size = 1 << free
    dp = np.full((size, 2 * free), math.inf)
    parent = np.full((size, 2 * free), -1, dtype=np.int16)
    for i in range(free):
        dp[1 << i, 2 * i:2 * i + 2] = costs[first_state, states[2 * i:2 * i + 2]]
    for mask in range(1, size):
        remaining = (size - 1) ^ mask
        while remaining:
            bit = remaining & -remaining
            remaining ^= bit
            i = bit.bit_length() - 1
            destinations = [2 * i, 2 * i + 1]
            values = dp[mask, :, None] + costs[states[:, None], states[destinations]]
            predecessors = np.argmin(values, axis=0)
            candidate = values[predecessors, np.arange(2)]
            next_mask = mask | bit
            for direction, destination in enumerate(destinations):
                if candidate[direction] < dp[next_mask, destination]:
                    dp[next_mask, destination] = candidate[direction]
                    parent[next_mask, destination] = predecessors[direction]
    final = dp[-1] + (costs[states, last[0]] + exits[last[0]] if last else exits[states])
    state, mask = int(np.argmin(final)), size - 1
    order = list(last)
    while mask:
        order.append(int(states[state]))
        previous = int(parent[mask, state])
        mask ^= 1 << (state // 2)
        state = previous
    order.append(first_state)
    order.reverse()
    # Retain established tie ordering and avoid floating-point-only changes.
    return (tuple(order) if total(order) < total(baseline) - 1e-9 else baseline), True


def optimize_block(paths, router, *, entry=None, exit=None):
    """Optimize safe travel; an explicit entry enables side-boundary starts.

    The entry requirement precedes travel minimization: find the endpoints
    nearest the four boundary corners, then choose the one nearest the actual
    entry by safe 3-D route length. All other visits/directions and the exit are
    optimized. Without an entry, retain the legacy fixed-anchor contract.
    """
    count = len(paths)
    if count == 0:
        return [(i, False) for i in range(count)], {"optimal": True, "before_mm": 0., "after_mm": 0.}
    costs = np.full((2 * count, 2 * count), math.inf)
    for i, first in enumerate(paths):
        for j, second in enumerate(paths):
            if i == j:
                continue
            for a in (0, 1):
                for b in (0, 1):
                    costs[2 * i + a, 2 * j + b] = route_length(router.route(first[0 if a else -1], second[-1 if b else 0]))
    entries, exits = np.zeros(2 * count), np.zeros(2 * count)
    for i, path in enumerate(paths):
        for direction in (0, 1):
            if entry is not None:
                entries[2 * i + direction] = route_length(router.route(entry, path[-1 if direction else 0]))
            if exit is not None:
                exits[2 * i + direction] = route_length(router.route(path[0 if direction else -1], exit))
    first_state = 0
    if entry is not None:
        left, lower, right, upper = router.bounds
        candidates = set()
        for corner_xy in ((left, lower), (right, lower), (right, upper), (left, upper)):
            corner = router.pose_at(np.asarray(corner_xy))
            distances = np.asarray([
                route_length(router.route(corner, p[-1 if d else 0]))
                for p in paths for d in (0, 1)
            ])
            candidates.update(np.flatnonzero(distances <= distances.min() + _EPS).tolist())
        first_state = min(candidates, key=lambda state: (entries[state], state))
    order, exact = _solve_cached(count, costs.tobytes(), first_state, exits.tobytes(), entry is None)
    def total(visits):
        return entries[visits[0]] + _cost(visits, costs) + exits[visits[-1]]
    return [(state // 2, bool(state % 2)) for state in order], {
        "optimal": exact,
        "method": "exact_dynamic_programming" if exact else "bounded_nearest_neighbour",
        "constraints": ("nearest reachable corner-adjacent endpoint; safe boundary network; process exit included"
                        if entry is not None else "fixed first/last course and direction; safe boundary network"),
        "course_count": count,
        "before_mm": float(total(tuple(2 * i for i in range(count)))),
        "after_mm": float(total(order)),
        "entry_mm": float(entries[order[0]]),
        "exit_mm": float(exits[order[-1]]),
    }
