"""Reusable hole-safe travel routing for post-slice path planners.

PrusaSlicer's avoid-crossing implementation is used while it owns the full
perimeter plan.  Once a planner derives a new set of paths (such as the
honeycomb centreline graph), its endpoints are not accepted by the native
bridge.  This module is the equivalent endpoint-routing layer for those
derived paths: every returned travel stays out of internal voids, and the
shortest valid direct route is always preferred.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from typing import TypeAlias

from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union
from shapely.prepared import prep


Point2D: TypeAlias = tuple[float, float]
RoutingGraph: TypeAlias = Mapping[Point2D, list[tuple[Point2D, float]]]


class HoleSafeTravelRouter:
    """Route endpoint-to-endpoint travel without entering an internal hole.

    The router first uses the straight segment when it is completely safe; it
    is the geometric minimum.  If a void blocks that segment, it uses an exact
    shortest path on the supplied wall graph.  A solid-only grid is retained
    as a final fallback for disconnected graph fragments.  All returned paths
    are checked against the same internal-hole geometry.
    """

    def __init__(self, solid, wall_graph: RoutingGraph, *, spacing_mm: float):
        self._solid = solid
        self._wall_graph = wall_graph
        self._spacing_mm = float(spacing_mm)
        self._grid: _SolidGridRouter | None = None
        polygons = list(_polygons(solid))
        holes = [Polygon(ring) for polygon in polygons for ring in polygon.interiors]
        self._forbidden = unary_union(holes).buffer(-1e-4) if holes else Polygon()
        self._prepared_forbidden = prep(self._forbidden)
        self._shortest_trees: dict[
            Point2D, tuple[dict[Point2D, float], dict[Point2D, Point2D]]
        ] = {}
        self._routes: dict[tuple[Point2D, Point2D], list[Point2D] | None] = {}

    def route(self, start: Point2D, end: Point2D) -> list[Point2D] | None:
        """Return the shortest available hole-safe connector, or ``None``.

        Results are cached in both directions.  This is important when a
        local route-order improvement reverses an otherwise unchanged chain.
        """

        start = _point(start)
        end = _point(end)
        key = (start, end)
        if key in self._routes:
            cached = self._routes[key]
            return None if cached is None else list(cached)

        route = self._route_uncached(start, end)
        if route is not None and not self.allows(LineString(route)):
            raise RuntimeError("hole-safe router generated a route through an internal void")
        self._routes[key] = route
        self._routes[(end, start)] = None if route is None else list(reversed(route))
        return None if route is None else list(route)

    def allows(self, geometry) -> bool:
        """Return whether a geometry stays outside every internal void."""

        if self._forbidden.is_empty:
            return True
        if geometry.geom_type == "Point":
            return not self._prepared_forbidden.contains(geometry)
        if not self._prepared_forbidden.intersects(geometry):
            return True
        intersection = geometry.intersection(self._forbidden)
        return intersection.is_empty or float(intersection.length) <= 1e-8

    def _grid_router(self) -> "_SolidGridRouter":
        # Most Prusa-derived paths can be connected through the supplied
        # no-hole routing graph.  Constructing a dense fallback grid before it
        # is actually required is disproportionately expensive on large
        # honeycombs, so retain it as a true last resort.
        if self._grid is None:
            self._grid = _SolidGridRouter(self._solid, spacing_mm=self._spacing_mm)
        return self._grid

    def _route_uncached(self, start: Point2D, end: Point2D) -> list[Point2D] | None:
        if start == end:
            return [start, end]
        direct = LineString((start, end))
        if self.allows(direct):
            return [start, end]

        if start in self._wall_graph and end in self._wall_graph:
            distances, parents = self._shortest_tree(start)
            if end in distances:
                return _path_from_tree(start, end, parents)
            return self._grid_router().route(start, end)

        starts = [
            (math.dist(start, node), node)
            for node in self._wall_graph
            if self._solid.covers(LineString((start, node)))
        ]
        ends = [
            (math.dist(end, node), node)
            for node in self._wall_graph
            if self._solid.covers(LineString((end, node)))
        ]
        if starts and ends:
            best = None
            for start_cost, start_node in sorted(starts)[:12]:
                distances, parents = self._shortest_tree(start_node)
                for end_cost, end_node in sorted(ends)[:12]:
                    if end_node not in distances:
                        continue
                    total = start_cost + distances[end_node] + end_cost
                    if best is None or total < best[0]:
                        best = (total, start_node, end_node, parents)
            if best is not None:
                _cost, start_node, end_node, parents = best
                return [start, *_path_from_tree(start_node, end_node, parents), end]
        return self._grid_router().route(start, end)

    def _shortest_tree(self, start: Point2D):
        cached = self._shortest_trees.get(start)
        if cached is None:
            cached = _dijkstra(self._wall_graph, start)
            self._shortest_trees[start] = cached
        return cached


class _SolidGridRouter:
    """Shortest 8-neighbour fallback that never enters an internal hole."""

    def __init__(self, solid, *, spacing_mm: float):
        self.spacing = max(float(spacing_mm), 0.25)
        polygons = list(_polygons(solid))
        holes = [Polygon(ring) for polygon in polygons for ring in polygon.interiors]
        # Voronoi clipping ends exactly on void boundaries.  Retain a tiny
        # numerical allowance so rounding a valid endpoint never makes the
        # router falsely treat it as being inside a hole.
        self.forbidden = unary_union(holes).buffer(-1e-4) if holes else Polygon()
        # Grid construction probes thousands of points.  A prepared predicate
        # has the same containment semantics for those point probes without
        # repeatedly constructing a full geometry intersection.
        self._prepared_forbidden = prep(self.forbidden)
        min_x, min_y, max_x, max_y = solid.bounds
        min_x -= self.spacing * 2.0
        min_y -= self.spacing * 2.0
        max_x += self.spacing * 2.0
        max_y += self.spacing * 2.0
        self.origin = (float(min_x), float(min_y))
        self.nodes: dict[tuple[int, int], Point2D] = {}
        width = int(math.floor((max_x - min_x) / self.spacing)) + 1
        height = int(math.floor((max_y - min_y) / self.spacing)) + 1
        for ix in range(width + 1):
            x = min_x + ix * self.spacing
            for iy in range(height + 1):
                y = min_y + iy * self.spacing
                if self.allows(Point(x, y)):
                    self.nodes[(ix, iy)] = (x, y)

    def route(self, start: Point2D, end: Point2D) -> list[Point2D] | None:
        first = self._nearest_visible(start)
        last = self._nearest_visible(end)
        if first is None or last is None:
            return None
        queue = [(0.0, first)]
        distance = {first: 0.0}
        parent = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if cost != distance.get(node):
                continue
            if node == last:
                break
            for other in self._neighbors(node):
                next_cost = cost + math.dist(self.nodes[node], self.nodes[other])
                if next_cost < distance.get(other, float("inf")):
                    distance[other] = next_cost
                    parent[other] = node
                    heapq.heappush(queue, (next_cost, other))
        if last not in distance:
            return None
        middle = [last]
        while middle[-1] != first:
            middle.append(parent[middle[-1]])
        middle.reverse()
        points = [start, *[self.nodes[node] for node in middle], end]
        return self._simplify(points)

    def _nearest_visible(self, point: Point2D):
        candidates = sorted(
            ((math.dist(point, value), key) for key, value in self.nodes.items()),
            key=lambda item: item[0],
        )
        for _distance, key in candidates[:128]:
            if self.allows(LineString((point, self.nodes[key]))):
                return key
        return None

    def _neighbors(self, node):
        ix, iy = node
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if not dx and not dy:
                    continue
                other = (ix + dx, iy + dy)
                if other in self.nodes and self.allows(LineString((self.nodes[node], self.nodes[other]))):
                    yield other

    def _simplify(self, points: list[Point2D]) -> list[Point2D]:
        kept = [points[0]]
        index = 0
        while index < len(points) - 1:
            next_index = len(points) - 1
            while next_index > index + 1 and not self.allows(LineString((points[index], points[next_index]))):
                next_index -= 1
            kept.append(points[next_index])
            index = next_index
        return kept

    def allows(self, geometry) -> bool:
        if self.forbidden.is_empty:
            return True
        # Touching a void boundary at a path endpoint is valid.  Any positive
        # length inside a hole is forbidden; ``crosses`` alone is insufficient
        # here because GEOS may classify a multi-segment route as neither a
        # simple crossing nor fully within the hole.
        if geometry.geom_type == "Point":
            return not self._prepared_forbidden.contains(geometry)
        if not self._prepared_forbidden.intersects(geometry):
            return True
        intersection = geometry.intersection(self.forbidden)
        return intersection.is_empty or float(intersection.length) <= 1e-8


def route_length(route: list[Point2D]) -> float:
    """Return polyline length in millimetres."""

    return sum(math.dist(first, second) for first, second in zip(route, route[1:]))


def _dijkstra(graph: RoutingGraph, start: Point2D):
    distances = {start: 0.0}
    parents: dict[Point2D, Point2D] = {}
    queue = [(0.0, start)]
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distances.get(node):
            continue
        for other, weight in graph[node]:
            next_cost = cost + weight
            if next_cost < distances.get(other, float("inf")):
                distances[other] = next_cost
                parents[other] = node
                heapq.heappush(queue, (next_cost, other))
    return distances, parents


def _path_from_tree(start: Point2D, end: Point2D, parents: Mapping[Point2D, Point2D]) -> list[Point2D]:
    points = [end]
    while points[-1] != start:
        points.append(parents[points[-1]])
    points.reverse()
    return points


def _polygons(geometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif hasattr(geometry, "geoms"):
        for item in geometry.geoms:
            yield from _polygons(item)


def _point(value) -> Point2D:
    return (float(value[0]), float(value[1]))
