"""Isolated one-layer honeycomb partitioning prototype.

This module is deliberately not imported by the production slicer.  It makes
the topology constraint visible on a small regular honeycomb: every coloured
partition is one continuous deposited trail, every wall is assigned once, and
the dashed links are shortest graph-safe travels between partitions.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import heapq
import math
from pathlib import Path
from xml.sax.saxutils import escape


Point = tuple[float, float]
Edge = tuple[int, int]


@dataclass(frozen=True, slots=True)
class HoneycombPartitionPrototype:
    """A small edge-once partition plan for one idealized honeycomb layer."""

    points: tuple[Point, ...]
    edges: tuple[Edge, ...]
    partitions: tuple[tuple[int, ...], ...]
    transfers: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MacroPartition:
    """One spatial work zone, containing several deposited sub-trails."""

    grid_cell: tuple[int, int]
    deposited_trails: tuple[tuple[int, ...], ...]
    intra_partition_travels: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class MacroPartitionPrototype:
    """A few spatial zones with internal no-hole travel between sub-trails."""

    points: tuple[Point, ...]
    edges: tuple[Edge, ...]
    partitions: tuple[MacroPartition, ...]
    between_partition_travels: tuple[tuple[int, ...], ...]


def build_honeycomb_partition_prototype(
    *, columns: int = 3, rows: int = 2, side_mm: float = 10.0
) -> HoneycombPartitionPrototype:
    """Build an edge-once, one-stroke partition demo for one honeycomb layer.

    The partition count is not selected beforehand.  It is the minimum number
    of Euler trails required by the degree-three honeycomb graph.  This is a
    deliberately small diagnostic case, not production slicing logic.
    """

    if columns < 1 or rows < 1 or side_mm <= 0:
        raise ValueError("columns, rows, and side_mm must be positive")
    point_ids: dict[Point, int] = {}
    raw_edges: set[Edge] = set()

    def point_id(point: Point) -> int:
        key = (round(point[0], 8), round(point[1], 8))
        if key not in point_ids:
            point_ids[key] = len(point_ids)
        return point_ids[key]

    for row in range(rows):
        for column in range(columns):
            center = (
                math.sqrt(3.0) * side_mm * (column + 0.5 * (row % 2)),
                1.5 * side_mm * row,
            )
            vertices = [
                (
                    center[0] + side_mm * math.cos(math.pi / 6.0 + index * math.pi / 3.0),
                    center[1] + side_mm * math.sin(math.pi / 6.0 + index * math.pi / 3.0),
                )
                for index in range(6)
            ]
            ids = [point_id(point) for point in vertices]
            raw_edges.update(_edge_key(start, end) for start, end in zip(ids, ids[1:] + ids[:1]))

    points = tuple(point for point, _index in sorted(point_ids.items(), key=lambda item: item[1]))
    edges = tuple(sorted(raw_edges))
    return build_honeycomb_partition_from_graph(points, edges)


def build_honeycomb_partition_from_graph(
    points: tuple[Point, ...], edges: tuple[Edge, ...]
) -> HoneycombPartitionPrototype:
    """Partition any one-layer honeycomb wall graph for diagnostic rendering."""

    if not points or not edges:
        raise ValueError("prototype graph needs at least one point and one wall edge")
    if any(left == right or left < 0 or right < 0 or left >= len(points) or right >= len(points) for left, right in edges):
        raise ValueError("prototype graph contains an invalid wall edge")
    if len(set(edges)) != len(edges):
        raise ValueError("prototype graph must contain unique undirected wall edges")
    partitions = tuple(tuple(trail) for trail in _minimum_edge_once_trails(points, edges))
    transfers = tuple(
        _shortest_wall_safe_route(points, edges, left[-1], right[0])
        for left, right in zip(partitions, partitions[1:])
    )
    return HoneycombPartitionPrototype(points, edges, partitions, transfers)


def build_macro_partition_prototype(
    points: tuple[Point, ...],
    edges: tuple[Edge, ...],
    *,
    columns: int = 4,
    rows: int = 3,
) -> MacroPartitionPrototype:
    """Group edge-once trails into a few spatial execution partitions.

    Unlike :func:`build_honeycomb_partition_from_graph`, this prototype makes
    each grid zone one complete execution task.  It permits a shortest route
    along the wall graph between deposited sub-trails *inside* a zone.  Those
    links are geometry-only here; they are deliberately not yet converted to
    E=constant segments or connected to the production slicer.
    """

    if columns < 1 or rows < 1:
        raise ValueError("macro partition rows and columns must be positive")
    atomic = build_honeycomb_partition_from_graph(points, edges).partitions
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    grouped: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
    for trail in atomic:
        center_x = sum(points[node][0] for node in trail) / len(trail)
        center_y = sum(points[node][1] for node in trail) / len(trail)
        column = min(columns - 1, int((center_x - min_x) / width * columns))
        row = min(rows - 1, int((center_y - min_y) / height * rows))
        grouped[(column, row)].append(trail)

    # Serpentine ordering keeps successive macro zones spatially adjacent.
    zone_order = [
        (column, row)
        for row in range(rows)
        for column in (range(columns) if row % 2 == 0 else range(columns - 1, -1, -1))
        if grouped[(column, row)]
    ]
    partitions: list[MacroPartition] = []
    between: list[tuple[int, ...]] = []
    current = min(range(len(points)), key=lambda node: (points[node][1], points[node][0]))
    for grid_cell in zone_order:
        trails, initial_travel, inner_travels = _order_deposit_trails(
            points,
            edges,
            grouped[grid_cell],
            current,
        )
        if partitions:
            between.append(initial_travel)
        partitions.append(MacroPartition(grid_cell, tuple(trails), tuple(inner_travels)))
        current = trails[-1][-1]
    return MacroPartitionPrototype(points, edges, tuple(partitions), tuple(between))


def render_macro_partition_svg(
    plan: MacroPartitionPrototype,
    output_path: str | Path,
) -> Path:
    """Render macro zones: solid deposit, coloured dash internal, dark dash between."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x_values = [point[0] for point in plan.points]
    y_values = [point[1] for point in plan.points]
    margin, footer = 56.0, 44.0
    scale = min(900.0 / max(max(x_values) - min(x_values), 1.0), 560.0 / max(max(y_values) - min(y_values), 1.0))
    width = int((max(x_values) - min(x_values)) * scale + margin * 2)
    height = int((max(y_values) - min(y_values)) * scale + margin * 2 + footer)

    def project(node: int) -> Point:
        x, y = plan.points[node]
        return margin + (x - min(x_values)) * scale, margin + (max(y_values) - y) * scale

    def line(nodes: tuple[int, ...]) -> str:
        return ' '.join(f'{x:.2f},{y:.2f}' for x, y in (project(node) for node in nodes))

    palette = _partition_palette(len(plan.partitions))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#fbfdff"/>',
        '<text x="24" y="28" font-family="Microsoft YaHei, sans-serif" font-size="18" font-weight="700" fill="#172033">蜂窝宏观分区路径原型</text>',
        '<text x="24" y="50" font-family="Microsoft YaHei, sans-serif" font-size="12" fill="#52606d">同色实线：区内不重走沉积；同色虚线：区内安全换段；深灰虚线：分区间安全空走</text>',
    ]
    for start, end in plan.edges:
        lines.append(f'<path d="M {project(start)[0]:.2f} {project(start)[1]:.2f} L {project(end)[0]:.2f} {project(end)[1]:.2f}" stroke="#d8e0eb" stroke-width="1.2" fill="none"/>')
    for route in plan.between_partition_travels:
        lines.append(f'<polyline points="{line(route)}" stroke="#334155" stroke-width="1.2" stroke-dasharray="5 4" fill="none" opacity="0.72"/>')
    for index, partition in enumerate(plan.partitions, start=1):
        color = palette[index - 1]
        for route in partition.intra_partition_travels:
            lines.append(f'<polyline points="{line(route)}" stroke="{color}" stroke-width="1.8" stroke-dasharray="5 3" fill="none" opacity="0.82"/>')
        for trail in partition.deposited_trails:
            lines.append(f'<polyline points="{line(trail)}" stroke="{color}" stroke-width="3.3" stroke-linejoin="round" stroke-linecap="round" fill="none" marker-end="url(#arrow)"/>')
        nodes = [node for trail in partition.deposited_trails for node in trail]
        center_x = sum(plan.points[node][0] for node in nodes) / len(nodes)
        center_y = sum(plan.points[node][1] for node in nodes) / len(nodes)
        label_x = margin + (center_x - min(x_values)) * scale
        label_y = margin + (max(y_values) - center_y) * scale
        lines.append(f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Microsoft YaHei, sans-serif" font-size="13" font-weight="700" fill="{color}" stroke="#fbfdff" stroke-width="3" paint-order="stroke">分区 {index}</text>')
    lines.append(f'<text x="24" y="{height - 18}" font-family="Microsoft YaHei, sans-serif" font-size="11" fill="#475569">{len(plan.partitions)} 个宏观分区；每条蜂窝墙边仅在实线沉积段出现一次。</text>')
    lines.append('</svg>')
    output.write_text('\n'.join(lines), encoding='utf-8')
    return output


def render_honeycomb_partition_svg(
    plan: HoneycombPartitionPrototype,
    output_path: str | Path,
    *,
    show_partition_labels: bool = True,
    show_legend: bool = True,
) -> Path:
    """Render a compact coloured plan view for review without extra packages."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    x_values = [point[0] for point in plan.points]
    y_values = [point[1] for point in plan.points]
    margin = 56.0
    scale = min(760.0 / max(max(x_values) - min(x_values), 1.0), 420.0 / max(max(y_values) - min(y_values), 1.0))
    width = int((max(x_values) - min(x_values)) * scale + margin * 2)
    height = int((max(y_values) - min(y_values)) * scale + margin * 2 + 86)

    def project(node: int) -> Point:
        x, y = plan.points[node]
        return (
            margin + (x - min(x_values)) * scale,
            margin + (max(y_values) - y) * scale,
        )

    palette = _partition_palette(len(plan.partitions))
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="context-stroke"/></marker></defs>',
        f'<rect width="{width}" height="{height}" fill="#fbfdff"/>',
        '<text x="24" y="28" font-family="Microsoft YaHei, sans-serif" font-size="18" font-weight="700" fill="#172033">最小单层蜂窝：分区一笔画原型</text>',
        '<text x="24" y="50" font-family="Microsoft YaHei, sans-serif" font-size="12" fill="#52606d">实线：不重复沉积墙边；虚线：分区间沿蜂窝墙图的最短安全空走</text>',
    ]
    for start, end in plan.edges:
        first, second = project(start), project(end)
        lines.append(f'<path d="M {first[0]:.2f} {first[1]:.2f} L {second[0]:.2f} {second[1]:.2f}" stroke="#cbd5e1" stroke-width="2" fill="none"/>')
    for transfer in plan.transfers:
        coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x, y in (project(node) for node in transfer))
        lines.append(f'<polyline points="{coordinates}" stroke="#475569" stroke-width="1.5" stroke-dasharray="5 4" fill="none" marker-end="url(#arrow)" opacity="0.82"/>')
    for index, trail in enumerate(plan.partitions, start=1):
        color = palette[index - 1]
        coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x, y in (project(node) for node in trail))
        lines.append(f'<polyline points="{coordinates}" stroke="{color}" stroke-width="4.2" stroke-linejoin="round" stroke-linecap="round" fill="none" marker-end="url(#arrow)"/>')
        center_x = sum(plan.points[node][0] for node in trail) / len(trail)
        center_y = sum(plan.points[node][1] for node in trail) / len(trail)
        label_x = margin + (center_x - min(x_values)) * scale
        label_y = margin + (max(y_values) - center_y) * scale
        if show_partition_labels:
            lines.append(f'<text x="{label_x:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Microsoft YaHei, sans-serif" font-size="11" font-weight="700" fill="{color}" stroke="#fbfdff" stroke-width="3" paint-order="stroke">分区 {index}</text>')
    if show_legend:
        legend_y = height - 28
        legend = '；'.join(f'分区 {index}: {color}' for index, color in enumerate(palette, start=1))
        lines.append(f'<text x="24" y="{legend_y}" font-family="Consolas, Microsoft YaHei, sans-serif" font-size="10" fill="#475569">{escape(legend)}</text>')
    lines.append('</svg>')
    output.write_text('\n'.join(lines), encoding='utf-8')
    return output


def _minimum_edge_once_trails(points: tuple[Point, ...], edges: tuple[Edge, ...]) -> list[list[int]]:
    """Split an undirected graph into the fewest continuous, edge-once trails."""

    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge_id, (left, right) in enumerate(edges):
        adjacency[left].append(edge_id)
        adjacency[right].append(edge_id)
    odd_nodes = sorted(node for node, incident in adjacency.items() if len(incident) % 2)
    pairs = _choose_balanced_odd_pairing(points, edges, odd_nodes, min(adjacency))
    return _trails_from_virtual_pairs(edges, pairs, min(adjacency))


def _trails_from_virtual_pairs(
    edges: tuple[Edge, ...], pairs: tuple[Edge, ...], start: int
) -> list[list[int]]:
    """Break an augmented Euler circuit at its virtual, non-material edges."""

    augmented = [*edges, *pairs]
    circuit = _euler_circuit(augmented, start)
    trails: list[list[int]] = []
    current: list[int] = []
    for start, end, edge_id in circuit:
        if edge_id >= len(edges):
            if len(current) > 1:
                trails.append(current)
            current = []
            continue
        if not current:
            current = [start]
        current.append(end)
    if len(current) > 1:
        trails.append(current)
    return trails


def _choose_balanced_odd_pairing(
    points: tuple[Point, ...],
    edges: tuple[Edge, ...],
    odd_nodes: list[int],
    start: int,
) -> tuple[Edge, ...]:
    """Choose a visually balanced virtual pairing for this tiny diagnostic.

    A virtual pair merely marks where one continuous deposited trail ends and
    the next begins.  For the small prototype graph we can enumerate pairings
    and avoid a misleading visual where one partition absorbs almost all
    walls while the rest are one-edge fragments.
    """

    if len(odd_nodes) > 12:
        return tuple(_pair_nearest_odds(points, odd_nodes))
    best: tuple[tuple[float, ...], tuple[Edge, ...]] | None = None
    for pairs in _all_odd_pairings(tuple(odd_nodes)):
        trails = _trails_from_virtual_pairs(edges, pairs, start)
        counts = sorted(len(trail) - 1 for trail in trails)
        areas = []
        for trail in trails:
            xs = [points[node][0] for node in trail]
            ys = [points[node][1] for node in trail]
            areas.append((max(xs) - min(xs)) * (max(ys) - min(ys)))
        # Prefer a useful minimum stroke length, then smaller imbalance and
        # finally spatially compact trails.  The final pairing tuple keeps the
        # result deterministic when scores tie.
        score = (
            float(counts[0]),
            float(counts[1]) if len(counts) > 1 else float(counts[0]),
            -float(counts[-1]),
            -sum(areas),
        )
        candidate = (score, pairs)
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[1]


def _all_odd_pairings(nodes: tuple[int, ...]):
    if not nodes:
        yield ()
        return
    left = nodes[0]
    for index in range(1, len(nodes)):
        right = nodes[index]
        for rest in _all_odd_pairings(nodes[1:index] + nodes[index + 1 :]):
            yield (_edge_key(left, right), *rest)


def _pair_nearest_odds(points: tuple[Point, ...], odd_nodes: list[int]):
    remaining = list(odd_nodes)
    while remaining:
        left = remaining.pop(0)
        right = min(remaining, key=lambda candidate: math.dist(points[left], points[candidate]))
        remaining.remove(right)
        yield left, right


def _euler_circuit(edges: list[Edge], start: int) -> list[tuple[int, int, int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for edge_id, (left, right) in enumerate(edges):
        adjacency[left].append(edge_id)
        adjacency[right].append(edge_id)
    used: set[int] = set()
    stack: list[tuple[int, int | None, int | None]] = [(start, None, None)]
    result: list[tuple[int, int, int]] = []
    while stack:
        node, _incoming, _previous = stack[-1]
        edge_id = next((candidate for candidate in adjacency[node] if candidate not in used), None)
        if edge_id is not None:
            used.add(edge_id)
            left, right = edges[edge_id]
            stack.append((right if node == left else left, edge_id, node))
            continue
        node, incoming, previous = stack.pop()
        if incoming is not None and previous is not None:
            result.append((previous, node, incoming))
    return list(reversed(result))


def _shortest_wall_safe_route(points: tuple[Point, ...], edges: tuple[Edge, ...], start: int, end: int) -> tuple[int, ...]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for left, right in edges:
        length = math.dist(points[left], points[right])
        adjacency[left].append((right, length))
        adjacency[right].append((left, length))
    queue: list[tuple[float, int]] = [(0.0, start)]
    distance = {start: 0.0}
    previous: dict[int, int] = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if node == end:
            route = [end]
            while route[-1] != start:
                route.append(previous[route[-1]])
            return tuple(reversed(route))
        if cost > distance[node] + 1e-12:
            continue
        for next_node, length in adjacency[node]:
            candidate = cost + length
            if candidate + 1e-12 >= distance.get(next_node, math.inf):
                continue
            distance[next_node] = candidate
            previous[next_node] = node
            heapq.heappush(queue, (candidate, next_node))
    raise ValueError("prototype honeycomb partitions are disconnected")


def _order_deposit_trails(
    points: tuple[Point, ...],
    edges: tuple[Edge, ...],
    trails: list[tuple[int, ...]],
    start: int,
) -> tuple[list[tuple[int, ...]], tuple[int, ...], list[tuple[int, ...]]]:
    """Order and orient deposited sub-trails with shortest wall-safe links."""

    remaining = list(trails)
    ordered: list[tuple[int, ...]] = []
    travels: list[tuple[int, ...]] = []
    current = start
    initial: tuple[int, ...] | None = None
    while remaining:
        best: tuple[float, int, tuple[int, ...], tuple[int, ...]] | None = None
        for index, original in enumerate(remaining):
            for candidate in (original, tuple(reversed(original))):
                route = _shortest_wall_safe_route(points, edges, current, candidate[0])
                proposal = (len(route), index, candidate, route)
                if best is None or proposal[:2] < best[:2]:
                    best = proposal
        assert best is not None
        _length, index, selected, route = best
        if initial is None:
            initial = route
        else:
            travels.append(route)
        ordered.append(selected)
        current = selected[-1]
        remaining.pop(index)
    assert initial is not None
    return ordered, initial, travels


def _partition_palette(count: int) -> list[str]:
    # Golden-angle ordering keeps neighbouring partition indexes visibly
    # distinct even when a full-scale honeycomb needs hundreds of colours.
    return [f'hsl({round((index * 137.508) % 360)}, 72%, 42%)' for index in range(count)]


def _edge_key(left: int, right: int) -> Edge:
    return (left, right) if left < right else (right, left)
