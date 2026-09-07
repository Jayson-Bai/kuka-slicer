from collections import Counter

import pytest

from kuka_slicer.honeycomb_pathing.partition_prototype import (
    _edge_key,
    build_macro_partition_prototype,
    build_honeycomb_partition_prototype,
    render_macro_partition_svg,
    render_honeycomb_partition_svg,
)


def test_minimal_honeycomb_partitions_cover_every_wall_once_and_render(tmp_path):
    plan = build_honeycomb_partition_prototype(columns=3, rows=2, side_mm=10.0)

    used = Counter(
        _edge_key(left, right)
        for partition in plan.partitions
        for left, right in zip(partition, partition[1:])
    )
    assert used == Counter({edge: 1 for edge in plan.edges})
    assert len(plan.partitions) == 5
    assert all(len(partition) >= 2 for partition in plan.partitions)
    assert min(len(partition) - 1 for partition in plan.partitions) >= 4
    # Every dashed inter-partition move is constrained to existing honeycomb
    # walls in this conservative prototype, so it cannot pass through a void.
    assert all(
        all(_edge_key(left, right) in plan.edges for left, right in zip(route, route[1:]))
        for route in plan.transfers
    )

    output = render_honeycomb_partition_svg(plan, tmp_path / "honeycomb_partition_prototype.svg")

    assert output.is_file()
    assert "分区 1" in output.read_text(encoding="utf-8")


def test_current_ui_scale_honeycomb_case_keeps_every_wall_once():
    # 17 × 13 cells at a 5 mm side are the regular-grid approximation of the
    # current 150 × 100 mm honeycomb model shown in the slicer UI.
    plan = build_honeycomb_partition_prototype(columns=17, rows=13, side_mm=5.0)

    used = Counter(
        _edge_key(left, right)
        for partition in plan.partitions
        for left, right in zip(partition, partition[1:])
    )
    x_values = [point[0] for point in plan.points]
    y_values = [point[1] for point in plan.points]
    assert max(x_values) - min(x_values) == pytest.approx(151.55444566)
    assert max(y_values) - min(y_values) == pytest.approx(100.0)
    assert len(plan.edges) == 722
    assert len(plan.partitions) == 220
    assert used == Counter({edge: 1 for edge in plan.edges})


def test_macro_partition_prototype_groups_the_current_ui_scale_into_work_zones(tmp_path):
    graph = build_honeycomb_partition_prototype(columns=17, rows=13, side_mm=5.0)
    plan = build_macro_partition_prototype(graph.points, graph.edges, columns=4, rows=3)

    used = Counter(
        _edge_key(left, right)
        for partition in plan.partitions
        for trail in partition.deposited_trails
        for left, right in zip(trail, trail[1:])
    )
    assert len(plan.partitions) == 12
    assert used == Counter({edge: 1 for edge in plan.edges})
    assert all(partition.intra_partition_travels for partition in plan.partitions)
    all_safe_travels = [
        *plan.between_partition_travels,
        *(route for partition in plan.partitions for route in partition.intra_partition_travels),
    ]
    assert all(
        all(_edge_key(left, right) in plan.edges for left, right in zip(route, route[1:]))
        for route in all_safe_travels
    )

    output = render_macro_partition_svg(plan, tmp_path / "macro_partition_prototype.svg")
    assert output.is_file()
    assert "宏观分区" in output.read_text(encoding="utf-8")
