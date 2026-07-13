import math

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from kuka_slicer.slicer import (
    DEFAULT_FIBER_LAYER_HEIGHT_MM,
    DEFAULT_FIBER_LINE_WIDTH_MM,
    DEFAULT_RESIN_INFILL_OVERLAP_PERCENT,
    DEFAULT_RESIN_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_LINE_WIDTH_MM,
    RaftLayerConfig,
    SliceConfig,
    _build_resin_paths,
    _filter_concentric_paths_by_spacing,
    _raft_lattice_infill_paths,
    _raft_zigzag_infill_paths,
    _smooth_path_corners,
    _uniform_concentric_offsets,
    add_raft_to_job,
    normalize_job_xy_origin,
    optimize_open_path_travel,
    slice_mesh_to_job,
)
from kuka_slicer.stl_io import Mesh
from kuka_slicer.ui_server import _simplify_preview_path, expand_fiber_template_for_resin_layers


def test_cube_slice_produces_closed_square_path():
    mesh = Mesh(_cube_triangles(size=10.0))

    job = slice_mesh_to_job(mesh, SliceConfig(layer_height=5.0, infill_pattern="contour"))

    assert len(job.material_paths) == 2
    group = job.material_paths[0]
    assert group.layer_index == 0
    assert group.material == "R"
    assert len(group.paths) >= 2
    path = group.paths[0]
    assert path.shape[1] == 3
    assert np.allclose(path[:, 2], 5.0)
    assert path.shape[0] >= 4


def test_layer_generation_includes_top_cap_layer():
    mesh = Mesh(_cube_triangles(size=5.0))

    job = slice_mesh_to_job(mesh, SliceConfig(layer_height=0.5, infill_pattern="contour"))

    assert len(job.material_paths) == 10
    assert np.allclose([group.paths[0][0, 2] for group in job.material_paths], np.arange(0.5, 5.5, 0.5))


def test_fiber_template_z_is_offset_from_resin_layer_z():
    mesh = Mesh(_cube_triangles(size=1.5))
    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=0.5, line_width=0.1, infill_pattern="contour"),
    )
    template_paths = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]

    fiber_paths_by_layer = expand_fiber_template_for_resin_layers(job, template_paths)

    resin_z_values = [group.paths[0][0, 2] for group in job.material_paths]
    assert np.allclose(resin_z_values, [0.5, 1.0, 1.5])
    assert sorted(fiber_paths_by_layer) == [0, 1]
    assert np.allclose(
        [fiber_paths_by_layer[layer_index][0][0][2] for layer_index in sorted(fiber_paths_by_layer)],
        [0.6, 1.1],
    )


def test_fiber_template_paths_are_smoothed_before_export():
    mesh = Mesh(_cube_triangles(size=1.5))
    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=0.5, line_width=0.1, infill_pattern="contour"),
    )
    template_paths = [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]

    fiber_paths_by_layer = expand_fiber_template_for_resin_layers(job, template_paths)
    exported_path = np.asarray(fiber_paths_by_layer[0][0], dtype=np.float32)

    assert exported_path.shape[0] > len(template_paths[0])
    assert np.allclose(exported_path[:, 2], 0.6)


def test_resin_line_infill_uses_default_overlap_spacing():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="lines_x"),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    contour_paths = [
        path for path, role in zip(job.material_paths[1].paths, roles) if role != "infill"
    ]
    assert len(contour_paths) == 2
    assert len(infill_paths) == 1
    assert infill_paths[0].shape[0] > 10
    assert np.allclose(
        _horizontal_scan_ys(infill_paths[0]),
        [5.4, 7.2, 9.0, 10.8, 12.6, 14.4, 16.2],
    )


def test_resin_line_infill_can_disable_overlap_for_legacy_spacing():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="lines_x", infill_overlap=0.0),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    assert np.allclose(_horizontal_scan_ys(infill_paths[0]), [6.0, 8.0, 10.0, 12.0, 14.0])


def test_resin_perimeters_use_overlap_spacing():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="contour"),
    )

    outer = _paths_with_role(paths, roles, "outer_contour")[0]
    inner = _paths_with_role(paths, roles, "inner_contour")[0]
    assert np.isclose(float(outer[:, 0].min()), 1.0, atol=0.05)
    assert np.isclose(float(inner[:, 0].min()), 2.8, atol=0.05)


def test_resin_infill_density_changes_path_spacing():
    mesh = Mesh(_cube_triangles(size=20.0))

    dense = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="lines_x", infill_density=100),
    )
    sparse = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="lines_x", infill_density=50),
    )

    dense_roles = dense.meta["path_roles"]["R"]["1"]
    sparse_roles = sparse.meta["path_roles"]["R"]["1"]
    dense_infill = _paths_with_role(dense.material_paths[1].paths, dense_roles, "infill")
    sparse_infill = _paths_with_role(sparse.material_paths[1].paths, sparse_roles, "infill")
    assert len(dense_infill) == 1
    assert len(sparse_infill) == 1
    assert len(_horizontal_scan_ys(dense_infill[0])) == 7
    assert len(_horizontal_scan_ys(sparse_infill[0])) == 3
    assert dense.meta["slicing"]["infill_density"] == 100
    assert dense.meta["slicing"]["infill_overlap"] == DEFAULT_RESIN_INFILL_OVERLAP_PERCENT


def test_part_bottom_and_top_layers_force_full_zigzag_infill():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="gyroid",
            infill_density=0,
        ),
    )

    bottom_roles = job.meta["path_roles"]["R"]["0"]
    middle_roles = job.meta["path_roles"]["R"]["1"]
    top_index = len(job.material_paths) - 1
    top_roles = job.meta["path_roles"]["R"][str(top_index)]
    bottom_infill = _paths_with_role(job.material_paths[0].paths, bottom_roles, "infill")
    middle_infill = _paths_with_role(job.material_paths[1].paths, middle_roles, "infill")
    top_infill = _paths_with_role(job.material_paths[top_index].paths, top_roles, "infill")

    assert bottom_infill
    assert not middle_infill
    assert top_infill
    assert len(bottom_infill) <= 20
    assert len(top_infill) <= 20
    assert _dominant_infill_angle(bottom_infill) == 45
    assert _dominant_infill_angle(top_infill) == -45
    assert all(not _path_has_non_adjacent_crossing(path) for path in bottom_infill)
    assert all(not _path_has_non_adjacent_crossing(path) for path in top_infill)
    assert job.meta["slicing"]["forced_part_cap_layers"] == {
        "infill_pattern": "zigzag_diagonal",
        "infill_density": 100.0,
        "bottom_angle_degrees": 45.0,
        "top_angle_degrees": -45.0,
    }


def test_contour_offset_infill_generates_closed_inner_rings():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="contour_offset"),
    )

    infill = _paths_with_role(paths, roles, "infill")
    assert len(infill) >= 2
    assert all(path.shape[0] > 2 for path in infill)
    assert all(np.allclose(path[0], path[-1]) for path in infill)


def test_contour_offset_keeps_printable_residual_ring_centered():
    contour = np.asarray(
        [[0, 0], [21, 0], [21, 21], [0, 21], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="contour_offset"),
    )
    infill = _paths_with_role(paths, roles, "infill")
    last = infill[-1]
    center_x = (float(last[:, 0].min()) + float(last[:, 0].max())) * 0.5
    center_y = (float(last[:, 1].min()) + float(last[:, 1].max())) * 0.5

    assert np.isclose(center_x, 10.5, atol=0.05)
    assert np.isclose(center_y, 10.5, atol=0.05)


def test_contour_offset_keeps_closed_residual_ring_when_narrow():
    contour = np.asarray(
        [[0, 0], [21, 0], [21, 21], [0, 21], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="contour_offset"),
    )
    infill = _paths_with_role(paths, roles, "infill")

    assert infill
    assert all(np.allclose(path[0], path[-1]) for path in infill)


def test_contour_offset_keeps_fixed_spacing_before_residual_ring():
    offsets = _uniform_concentric_offsets(9.5, line_width=2.0, path_spacing=1.8)

    assert np.allclose(np.diff(offsets[:-1]), [1.8, 1.8, 1.8, 1.8])
    assert offsets[-1] == 9.5


def test_contour_offset_filters_paths_closer_than_line_width():
    outer = np.asarray([[0, 0], [20, 0]], dtype=np.float32)
    too_close = np.asarray([[0, 1], [20, 1]], dtype=np.float32)
    far_enough = np.asarray([[0, 2], [20, 2]], dtype=np.float32)

    filtered = _filter_concentric_paths_by_spacing([outer, too_close, far_enough], 2.0, 1e-5)

    assert len(filtered) == 2
    assert filtered[0] is outer
    assert filtered[1] is far_enough


def test_contour_offset_centers_residual_ring_per_local_region():
    contour = np.asarray(
        [[0, 0], [41, 0], [41, 13], [0, 13], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="contour_offset"),
    )
    infill = _paths_with_role(paths, roles, "infill")
    last = infill[-1]
    center_y = (float(last[:, 1].min()) + float(last[:, 1].max())) * 0.5

    assert np.isclose(center_y, 6.5, atol=0.05)


def test_resin_infill_respects_inner_holes():
    outer = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )
    hole = np.asarray(
        [[8, 8], [12, 8], [12, 12], [8, 12], [8, 8]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [outer, hole],
        SliceConfig(layer_height=1.0, line_width=0.5, infill_pattern="lines_x"),
    )

    assert roles.count("outer_contour") == 2
    assert roles.count("inner_contour") == 2
    hole_polygon = Polygon(hole[:-1])
    for path, role in zip(paths, roles):
        if role == "infill":
            assert not LineString(path).crosses(hole_polygon)
            assert not LineString(path).within(hole_polygon)


def test_resin_infill_smoothing_rounds_sharp_corners():
    path = np.asarray([[0, 0], [10, 0], [10, 10]], dtype=np.float32)

    smoothed = _smooth_path_corners(path, max_radius=2.0, angle_threshold_degrees=150.0, tolerance=1e-5)

    assert smoothed.shape[0] > path.shape[0]
    assert np.allclose(smoothed[0], path[0])
    assert np.allclose(smoothed[-1], path[-1])
    assert not any(np.allclose(point, [10, 0]) for point in smoothed[1:-1])


def test_zigzag_infill_does_not_self_cross_in_concave_region():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 8], [12, 8], [12, 14], [20, 14], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=0.5, infill_pattern="lines_x"),
    )

    for path, role in zip(paths, roles):
        if role == "infill":
            assert not _path_has_non_adjacent_crossing(path)


def test_zigzag_infill_connects_adjacent_annulus_segments_when_safe():
    outer = _circle_path((0.0, 0.0), 20.0, 128)
    hole = _circle_path((0.0, 0.0), 10.0, 96)

    paths, roles = _build_resin_paths(
        [outer, hole],
        SliceConfig(layer_height=1.0, line_width=1.0, infill_pattern="lines_x"),
    )
    infill = _paths_with_role(paths, roles, "infill")

    assert any(path.shape[0] > 6 for path in infill)
    for path in infill:
        assert not _path_has_non_adjacent_crossing(path)


def test_perimeter_roles_mark_outer_and_inner_wall_pairs():
    outer = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )
    hole = np.asarray(
        [[8, 8], [12, 8], [12, 12], [8, 12], [8, 8]],
        dtype=np.float32,
    )

    _, roles = _build_resin_paths(
        [outer, hole],
        SliceConfig(layer_height=1.0, line_width=0.5, infill_pattern="contour"),
    )

    assert roles == ["outer_contour", "outer_contour", "inner_contour", "inner_contour"]


def test_alternating_diagonal_infill_flips_angle_by_layer():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="alternating_diagonal",
        ),
    )

    first_roles = job.meta["path_roles"]["R"]["1"]
    second_roles = job.meta["path_roles"]["R"]["2"]
    first_layer_path = _paths_with_role(job.material_paths[1].paths, first_roles, "infill")[0]
    second_layer_path = _paths_with_role(job.material_paths[2].paths, second_roles, "infill")[0]
    first_delta = first_layer_path[1, :2] - first_layer_path[0, :2]
    second_delta = second_layer_path[1, :2] - second_layer_path[0, :2]
    assert first_delta[0] * first_delta[1] < 0
    assert second_delta[0] * second_delta[1] > 0


def test_triangular_infill_generates_single_layer_lattice_without_edge_overlaps():
    mesh = Mesh(_cube_triangles(size=30.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=50,
        ),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    directions, overlap_length, longest_segment = _infill_direction_overlap_stats(infill_paths)
    assert infill_paths
    assert len(infill_paths) <= 12
    assert directions == {0, 60, 120}
    assert overlap_length < 1e-5
    assert longest_segment > 6.0


def test_triangular_infill_supports_zero_density_without_infill_paths():
    mesh = Mesh(_cube_triangles(size=30.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=0,
        ),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    assert not _paths_with_role(job.material_paths[1].paths, roles, "infill")


def test_triangular_infill_keeps_lattice_shape_at_low_density():
    mesh = Mesh(_cube_triangles(size=30.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=10,
        ),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    directions, overlap_length, longest_segment = _infill_direction_overlap_stats(infill_paths)
    assert infill_paths
    assert directions == {0, 60, 120}
    assert overlap_length < 1e-5
    assert longest_segment > 10.0


def test_triangular_infill_has_no_edge_overlaps_at_full_density():
    mesh = Mesh(_cube_triangles(size=30.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=100,
        ),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    directions, overlap_length, _ = _infill_direction_overlap_stats(infill_paths)
    assert infill_paths
    assert directions == {0, 60, 120}
    assert overlap_length < 1e-5


def test_triangular_infill_filters_sub_line_width_boundary_features():
    mesh = Mesh(_cube_triangles(size=30.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=50,
        ),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    lengths = [
        float(np.linalg.norm(end[:2] - start[:2]))
        for path in infill_paths
        for start, end in zip(path, path[1:])
        if np.linalg.norm(end[:2] - start[:2]) > 1e-5
    ]
    assert lengths
    assert min(lengths) >= 2.0


def test_gyroid_infill_generates_continuous_curves_at_high_density():
    mesh = Mesh(_cube_triangles(size=30.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="gyroid",
            infill_density=85,
        ),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    overlap_length = _infill_path_overlap_length(infill_paths)
    assert infill_paths
    assert len(infill_paths) <= 35
    assert max(len(path) for path in infill_paths) >= 20
    assert overlap_length < 1e-5


def test_gyroid_infill_supports_zero_and_full_density():
    mesh = Mesh(_cube_triangles(size=30.0))

    empty_job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="gyroid",
            infill_density=0,
        ),
    )
    full_job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="gyroid",
            infill_density=100,
        ),
    )

    empty_roles = empty_job.meta["path_roles"]["R"]["1"]
    full_roles = full_job.meta["path_roles"]["R"]["1"]
    assert not _paths_with_role(empty_job.material_paths[1].paths, empty_roles, "infill")
    assert _paths_with_role(full_job.material_paths[1].paths, full_roles, "infill")


def _infill_direction_overlap_stats(
    infill_paths: list[np.ndarray],
) -> tuple[set[int], float, float]:
    directions: set[int] = set()
    segments: list[LineString] = []
    longest_segment = 0.0
    for path in infill_paths:
        deltas = np.diff(path[:, :2], axis=0)
        for index, delta in enumerate(deltas):
            length = float(np.linalg.norm(delta))
            if length <= 1.0:
                continue
            longest_segment = max(longest_segment, length)
            segments.append(
                LineString(
                    [
                        tuple(float(value) for value in path[index, :2]),
                        tuple(float(value) for value in path[index + 1, :2]),
                    ]
                )
            )
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0
            for expected in (0.0, 60.0, 120.0):
                if abs(angle - expected) < 2.0:
                    directions.add(int(expected))

    overlap_length = 0.0
    for first_index, first_segment in enumerate(segments):
        for second_segment in segments[first_index + 1 :]:
            intersection = first_segment.intersection(second_segment)
            if not intersection.is_empty:
                overlap_length += float(intersection.length)
    return directions, overlap_length, longest_segment


def _infill_path_overlap_length(infill_paths: list[np.ndarray]) -> float:
    segments: list[LineString] = []
    for path in infill_paths:
        for start, end in zip(path, path[1:]):
            if np.linalg.norm(end[:2] - start[:2]) <= 1e-5:
                continue
            segments.append(
                LineString(
                    [
                        tuple(float(value) for value in start[:2]),
                        tuple(float(value) for value in end[:2]),
                    ]
                )
            )

    overlap_length = 0.0
    for first_index, first_segment in enumerate(segments):
        for second_segment in segments[first_index + 1 :]:
            intersection = first_segment.intersection(second_segment)
            if not intersection.is_empty:
                overlap_length += float(intersection.length)
    return overlap_length


def _infill_total_length(infill_paths: list[np.ndarray]) -> float:
    total = 0.0
    for path in infill_paths:
        total += float(np.sum(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)))
    return total


def test_material_defaults_are_hard_coded():
    resin = SliceConfig(material="R")
    fiber = SliceConfig(material="F")

    assert resin.layer_height == DEFAULT_RESIN_LAYER_HEIGHT_MM
    assert resin.line_width == DEFAULT_RESIN_LINE_WIDTH_MM
    assert resin.infill_overlap == DEFAULT_RESIN_INFILL_OVERLAP_PERCENT
    assert fiber.layer_height == DEFAULT_FIBER_LAYER_HEIGHT_MM
    assert fiber.line_width == DEFAULT_FIBER_LINE_WIDTH_MM


def test_preview_simplification_keeps_contour_corners():
    points = [[0.0, 0.0, 0.5], [0.0, 20.0, 0.5]]
    points.extend([[float(x), 20.0, 0.5] for x in range(1, 202)])
    points.append([0.0, 0.0, 0.5])

    simplified = _simplify_preview_path(points, max_points=2000)

    assert [0.0, 20.0, 0.5] in simplified


def test_open_path_travel_optimizer_reverses_next_path_when_closer():
    paths = [
        np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0, 0.0], [10.0, 1.0, 0.0]], dtype=np.float32),
    ]

    optimized = optimize_open_path_travel(paths)

    assert np.allclose(optimized[0], paths[0])
    assert np.allclose(optimized[1], paths[1][::-1])


def test_raft_layers_shift_part_layers_and_z():
    mesh = Mesh(_cube_triangles(size=10.0))
    config = SliceConfig(layer_height=5.0, line_width=1.0, infill_pattern="lines_x")
    job = slice_mesh_to_job(mesh, config)

    z_shift = add_raft_to_job(
        job,
        mesh,
        config,
        [
            RaftLayerConfig(outward_offset=2.0, layer_height=0.3, infill_density=80),
            RaftLayerConfig(outward_offset=1.0, layer_height=0.2, infill_density=60),
        ],
        top_gap=0.4,
    )

    resin_groups = [group for group in job.material_paths if group.material == "R"]
    assert np.isclose(z_shift, 0.9)
    assert [group.layer_index for group in resin_groups[:4]] == [0, 1, 2, 3]
    assert np.isclose(resin_groups[0].paths[0][0, 2], 0.3)
    assert np.isclose(resin_groups[1].paths[0][0, 2], 0.5)
    assert np.isclose(resin_groups[2].paths[0][0, 2], 5.9)
    assert job.meta["raft"]["layer_count"] == 2


def test_single_raft_layer_touching_part_uses_lattice_independent_of_part_pattern():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="contour_offset")
    job = slice_mesh_to_job(mesh, config)

    add_raft_to_job(
        job,
        mesh,
        config,
        [RaftLayerConfig(outward_offset=2.0, layer_height=0.5, infill_density=100)],
        top_gap=0.2,
    )

    raft_roles = job.meta["path_roles"]["R"]["0"]
    raft_infill = _paths_with_role(job.material_paths[0].paths, raft_roles, "infill")
    assert raft_infill
    assert len(raft_infill) <= 3
    assert max(path.shape[0] for path in raft_infill) > 100
    assert all(not _path_has_non_adjacent_crossing(path) for path in raft_infill)


def test_only_top_raft_layer_touching_part_uses_lattice():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="contour_offset")
    job = slice_mesh_to_job(mesh, config)

    add_raft_to_job(
        job,
        mesh,
        config,
        [
            RaftLayerConfig(outward_offset=2.0, layer_height=0.5, infill_density=100),
            RaftLayerConfig(outward_offset=1.0, layer_height=0.5, infill_density=50),
        ],
        top_gap=0.2,
    )

    bottom_roles = job.meta["path_roles"]["R"]["0"]
    top_roles = job.meta["path_roles"]["R"]["1"]
    bottom_infill = _paths_with_role(job.material_paths[0].paths, bottom_roles, "infill")
    top_infill = _paths_with_role(job.material_paths[1].paths, top_roles, "infill")

    assert bottom_infill
    assert top_infill
    assert len(bottom_infill) == 1
    assert len(top_infill) <= 3
    assert max(path.shape[0] for path in top_infill) >= 50
    assert all(not _path_has_non_adjacent_crossing(path) for path in top_infill)


def test_raft_lattice_density_changes_spacing():
    geometry = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    dense = _raft_lattice_infill_paths(geometry, config, infill_density=100.0)
    sparse = _raft_lattice_infill_paths(geometry, config, infill_density=50.0)

    assert dense
    assert sparse
    assert _infill_total_length(dense) > _infill_total_length(sparse)
    assert dense[0].shape[0] > sparse[0].shape[0]


def test_raft_zigzag_infill_avoids_long_sharp_connectors_near_holes():
    outer = [(0, 0), (42, 0), (42, 20), (0, 20)]
    hole = list(Point(30, 10).buffer(3.0, resolution=32).exterior.coords)
    geometry = Polygon(outer, holes=[hole])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    paths = _raft_zigzag_infill_paths(geometry, config, infill_density=100.0)

    assert paths
    for path in paths:
        for start, end in zip(path[:-1], path[1:]):
            delta = end[:2] - start[:2]
            if abs(float(delta[0])) > 0.1 and abs(float(delta[1])) > 0.1:
                assert np.linalg.norm(delta) <= 3.0


def test_normalize_job_xy_origin_moves_lower_left_to_zero():
    mesh = Mesh(_cube_triangles(size=10.0) - np.asarray([5.0, 2.0, 0.0], dtype=np.float32))
    job = slice_mesh_to_job(mesh, SliceConfig(layer_height=5.0, infill_pattern="contour"))

    translation = normalize_job_xy_origin(job)
    all_points = np.vstack([path for group in job.material_paths for path in group.paths])

    assert np.allclose(all_points[:, :2].min(axis=0), [0.0, 0.0])
    assert np.isclose(all_points[:, 2].min(), 5.0)
    assert np.allclose(
        translation,
        [
            -job.meta["xy_origin_normalization"]["source_min_x"],
            -job.meta["xy_origin_normalization"]["source_min_y"],
        ],
    )
    assert job.meta["xy_origin_normalization"]["applied"] is True


def test_sinusoidal_curve_changes_path_z():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            curve_mode="sinusoidal",
            curve_amplitude=1.0,
            curve_period=20.0,
        ),
    )

    assert max(np.ptp(path[:, 2]) for path in job.material_paths[0].paths) > 0.5


def _cube_triangles(size: float) -> np.ndarray:
    v = np.asarray(
        [
            [0, 0, 0],
            [size, 0, 0],
            [size, size, 0],
            [0, size, 0],
            [0, 0, size],
            [size, 0, size],
            [size, size, size],
            [0, size, size],
        ],
        dtype=np.float32,
    )
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 6, 5), (4, 7, 6),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    return np.asarray([[v[a], v[b], v[c]] for a, b, c in faces], dtype=np.float32)


def _paths_with_role(paths: list[np.ndarray], roles: list[str], role: str) -> list[np.ndarray]:
    return [path for path, path_role in zip(paths, roles) if path_role == role]


def _horizontal_scan_ys(path: np.ndarray) -> list[float]:
    ys = []
    for index in range(path.shape[0] - 1):
        if np.isclose(path[index, 1], path[index + 1, 1]):
            ys.append(round(float(path[index, 1]), 6))
    return sorted(set(ys))


def _dominant_infill_angle(paths: list[np.ndarray]) -> int:
    counts: dict[int, int] = {}
    for path in paths:
        for delta in np.diff(path[:, :2], axis=0):
            length = float(np.linalg.norm(delta))
            if length <= 0.4:
                continue
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
            normalized = ((angle + 90.0) % 180.0) - 90.0
            if abs(normalized - 45.0) < 2.0:
                counts[45] = counts.get(45, 0) + 1
            elif abs(normalized + 45.0) < 2.0:
                counts[-45] = counts.get(-45, 0) + 1
    return max(counts, key=counts.get)


def _path_has_non_adjacent_crossing(path: np.ndarray) -> bool:
    segments = [
        LineString([tuple(path[index, :2]), tuple(path[index + 1, :2])])
        for index in range(path.shape[0] - 1)
    ]
    for index, segment in enumerate(segments):
        for other_index, other in enumerate(segments[index + 1 :], start=index + 1):
            if abs(index - other_index) <= 1:
                continue
            intersection = segment.intersection(other)
            if intersection.is_empty:
                continue
            endpoint_union = (
                Point(tuple(path[index, :2])).buffer(1e-5)
                .union(Point(tuple(path[index + 1, :2])).buffer(1e-5))
                .union(Point(tuple(path[other_index, :2])).buffer(1e-5))
                .union(Point(tuple(path[other_index + 1, :2])).buffer(1e-5))
            )
            if not endpoint_union.covers(intersection):
                return True
    return False


def _circle_path(center: tuple[float, float], radius: float, point_count: int) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    points = np.asarray(
        [
            [center[0] + np.cos(angle) * radius, center[1] + np.sin(angle) * radius]
            for angle in angles
        ],
        dtype=np.float32,
    )
    return np.vstack([points, points[0]])
