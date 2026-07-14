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
    _connect_resin_infill_paths,
    _filter_concentric_paths_by_spacing,
    _libslic3r_fill_surface_overlap_offset,
    _raft_lattice_infill_paths,
    _raft_zigzag_infill_paths,
    _resin_infill_surface_geometry,
    _smooth_path_corners,
    _uniform_concentric_offsets,
    add_raft_to_job,
    merge_adjacent_connected_paths,
    normalize_job_xy_origin,
    optimize_triangle_infill_travel,
    recommended_geometry_tolerance,
    recommended_pyslm_strategy_defaults,
    slice_mesh_to_job,
)
from kuka_slicer.stl_io import Mesh
from kuka_slicer.ui_server import (
    _index_html,
    _raft_layers_from_params,
    _simplify_preview_path,
    expand_fiber_template_for_resin_layers,
)


def test_cube_slice_produces_closed_square_path():
    mesh = Mesh(_cube_triangles(size=10.0))

    job = slice_mesh_to_job(mesh, SliceConfig(layer_height=5.0, infill_pattern="none"))

    assert len(job.material_paths) == 2
    group = job.material_paths[0]
    assert group.layer_index == 0
    assert group.material == "R"
    assert len(group.paths) >= 2
    path = group.paths[0]
    assert path.shape[1] == 3
    assert np.allclose(path[:, 2], 5.0)
    assert path.shape[0] >= 4


def test_layer_generation_includes_top_z_layer():
    mesh = Mesh(_cube_triangles(size=5.0))

    job = slice_mesh_to_job(mesh, SliceConfig(layer_height=0.5, infill_pattern="none"))

    assert len(job.material_paths) == 10
    assert np.allclose([group.paths[0][0, 2] for group in job.material_paths], np.arange(0.5, 5.5, 0.5))


def test_fiber_template_z_is_offset_from_resin_layer_z():
    mesh = Mesh(_cube_triangles(size=1.5))
    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=0.5, line_width=0.1, infill_pattern="none"),
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
        SliceConfig(layer_height=0.5, line_width=0.1, infill_pattern="none"),
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
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="aligned_rectilinear"),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    contour_paths = [
        path for path, role in zip(job.material_paths[1].paths, roles) if role != "infill"
    ]
    assert len(contour_paths) == 2
    assert len(infill_paths) == 1
    assert infill_paths[0].shape[0] > 8
    assert np.allclose(
        _horizontal_scan_ys(infill_paths[0]),
        [3.6, 5.4, 7.2, 9.0, 10.8, 12.6, 14.4, 16.2],
    )


def test_resin_line_infill_can_disable_overlap_for_legacy_spacing():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="aligned_rectilinear", infill_overlap=0.0),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    assert len(infill_paths) == 1
    assert np.allclose(
        _horizontal_scan_ys(infill_paths[0]),
        [6.0, 8.0, 10.0, 12.0, 14.0],
    )


def test_resin_perimeters_use_overlap_spacing():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="none"),
    )

    outer = _paths_with_role(paths, roles, "outer_contour")[0]
    inner = _paths_with_role(paths, roles, "inner_contour")[0]
    assert np.isclose(float(outer[:, 0].min()), 1.0, atol=0.05)
    assert np.isclose(float(inner[:, 0].min()), 2.8, atol=0.05)


def test_libslic3r_fill_surface_overlap_offset_matches_spacing_formula():
    offset = _libslic3r_fill_surface_overlap_offset(
        line_spacing=1.8,
        line_width=2.0,
        overlap_percent=10.0,
    )

    assert np.isclose(offset, -0.7)


def test_resin_infill_surface_uses_last_perimeter_and_overlap_offset():
    geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    infill_surface = _resin_infill_surface_geometry(geometry, config)

    min_x, min_y, max_x, max_y = infill_surface.bounds
    assert np.isclose(min_x, 3.5, atol=0.05)
    assert np.isclose(min_y, 3.5, atol=0.05)
    assert np.isclose(max_x, 16.5, atol=0.05)
    assert np.isclose(max_y, 16.5, atol=0.05)


def test_resin_infill_surface_respects_configured_perimeter_count():
    geometry = Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])
    one_wall = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_overlap=10.0,
        perimeter_count=1,
    )
    three_walls = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_overlap=10.0,
        perimeter_count=3,
    )

    one_wall_min_x = _resin_infill_surface_geometry(geometry, one_wall).bounds[0]
    three_wall_min_x = _resin_infill_surface_geometry(geometry, three_walls).bounds[0]

    assert three_wall_min_x > one_wall_min_x


def test_resin_infill_density_changes_path_spacing():
    mesh = Mesh(_cube_triangles(size=20.0))

    dense = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="aligned_rectilinear", infill_density=100),
    )
    sparse = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="aligned_rectilinear", infill_density=50),
    )

    dense_roles = dense.meta["path_roles"]["R"]["1"]
    sparse_roles = sparse.meta["path_roles"]["R"]["1"]
    dense_infill = _paths_with_role(dense.material_paths[1].paths, dense_roles, "infill")
    sparse_infill = _paths_with_role(sparse.material_paths[1].paths, sparse_roles, "infill")
    assert len(dense_infill) == 1
    assert len(sparse_infill) == 1
    assert len(_horizontal_scan_ys(dense_infill[0])) == 8
    assert len(_horizontal_scan_ys(sparse_infill[0])) == 4
    assert dense.meta["slicing"]["infill_density"] == 100
    assert dense.meta["slicing"]["infill_overlap"] == DEFAULT_RESIN_INFILL_OVERLAP_PERCENT


def test_part_bottom_and_top_layers_force_zigzag_full_density():
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
    assert _dominant_infill_angle(bottom_infill) == 45
    assert _dominant_infill_angle(top_infill) == -45
    assert job.meta["slicing"]["infill_density"] == 0
    assert job.meta["slicing"]["part_cap_layers"] == {
        "bottom": 0,
        "top": top_index,
        "infill_pattern": "zigzag",
        "infill_density": 100.0,
    }


def test_part_caps_do_not_reclassify_raft_layers():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        infill_pattern="gyroid",
        infill_density=0,
    )
    job = slice_mesh_to_job(mesh, config)

    add_raft_to_job(
        job,
        mesh,
        config,
        [
            RaftLayerConfig(outward_offset=2.0, layer_height=0.5, infill_density=10),
            RaftLayerConfig(outward_offset=1.0, layer_height=0.5, infill_density=50),
        ],
        top_gap=0.2,
    )

    raft_roles = job.meta["path_roles"]["R"]["0"]
    part_bottom_roles = job.meta["path_roles"]["R"]["2"]
    raft_infill = _paths_with_role(job.material_paths[0].paths, raft_roles, "infill")
    part_bottom_infill = _paths_with_role(
        job.material_paths[2].paths,
        part_bottom_roles,
        "infill",
    )

    assert raft_infill
    assert part_bottom_infill
    assert _has_infill_direction(raft_infill, 0.0)
    assert _has_infill_direction(part_bottom_infill, 45.0)
    assert job.meta["raft"]["layers"][0]["infill_density"] == 10


def test_explicit_raft_patterns_use_opposite_legacy_zigzag_directions():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="gyroid")
    job = slice_mesh_to_job(mesh, config)

    add_raft_to_job(
        job,
        mesh,
        config,
        [
            RaftLayerConfig(
                outward_offset=2.0,
                layer_height=0.5,
                infill_density=100,
                infill_pattern="zigzag",
            ),
            RaftLayerConfig(
                outward_offset=1.0,
                layer_height=0.5,
                infill_density=70,
                infill_pattern="zigzag",
            ),
        ],
        top_gap=0.0,
    )

    first_roles = job.meta["path_roles"]["R"]["0"]
    second_roles = job.meta["path_roles"]["R"]["1"]
    first_infill = _paths_with_role(job.material_paths[0].paths, first_roles, "infill")
    second_infill = _paths_with_role(job.material_paths[1].paths, second_roles, "infill")

    assert first_infill
    assert second_infill
    assert _dominant_infill_angle(first_infill) == 45
    assert _dominant_infill_angle(second_infill) == -45
    assert job.meta["raft"]["top_gap"] == 0.0
    assert job.meta["raft"]["layers"] == [
        {
            "outward_offset": 2.0,
            "layer_height": 0.5,
            "infill_density": 100,
            "infill_pattern": "zigzag",
        },
        {
            "outward_offset": 1.0,
            "layer_height": 0.5,
            "infill_density": 70,
            "infill_pattern": "zigzag",
        },
    ]


def test_ui_raft_defaults_parse_chinese_zigzag_names():
    layers = _raft_layers_from_params(
        {"raft_infill_patterns": ["之字形,之字形"]},
        layer_count=2,
    )

    assert [layer.infill_pattern for layer in layers] == ["zigzag", "zigzag"]
    assert [layer.infill_density for layer in layers] == [100.0, 70.0]


def test_concentric_infill_generates_closed_inner_rings():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="concentric"),
    )

    infill = _paths_with_role(paths, roles, "infill")
    assert len(infill) >= 2
    assert all(path.shape[0] > 2 for path in infill)
    assert all(np.allclose(path[0], path[-1]) for path in infill)


def test_concentric_keeps_printable_residual_ring_centered():
    contour = np.asarray(
        [[0, 0], [21, 0], [21, 21], [0, 21], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="concentric"),
    )
    infill = _paths_with_role(paths, roles, "infill")
    last = infill[-1]
    center_x = (float(last[:, 0].min()) + float(last[:, 0].max())) * 0.5
    center_y = (float(last[:, 1].min()) + float(last[:, 1].max())) * 0.5

    assert np.isclose(center_x, 10.5, atol=0.05)
    assert np.isclose(center_y, 10.5, atol=0.05)


def test_concentric_keeps_closed_residual_ring_when_narrow():
    contour = np.asarray(
        [[0, 0], [21, 0], [21, 21], [0, 21], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="concentric"),
    )
    infill = _paths_with_role(paths, roles, "infill")

    assert infill
    assert all(np.allclose(path[0], path[-1]) for path in infill)


def test_concentric_keeps_fixed_spacing_before_residual_ring():
    offsets = _uniform_concentric_offsets(9.5, line_width=2.0, path_spacing=1.8)

    assert np.allclose(np.diff(offsets[:-1]), [1.8, 1.8, 1.8, 1.8])
    assert offsets[-1] == 9.5


def test_concentric_filters_paths_closer_than_line_width():
    outer = np.asarray([[0, 0], [20, 0]], dtype=np.float32)
    too_close = np.asarray([[0, 1], [20, 1]], dtype=np.float32)
    far_enough = np.asarray([[0, 2], [20, 2]], dtype=np.float32)

    filtered = _filter_concentric_paths_by_spacing([outer, too_close, far_enough], 2.0, 1e-5)

    assert len(filtered) == 2
    assert filtered[0] is outer
    assert filtered[1] is far_enough


def test_concentric_centers_residual_ring_per_local_region():
    contour = np.asarray(
        [[0, 0], [41, 0], [41, 13], [0, 13], [0, 0]],
        dtype=np.float32,
    )

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(layer_height=1.0, line_width=2.0, infill_pattern="concentric"),
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
        SliceConfig(layer_height=1.0, line_width=0.5, infill_pattern="aligned_rectilinear"),
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
        SliceConfig(layer_height=1.0, line_width=0.5, infill_pattern="aligned_rectilinear"),
    )

    for path, role in zip(paths, roles):
        if role == "infill":
            assert not _path_has_non_adjacent_crossing(path)


def test_zigzag_infill_connects_annulus_segments_without_crossing_hole():
    outer = _circle_path((0.0, 0.0), 20.0, 128)
    hole = _circle_path((0.0, 0.0), 10.0, 96)

    paths, roles = _build_resin_paths(
        [outer, hole],
        SliceConfig(layer_height=1.0, line_width=1.0, infill_pattern="aligned_rectilinear"),
    )
    infill = _paths_with_role(paths, roles, "infill")

    assert infill
    assert any(path.shape[0] > 2 for path in infill)
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
        SliceConfig(layer_height=1.0, line_width=0.5, infill_pattern="none"),
    )

    assert roles == ["outer_contour", "outer_contour", "inner_contour", "inner_contour"]


def test_rectilinear_infill_flips_angle_by_layer():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="rectilinear",
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
    # The legacy triangle optimizer may merge endpoint-connected paths.
    assert 0 < len(infill_paths) < 12
    assert all(path.shape[0] >= 2 for path in infill_paths)
    assert any(path.shape[0] > 2 for path in infill_paths)
    assert directions == {0, 60, 120}
    assert overlap_length < 1e-5
    assert longest_segment > 6.0


def test_legacy_triangle_path_optimization_reduces_open_travel():
    mesh = Mesh(_cube_triangles(size=30.0))

    def travel_for(config: SliceConfig) -> tuple[float, list[np.ndarray]]:
        job = slice_mesh_to_job(mesh, config)
        roles = job.meta["path_roles"]["R"]["0"]
        paths = [
            path
            for path, role in zip(job.material_paths[0].paths, roles)
            if role == "infill"
        ]
        travel = sum(
            float(np.linalg.norm(paths[index][0, :2] - paths[index - 1][-1, :2]))
            for index in range(1, len(paths))
        )
        return travel, paths

    disabled_travel, disabled_paths = travel_for(
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=70.0,
            triangle_path_optimization=False,
        )
    )
    enabled_travel, enabled_paths = travel_for(
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            infill_pattern="triangles",
            infill_density=70.0,
            triangle_path_optimization=True,
        )
    )

    assert disabled_paths
    assert len(enabled_paths) <= len(disabled_paths)
    assert enabled_travel <= disabled_travel


def test_legacy_zigzag_path_optimization_reduces_part_cap_paths():
    mesh = Mesh(_cube_triangles(size=30.0))

    def infill_counts(optimized: bool) -> list[int]:
        job = slice_mesh_to_job(
            mesh,
            SliceConfig(
                layer_height=5.0,
                line_width=2.0,
                infill_pattern="zigzag",
                infill_density=70.0,
                zigzag_path_optimization=optimized,
            ),
        )
        return [
            sum(role == "infill" for role in job.meta["path_roles"]["R"][str(group.layer_index)])
            for group in job.material_paths
        ]

    disabled = infill_counts(False)
    enabled = infill_counts(True)

    assert enabled[0] < disabled[0]
    assert enabled[-1] <= disabled[-1]
    assert all(current <= original for current, original in zip(enabled, disabled))


def test_triangle_path_optimizer_reorders_and_reverses_open_paths():
    paths = [
        np.asarray([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[10.0, 0.0, 0.5], [11.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[2.0, 0.0, 0.5], [3.0, 0.0, 0.5]], dtype=np.float32),
    ]
    optimized = optimize_triangle_infill_travel(paths)

    def travel(paths: list[np.ndarray]) -> float:
        return sum(
            float(np.linalg.norm(paths[index][0, :2] - paths[index - 1][-1, :2]))
            for index in range(1, len(paths))
        )

    assert travel(optimized) < travel(paths)
    assert sum(len(path) for path in optimized) == sum(len(path) for path in paths)


def test_triangle_path_optimizer_merges_only_sequentially_connected_paths():
    paths = [
        np.asarray([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[1.0, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[4.0, 0.0, 0.5], [5.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[5.0, 0.0, 0.5], [6.0, 0.0, 0.5]], dtype=np.float32),
    ]

    merged = merge_adjacent_connected_paths(paths)

    assert len(merged) == 2
    assert np.allclose(merged[0][:, :2], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert np.allclose(merged[1][:, :2], [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]])


def test_triangle_path_optimizer_merges_small_endpoint_gaps_within_tolerance():
    paths = [
        np.asarray([[0.0, 0.0, 0.5], [1.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[1.08, 0.0, 0.5], [2.0, 0.0, 0.5]], dtype=np.float32),
        np.asarray([[2.08, 0.0, 0.5], [3.0, 0.0, 0.5]], dtype=np.float32),
    ]

    merged = merge_adjacent_connected_paths(paths, tolerance=0.1)

    assert len(merged) == 1
    assert np.allclose(merged[0][0, :2], [0.0, 0.0])
    assert np.allclose(merged[0][-1, :2], [3.0, 0.0])


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
            triangle_path_optimization=False,
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


def test_slice_config_exposes_ui_tunable_path_parameters_in_meta():
    mesh = Mesh(_cube_triangles(size=10.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            line_width=1.0,
            infill_pattern="aligned_rectilinear",
            perimeter_count=3,
            smoothing_angle=120.0,
            smoothing_radius_factor=0.25,
        ),
    )

    slicing = job.meta["slicing"]
    assert slicing["perimeter_count"] == 3
    assert slicing["smoothing_angle"] == 120.0
    assert slicing["smoothing_radius_factor"] == 0.25
    assert slicing["slicing_kernel"] == "legacy"
    assert "forced_part_cap_layers" not in slicing


def test_slice_config_defaults_to_legacy_kernel():
    assert SliceConfig().slicing_kernel == "legacy"


def test_explicit_legacy_kernel_matches_default_kernel():
    mesh = Mesh(_cube_triangles(size=10.0))
    base_config = SliceConfig(layer_height=5.0, infill_pattern="aligned_rectilinear")

    default_job = slice_mesh_to_job(mesh, base_config)
    explicit_legacy_job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=5.0,
            infill_pattern="aligned_rectilinear",
            slicing_kernel="legacy",
        ),
    )

    assert default_job.meta == explicit_legacy_job.meta
    assert len(default_job.material_paths) == len(explicit_legacy_job.material_paths)
    for default_group, explicit_group in zip(
        default_job.material_paths,
        explicit_legacy_job.material_paths,
    ):
        assert default_group.layer_index == explicit_group.layer_index
        assert default_group.material == explicit_group.material
        assert len(default_group.paths) == len(explicit_group.paths)
        for default_path, explicit_path in zip(default_group.paths, explicit_group.paths):
            np.testing.assert_allclose(default_path, explicit_path)


def test_slice_config_rejects_unknown_kernel():
    try:
        SliceConfig(slicing_kernel="unknown")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "slicing_kernel" in str(exc)
    else:
        raise AssertionError("expected invalid slicing kernel to fail")


def test_pyslm_kernel_rejects_non_native_infill_patterns():
    from kuka_slicer.pyslm_backend import _validate_pyslm_config

    fallback_patterns = ("grid", "triangles", "gyroid", "concentric")
    for pattern in fallback_patterns:
        config = SliceConfig(
            layer_height=5.0,
            slicing_kernel="pyslm",
            infill_pattern=pattern,
        )
        try:
            _validate_pyslm_config(config)
        except ValueError as exc:
            assert "PySLM native kernel currently supports" in str(exc)
        else:
            raise AssertionError(f"expected PySLM to reject {pattern}")


def test_pyslm_config_exposes_native_defaults():
    config = SliceConfig().pyslm

    assert SliceConfig().triangle_path_optimization is True
    assert SliceConfig().zigzag_path_optimization is True
    assert config.hatcher == "basic"
    assert config.hatch_sort == "none"
    assert config.scan_contour_first is True
    assert config.fix_polygons is True


def test_recommended_geometry_tolerance_tracks_print_scale():
    assert recommended_geometry_tolerance(layer_height=0.5, line_width=2.0) == 0.0005
    assert recommended_geometry_tolerance(layer_height=0.001, line_width=2.0) == 1e-5
    assert recommended_geometry_tolerance(layer_height=50.0, line_width=20.0) == 0.01


def test_recommended_pyslm_strategy_defaults_follow_resin_scale():
    defaults = recommended_pyslm_strategy_defaults(layer_height=0.5, line_width=2.0)

    assert defaults.width == 10.0
    assert defaults.overlap == 0.1
    assert defaults.offset == 0.5

    smaller = recommended_pyslm_strategy_defaults(layer_height=0.1, line_width=1.0)
    assert smaller.width == 5.0
    assert math.isclose(smaller.overlap, 0.02)


def test_ui_uses_prusaslicer_style_infill_pattern_names():
    html = _index_html()

    for pattern in (
        "none",
        "rectilinear",
        "aligned_rectilinear",
        "line",
        "grid",
        "triangles",
        "gyroid",
        "concentric",
        "zigzag",
    ):
        assert f'value="{pattern}"' in html
    for legacy_pattern in (
        "lines_x",
        "lines_y",
        "alternating_diagonal",
    ):
        assert legacy_pattern not in html

    for translated_label in (
        "仅轮廓",
        "交替直线填充",
        "对齐直线填充",
        "单向线填充",
        "网格填充",
        "三角形填充",
        "陀螺曲线填充",
        "同心轮廓填充",
        "之字形填充",
    ):
        assert translated_label in html

    for untranslated_label in (
        "None (perimeter only)",
        "Rectilinear",
        "Aligned Rectilinear",
        "Triangles",
        "Gyroid",
        "Concentric",
        "Zig Zag",
    ):
        assert untranslated_label not in html


def test_ui_exposes_slicing_kernel_input():
    html = _index_html()

    for translated_label in (
        "切片内核",
        "原始内核（稳定）",
        "PySLM（实验）",
        "PySLM 原生扫描参数",
        "PySLM 填充策略",
        "条带/岛状参数（自动）",
        "自动设置条带/岛状参数",
        "扫描线排序",
        "填充角度 °",
        "层间角度增量 °",
        "填充线间距 mm",
        "轮廓偏移 mm",
        "光斑补偿 mm",
        "体积填充偏移 mm",
        "外轮廓数量",
        "内轮廓数量",
        "条带宽度 mm",
        "条带平移系数",
        "岛状宽度 mm",
        "岛状平移系数",
        "切层边界简化 mm",
        "简化模式",
        "轮廓优先扫描",
        "修复切层多边形",
        "保持拓扑结构",
        "原始内核填充路径",
        "三角形填充路径优化",
        "之字形填充路径优化",
    ):
        assert translated_label in html

    for untranslated_label in (
        "Slicing kernel",
        "Legacy (stable)",
        "PySLM (experimental)",
        "PySLM native settings",
        "Hatcher strategy",
        "树脂填充路径",
        "Hatch sort",
        "Hatch angle deg",
        "Layer angle increment deg",
        "Hatch distance mm",
        "Contour offset mm",
        "Spot compensation mm",
        "Volume offset hatch mm",
        "Outer contours",
        "Inner contours",
        "Slice simplification mm",
        "Simplification mode",
        "Scan contours first",
        "Fix slice polygons",
        "Preserve topology",
    ):
        assert untranslated_label not in html

    for control_id in (
        "stlFile",
        "fiberJsonFile",
        "layerHeight",
        "buildAxis",
        "zMin",
        "zMax",
        "tolerance",
        "lineWidth",
        "perimeterCount",
        "infillPattern",
        "infillDensity",
        "infillOverlap",
        "trianglePathOptimization",
        "zigzagPathOptimization",
        "slicingKernel",
        "pyslmNativeSettings",
        "pyslmPatternSettings",
        "pyslmPatternAuto",
        "legacyInfillControl",
        "pyslmHatcher",
        "pyslmHatchSort",
        "pyslmHatchAngle",
        "pyslmLayerAngleIncrement",
        "pyslmHatchDistance",
        "pyslmContourOffset",
        "pyslmSpotCompensation",
        "pyslmVolumeOffset",
        "pyslmOuterContours",
        "pyslmInnerContours",
        "pyslmStripeWidth",
        "pyslmStripeOverlap",
        "pyslmStripeOffset",
        "pyslmIslandWidth",
        "pyslmIslandOverlap",
        "pyslmIslandOffset",
        "pyslmSimplificationFactor",
        "pyslmSimplificationMode",
        "pyslmScanContourFirst",
        "pyslmFixPolygons",
        "pyslmSimplificationPreserveTopology",
        "smoothingAngle",
        "smoothingRadiusFactor",
        "raftLayerCount",
        "raftTopGap",
        "raftOffsets",
        "raftLayerHeights",
        "raftInfillDensities",
        "raftInfillPatterns",
        "curveMode",
        "curveAmplitude",
        "curvePeriod",
    ):
        assert f'id="{control_id}"' in html
    assert 'value="legacy" selected' in html
    assert 'value="pyslm"' in html
    assert 'id="trianglePathOptimization" type="checkbox" checked' in html
    assert 'id="raftTopGap" name="raftTopGap" type="number" min="0" step="0.001" value="0"' in html
    assert 'id="raftInfillDensities" name="raftInfillDensities" type="text" value="100,70"' in html
    assert 'id="raftInfillPatterns" name="raftInfillPatterns" type="text" value="之字形,之字形"' in html
    for hatcher in ("basic", "stripe", "island", "basic_island"):
        assert f'value="{hatcher}"' in html
    assert "bottomCapAngle" not in html
    assert "topCapAngle" not in html


def test_preview_simplification_keeps_contour_corners():
    points = [[0.0, 0.0, 0.5], [0.0, 20.0, 0.5]]
    points.extend([[float(x), 20.0, 0.5] for x in range(1, 202)])
    points.append([0.0, 0.0, 0.5])

    simplified = _simplify_preview_path(points, max_points=2000)

    assert [0.0, 20.0, 0.5] in simplified


def test_infill_paths_connect_safe_neighbors_after_generation():
    geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    config = SliceConfig(layer_height=1.0, line_width=1.0, infill_pattern="aligned_rectilinear")

    paths = _raft_zigzag_infill_paths(geometry, config, infill_density=100.0)

    assert 1 <= len(paths) <= 2
    assert any(path.shape[0] > 2 for path in paths)
    assert all(geometry.buffer(0.2).covers(LineString(path[:, :2])) for path in paths)


def test_resin_path_connector_keeps_closed_paths_separate():
    geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    closed = np.asarray(
        [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]],
        dtype=np.float32,
    )
    paths = _connect_resin_infill_paths(
        [
            closed,
            np.asarray([[1, 8], [19, 8]], dtype=np.float32),
            np.asarray([[19, 9], [1, 9]], dtype=np.float32),
        ],
        geometry,
        spacing=1.0,
        tolerance=1e-5,
    )

    assert any(np.array_equal(path, closed) for path in paths)
    assert any(path.shape[0] > 2 for path in paths if not np.array_equal(path, closed))


def test_raft_layers_shift_part_layers_and_z():
    mesh = Mesh(_cube_triangles(size=10.0))
    config = SliceConfig(layer_height=5.0, line_width=1.0, infill_pattern="aligned_rectilinear")
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
    config = SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="concentric")
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
    assert len(raft_infill) > 1
    assert max(path.shape[0] for path in raft_infill) >= 20
    assert all(not _path_has_non_adjacent_crossing(path) for path in raft_infill)


def test_only_top_raft_layer_touching_part_uses_lattice():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="concentric")
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
    assert max(path.shape[0] for path in top_infill) >= 20
    assert all(not _path_has_non_adjacent_crossing(path) for path in top_infill)


def test_raft_lattice_density_changes_spacing():
    geometry = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    dense = _raft_lattice_infill_paths(geometry, config, infill_density=100.0)
    sparse = _raft_lattice_infill_paths(geometry, config, infill_density=50.0)

    assert dense
    assert sparse
    assert _infill_total_length(dense) > _infill_total_length(sparse)


def test_raft_zigzag_infill_segments_do_not_add_connector_routes():
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
    job = slice_mesh_to_job(mesh, SliceConfig(layer_height=5.0, infill_pattern="none"))

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


def _has_infill_direction(paths: list[np.ndarray], expected_angle: float) -> bool:
    for path in paths:
        for delta in np.diff(path[:, :2], axis=0):
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0
            if abs(angle - expected_angle) < 2.0:
                return True
    return False


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
