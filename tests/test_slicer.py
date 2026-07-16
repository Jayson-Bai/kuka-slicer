import math

import numpy as np
import pytest
from shapely import maximum_inscribed_circle
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

import kuka_slicer.slicer as slicer_module
from kuka_slicer.external_npz import ExternalSourceJob, MaterialPaths
from kuka_slicer.slicer import (
    DEFAULT_FIBER_LAYER_HEIGHT_MM,
    DEFAULT_FIBER_LINE_WIDTH_MM,
    DEFAULT_RESIN_INFILL_OVERLAP_PERCENT,
    DEFAULT_RESIN_LAYER_HEIGHT_MM,
    DEFAULT_RESIN_LINE_WIDTH_MM,
    RaftLayerConfig,
    SliceConfig,
    _build_resin_paths,
    _centerline_connector_is_clear,
    _concentric_infill_geometry,
    _connect_concentric_infill_paths,
    _connect_resin_infill_paths,
    _filter_concentric_paths_by_spacing,
    _finish_solid_fill_paths,
    _libslic3r_fill_surface_overlap_offset,
    _perimeter_paths_from_geometry,
    _raft_geometry_for_layer,
    _raft_lattice_infill_paths,
    _raft_reserved_void_geometry,
    _raft_zigzag_infill_paths,
    _resin_infill_surface_geometry,
    _residual_correction_has_sufficient_novel_area,
    _smooth_path_corners_into_paths,
    _smooth_path_corners,
    _solid_spacing_adjustment_limit,
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
    _preview_payload,
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
    scan_ys = _horizontal_scan_ys_for_paths(infill_paths)
    assert np.allclose(scan_ys, [4.6, 6.4, 8.2, 10.0, 11.8, 13.6, 15.4])
    assert np.allclose(np.diff(scan_ys), 1.8)


def test_resin_line_infill_can_disable_overlap_for_legacy_spacing():
    mesh = Mesh(_cube_triangles(size=20.0))

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="aligned_rectilinear", infill_overlap=0.0),
    )

    roles = job.meta["path_roles"]["R"]["1"]
    infill_paths = _paths_with_role(job.material_paths[1].paths, roles, "infill")
    assert len(infill_paths) == 1
    scan_ys = _horizontal_scan_ys_for_paths(infill_paths)
    assert np.allclose(scan_ys, [5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    assert np.allclose(np.diff(scan_ys), 2.0)


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


def test_libslic3r_fill_surface_overlap_offset_uses_physical_line_width():
    offset = _libslic3r_fill_surface_overlap_offset(
        line_width=2.0,
        overlap_percent=10.0,
    )

    assert np.isclose(offset, -0.8)


def test_resin_infill_surface_uses_last_perimeter_and_overlap_offset():
    geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    infill_surface = _resin_infill_surface_geometry(geometry, config)

    min_x, min_y, max_x, max_y = infill_surface.bounds
    assert np.isclose(min_x, 4.6, atol=0.05)
    assert np.isclose(min_y, 4.6, atol=0.05)
    assert np.isclose(max_x, 15.4, atol=0.05)
    assert np.isclose(max_y, 15.4, atol=0.05)


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
    roi = Polygon([(6, 6), (14, 6), (14, 14), (6, 14)])
    dense_length = unary_union([LineString(path[:, :2]) for path in dense_infill]).intersection(roi).length
    sparse_length = unary_union([LineString(path[:, :2]) for path in sparse_infill]).intersection(roi).length
    assert dense_infill
    assert sparse_infill
    assert dense_length > sparse_length * 1.7
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
    assert _has_infill_direction(bottom_infill, 0.0)
    assert _has_infill_direction(top_infill, 45.0)
    assert job.meta["slicing"]["infill_density"] == 0
    cap_layers = job.meta["slicing"]["part_cap_layers"]
    assert cap_layers["bottom"] == 0
    assert cap_layers["top"] == top_index
    assert cap_layers["infill_pattern"] == "zigzag"
    assert cap_layers["infill_density"] == 100.0
    assert cap_layers["bottom_angle_degrees"] == 0.0
    assert cap_layers["top_angle_degrees"] == 45.0


def test_isotropic_infill_repeats_four_direction_zigzag_schedule():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    mesh = Mesh(triangles)
    config = SliceConfig(
        layer_height=0.5,
        line_width=2.0,
        infill_pattern="isotropic",
        infill_density=60.0,
    )

    job = slice_mesh_to_job(mesh, config)

    expected_angles = [0.0, 45.0, 0.0, 135.0, 90.0, 45.0, 0.0, 135.0, 90.0, 45.0]
    assert [group.layer_index for group in job.material_paths] == list(range(10))
    for group, expected_angle in zip(job.material_paths, expected_angles):
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        infill = _paths_with_role(group.paths, roles, "infill")
        assert infill
        assert _has_infill_direction(infill, expected_angle)

    schedule = job.meta["slicing"]["isotropic_schedule"]
    assert schedule["repeat_count"] == 2
    assert schedule["repeat_angles_degrees"] == [45.0, 0.0, -45.0, 90.0]
    assert schedule["layer_angles_degrees"] == [
        0.0,
        45.0,
        0.0,
        -45.0,
        90.0,
        45.0,
        0.0,
        -45.0,
        90.0,
        45.0,
    ]


def test_constant_section_isotropic_plans_each_effective_angle_once_and_copies_layers(
    monkeypatch,
):
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    mesh = Mesh(triangles)
    planned_angles: list[float | None] = []
    original_build_resin_paths = slicer_module._build_resin_paths

    def counting_build_resin_paths(*args, **kwargs):
        planned_angles.append(kwargs.get("forced_zigzag_angle"))
        return original_build_resin_paths(*args, **kwargs)

    monkeypatch.setattr(
        slicer_module,
        "_build_resin_paths",
        counting_build_resin_paths,
    )

    job = slice_mesh_to_job(
        mesh,
        SliceConfig(
            layer_height=0.5,
            line_width=2.0,
            infill_pattern="isotropic",
            infill_density=100.0,
        ),
    )

    assert len(planned_angles) == 4
    assert set(planned_angles) == {0.0, 45.0, -45.0, 90.0}
    assert len(job.material_paths) == 10

    first_zero_degree_layer = job.material_paths[0]
    repeated_zero_degree_layer = job.material_paths[2]
    assert len(first_zero_degree_layer.paths) == len(repeated_zero_degree_layer.paths)
    for first_path, repeated_path in zip(
        first_zero_degree_layer.paths,
        repeated_zero_degree_layer.paths,
    ):
        assert np.allclose(first_path[:, :2], repeated_path[:, :2])
        assert not np.shares_memory(first_path, repeated_path)

    repeated_snapshot = repeated_zero_degree_layer.paths[0].copy()
    first_zero_degree_layer.paths[0][0, 0] += 123.0
    assert np.array_equal(repeated_zero_degree_layer.paths[0], repeated_snapshot)


def test_preview_payload_uses_slim_role_aware_layer_schema_and_complete_bounds():
    resin_outer = np.asarray(
        [[0, 0, 0.5], [10, 0, 0.5], [10, 10, 0.5], [0, 10, 0.5], [0, 0, 0.5]],
        dtype=np.float32,
    )
    resin_inner = np.asarray(
        [[3, 3, 0.5], [7, 3, 0.5], [7, 7, 0.5], [3, 7, 0.5], [3, 3, 0.5]],
        dtype=np.float32,
    )
    resin_infill = np.asarray(
        [[1, 5, 0.5], [9, 5, 0.5]],
        dtype=np.float32,
    )
    fiber = np.asarray(
        [[-2, 2, 0.6], [12, 2, 0.6]],
        dtype=np.float32,
    )
    job = ExternalSourceJob(
        material_paths=[
            MaterialPaths(0, "R", [resin_outer, resin_inner, resin_infill]),
            MaterialPaths(0, "F", [fiber]),
        ],
        meta={
            "path_roles": {
                "R": {"0": ["outer_contour", "inner_contour", "infill"]}
            }
        },
    )

    preview = _preview_payload(
        Mesh(_cube_triangles(size=10.0)),
        SliceConfig(line_width=2.0),
        job,
    )

    assert set(preview) == {"bounds", "line_widths", "layers"}
    assert len(preview["layers"]) == 1
    layer = preview["layers"][0]
    assert set(layer) == {"index", "resin_paths", "fiber_paths"}
    assert layer["index"] == 0
    assert [entry["role"] for entry in layer["resin_paths"]] == [
        "outer_contour",
        "inner_contour",
        "infill",
    ]
    assert layer["resin_paths"][2]["points"] == resin_infill.tolist()
    assert layer["fiber_paths"] == [fiber.tolist()]
    assert preview["bounds"] == {
        "min_x": -2.0,
        "max_x": 12.0,
        "min_y": 0.0,
        "max_y": 10.0,
        "min_z": 0.5,
        "max_z": pytest.approx(0.6),
    }


def test_isotropic_infill_explicit_z_bounds_keep_four_direction_schedule():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    job = slice_mesh_to_job(
        Mesh(triangles),
        SliceConfig(
            layer_height=0.5,
            line_width=2.0,
            infill_pattern="isotropic",
            infill_density=60.0,
            z_min=0.0,
            z_max=5.0,
        ),
    )

    assert [group.layer_index for group in job.material_paths] == list(range(10))
    assert np.allclose(
        [group.paths[0][0, 2] for group in job.material_paths],
        np.arange(0.5, 5.5, 0.5),
    )
    assert job.meta["slicing"]["isotropic_schedule"]["repeat_count"] == 2


def test_isotropic_infill_with_raft_has_complete_fixed_layer_sequence():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    mesh = Mesh(triangles)
    config = SliceConfig(
        layer_height=0.5,
        line_width=2.0,
        infill_pattern="isotropic",
        infill_density=60.0,
    )
    job = slice_mesh_to_job(mesh, config)

    add_raft_to_job(
        job,
        mesh,
        config,
        [
            RaftLayerConfig(outward_offset=2.0, infill_density=60.0),
            RaftLayerConfig(outward_offset=1.0, infill_density=60.0),
        ],
        top_gap=0.0,
    )

    expected_angles = [90.0, 135.0, 0.0, 45.0, 0.0, 135.0, 90.0, 45.0, 0.0, 135.0, 90.0, 45.0]
    assert [group.layer_index for group in job.material_paths] == list(range(12))
    for group, expected_angle in zip(job.material_paths, expected_angles):
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        infill = _paths_with_role(group.paths, roles, "infill")
        assert infill
        assert _has_infill_direction(infill, expected_angle)


def test_isotropic_infill_rejects_incomplete_height_cycle():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.2
    mesh = Mesh(triangles)

    try:
        slice_mesh_to_job(
            mesh,
            SliceConfig(layer_height=0.5, infill_pattern="isotropic"),
        )
    except ValueError as exc:
        assert "2N+1 mm" in str(exc)
        assert "拒绝切片" in str(exc)
    else:
        raise AssertionError("expected incomplete isotropic height cycle to fail")


def test_isotropic_infill_rejects_non_default_layer_height():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    mesh = Mesh(triangles)

    try:
        slice_mesh_to_job(
            mesh,
            SliceConfig(layer_height=0.4, infill_pattern="isotropic"),
        )
    except ValueError as exc:
        assert "0.5 mm" in str(exc)
        assert "拒绝切片" in str(exc)
    else:
        raise AssertionError("expected isotropic layer height validation to fail")


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
    assert _has_infill_direction(raft_infill, 90.0)
    assert _has_infill_direction(part_bottom_infill, 0.0)
    assert job.meta["raft"]["layers"][0]["infill_density"] == 100.0
    assert job.meta["raft"]["top_gap"] == 0.0


def test_raft_ignores_user_patterns_and_uses_fixed_zigzag_directions():
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
    assert _has_infill_direction(first_infill, 90.0)
    assert _dominant_infill_angle(second_infill) == -45
    assert job.meta["raft"]["top_gap"] == 0.0
    assert job.meta["raft"]["layers"] == [
        {
            "outward_offset": 2.0,
            "layer_height": 0.5,
            "infill_density": 100.0,
            "infill_pattern": "zigzag",
            "angle_degrees": 90.0,
        },
        {
            "outward_offset": 1.0,
            "layer_height": 0.5,
            "infill_density": 100.0,
            "infill_pattern": "zigzag",
            "angle_degrees": -45.0,
        },
    ]


def test_ui_raft_params_only_accept_offsets():
    layers = _raft_layers_from_params({"raft_offsets": ["2,1"]})

    assert [layer.outward_offset for layer in layers] == [2.0, 1.0]
    assert [layer.infill_pattern for layer in layers] == [None, None]
    assert [layer.infill_density for layer in layers] == [100.0, 100.0]


def test_concentric_infill_connects_all_rings_without_losing_coverage():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        infill_pattern="concentric",
        smoothing_radius_factor=0.0,
    )
    surface = _resin_infill_surface_geometry(Polygon(contour[:-1]), config)
    raw_rings = _concentric_infill_geometry(
        surface,
        config.line_width,
        config.line_width * (1.0 - config.infill_overlap / 100.0),
        config.tolerance,
    )

    paths, roles = _build_resin_paths([contour], config)

    infill = _paths_with_role(paths, roles, "infill")
    raw_linework = unary_union([LineString(path) for path in raw_rings])
    planned_line = LineString(infill[0])

    assert len(raw_rings) >= 2
    assert len(infill) == 1
    assert not np.allclose(infill[0][0], infill[0][-1])
    assert raw_linework.difference(planned_line.buffer(1e-4)).length <= 1e-3
    assert planned_line.length >= raw_linework.length - 1e-3


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

    assert len(infill) == 1
    assert not np.allclose(infill[0][0], infill[0][-1])


def test_concentric_keeps_fixed_spacing_before_residual_ring():
    offsets = _uniform_concentric_offsets(9.5, line_width=2.0, path_spacing=1.8)

    assert offsets[0] == 0.0
    assert np.allclose(np.diff(offsets[:-1]), [1.8, 1.8, 1.8, 1.8, 1.8])
    assert offsets[-1] == 9.5


def test_concentric_filters_paths_closer_than_line_width():
    outer = np.asarray([[0, 0], [20, 0]], dtype=np.float32)
    too_close = np.asarray([[0, 1.7], [20, 1.7]], dtype=np.float32)
    far_enough = np.asarray([[0, 1.8], [20, 1.8]], dtype=np.float32)

    filtered = _filter_concentric_paths_by_spacing([outer, too_close, far_enough], 1.8, 1e-5)

    assert len(filtered) == 2
    assert filtered[0] is outer
    assert filtered[1] is far_enough


def test_concentric_concave_offsets_keep_pitch_and_connect_continuously():
    geometry = Polygon(
        [
            (0, 0),
            (50, 0),
            (50, 10),
            (20, 10),
            (20, 40),
            (50, 40),
            (50, 50),
            (0, 50),
        ]
    )

    rings = _concentric_infill_geometry(
        geometry,
        line_width=2.0,
        path_spacing=1.8,
        tolerance=1e-5,
    )
    planned = _connect_concentric_infill_paths(
        rings,
        geometry,
        spacing=1.8,
        minimum_clearance=1.8,
        tolerance=1e-5,
    )

    assert np.allclose([float(path[:, 0].min()) for path in rings], np.arange(0.0, 10.8, 1.8))
    assert len(planned) == 1
    assert unary_union([LineString(path) for path in rings]).difference(
        LineString(planned[0]).buffer(1e-4)
    ).length <= 1e-3


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


def test_zigzag_post_optimization_stays_in_bead_aware_concave_corridor():
    contour = np.asarray(
        [[0, 0], [20, 0], [20, 8], [12, 8], [12, 14], [20, 14], [20, 20], [0, 20], [0, 0]],
        dtype=np.float32,
    )
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        zigzag_path_optimization=True,
    )
    safe_corridor = _resin_infill_surface_geometry(Polygon(contour[:-1]), config)

    paths, roles = _build_resin_paths([contour], config, layer_index=1)

    for path, role in zip(paths, roles):
        if role == "infill":
            assert safe_corridor.buffer(1e-5).covers(LineString(path[:, :2]))


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
    first_layer_paths = _paths_with_role(job.material_paths[1].paths, first_roles, "infill")
    second_layer_paths = _paths_with_role(job.material_paths[2].paths, second_roles, "infill")
    assert _diagonal_segment_sign(first_layer_paths) < 0
    assert _diagonal_segment_sign(second_layer_paths) > 0


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
    assert len(infill_paths) <= 6
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


def test_legacy_zigzag_path_optimization_keeps_part_cap_paths_continuous():
    mesh = Mesh(_cube_triangles(size=30.0))

    def job_for(optimized: bool):
        return slice_mesh_to_job(
            mesh,
            SliceConfig(
                layer_height=5.0,
                line_width=2.0,
                infill_pattern="zigzag",
                infill_density=70.0,
                zigzag_path_optimization=optimized,
            ),
        )

    def infill_counts(optimized: bool) -> list[int]:
        job = job_for(optimized)
        return [
            sum(role == "infill" for role in job.meta["path_roles"]["R"][str(group.layer_index)])
            for group in job.material_paths
        ]

    def first_layer_travel(optimized: bool) -> float:
        job = job_for(optimized)
        group = job.material_paths[0]
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        paths = [path for path, role in zip(group.paths, roles) if role == "infill"]
        return sum(
            float(np.linalg.norm(paths[index][0, :2] - paths[index - 1][-1, :2]))
            for index in range(1, len(paths))
        )

    disabled = infill_counts(False)
    enabled = infill_counts(True)

    assert enabled[0] <= disabled[0]
    assert enabled[-1] <= disabled[-1]
    assert all(current <= original for current, original in zip(enabled, disabled))
    assert first_layer_travel(True) <= first_layer_travel(False)

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
    assert np.allclose(
        merged[0][:, :2],
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.08, 0.0],
            [2.0, 0.0],
            [2.08, 0.0],
            [3.0, 0.0],
        ],
    )


def test_legacy_triangle_path_optimization_preserves_planned_linework():
    contour = _square_contour(40.0)

    def infill_linework(optimized: bool):
        paths, roles = _build_resin_paths(
            [contour],
            SliceConfig(
                layer_height=1.0,
                line_width=2.0,
                infill_pattern="triangles",
                infill_density=70.0,
                triangle_path_optimization=optimized,
                smoothing_radius_factor=0.0,
            ),
            layer_index=1,
        )
        return unary_union(
            [
                LineString(path[:, :2])
                for path, role in zip(paths, roles)
                if role == "infill"
            ]
        )

    original = infill_linework(False)
    optimized = infill_linework(True)

    assert original.difference(optimized.buffer(1e-5)).length <= 1e-4
    assert optimized.difference(original.buffer(1e-5)).length <= 1e-4


def test_zero_smoothing_radius_keeps_zigzag_turns_sharp():
    paths, roles = _build_resin_paths(
        [_square_contour(20.0)],
        SliceConfig(
            layer_height=1.0,
            line_width=2.0,
            infill_pattern="aligned_rectilinear",
            infill_density=50.0,
            smoothing_radius_factor=0.0,
        ),
        layer_index=1,
    )
    infill = _paths_with_role(paths, roles, "infill")

    assert infill
    deltas = np.vstack([np.diff(path[:, :2], axis=0) for path in infill])
    assert all(abs(float(delta[0])) <= 1e-6 or abs(float(delta[1])) <= 1e-6 for delta in deltas)
    assert any(
        abs(float(np.dot(first, second))) <= 1e-6
        for path in infill
        for first, second in zip(np.diff(path[:, :2], axis=0), np.diff(path[:, :2], axis=0)[1:])
    )


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


@pytest.mark.parametrize(
    ("pattern", "max_path_count"),
    [
        ("rectilinear", 1),
        ("aligned_rectilinear", 1),
        ("line", 1),
        ("grid", 2),
        ("triangles", 8),
        ("gyroid", 30),
        ("concentric", 1),
        ("zigzag", 1),
    ],
)
def test_prusa_infill_patterns_bound_interruptions_without_retracing(
    pattern: str,
    max_path_count: int,
):
    contour = _square_contour(40.0)

    paths, roles = _build_resin_paths(
        [contour],
        SliceConfig(
            layer_height=1.0,
            line_width=2.0,
            infill_pattern=pattern,
            infill_density=100.0,
        ),
        layer_index=1,
    )
    infill = _paths_with_role(paths, roles, "infill")

    assert infill
    assert len(infill) <= max_path_count
    assert all(path.shape[0] >= 2 for path in infill)


@pytest.mark.parametrize(
    (
        "angle",
        "minimum_coverage",
        "maximum_void_diameter",
        "maximum_path_count",
    ),
    [
        (0.0, 0.997, 0.50, 2),
        (45.0, 0.997, 0.50, 3),
        (-45.0, 0.997, 0.50, 2),
        (90.0, 0.997, 0.50, 2),
    ],
)
def test_prusa_full_density_round_beads_cover_wall_transition_without_piling(
    angle: float,
    minimum_coverage: float,
    maximum_void_diameter: float,
    maximum_path_count: int,
):
    outer = np.asarray(
        [[0, 0], [60, 0], [60, 40], [0, 40], [0, 0]],
        dtype=np.float32,
    )
    hole = np.asarray(
        [[25, 15], [35, 15], [35, 25], [25, 25], [25, 15]],
        dtype=np.float32,
    )
    solid = Polygon(outer[:-1], holes=[hole[:-1]])
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=100.0,
        infill_overlap=10.0,
    )

    paths, roles = _build_resin_paths(
        [outer, hole],
        config,
        layer_index=1,
        forced_zigzag_angle=angle,
    )
    infill = _paths_with_role(paths, roles, "infill")
    last_walls = _paths_with_role(paths, roles, "inner_contour")
    full_width_stroke = _round_bead_union(paths, config.line_width)
    infill_linework = unary_union([LineString(path[:, :2]) for path in infill])
    repeated_length = sum(LineString(path[:, :2]).length for path in infill) - infill_linework.length
    expected_pitch = config.line_width * (1.0 - config.infill_overlap / 100.0)
    pitch_adjustment = _solid_spacing_adjustment_limit(
        expected_pitch,
        config.line_width,
    )
    wall_clearance = infill_linework.distance(
        unary_union([LineString(path[:, :2]) for path in last_walls])
    )
    physical_centerline_region = solid.buffer(
        -(config.line_width * 0.5 - config.tolerance * 2.0),
        join_style="round",
    )
    coverage = solid.intersection(full_width_stroke).area / solid.area
    scan_gaps = _hatch_scan_level_gaps(infill, angle, config.line_width)

    assert coverage >= minimum_coverage
    assert _maximum_uncovered_void_diameter(
        solid,
        full_width_stroke,
        edge_inset=0.3,
    ) <= maximum_void_diameter
    assert _infill_extra_dose_ratio(infill, solid, config.line_width) <= 0.01
    assert _segment_bead_overlap_ratio(infill, config.line_width) <= 0.23
    assert scan_gaps
    assert min(scan_gaps) >= expected_pitch - pitch_adjustment - 0.02
    assert max(scan_gaps) <= expected_pitch + pitch_adjustment + 0.02
    assert abs(repeated_length) <= 1e-4
    assert wall_clearance == pytest.approx(expected_pitch, abs=0.02)
    assert len(infill) <= maximum_path_count
    assert all(LineString(path[:, :2]).is_simple for path in infill)
    minimum_vertex_angles = [
        _minimum_nondegenerate_vertex_angle(
            path,
            minimum_segment_length=config.line_width * 0.005,
        )
        for path in infill
    ]
    assert min(minimum_vertex_angles, default=180.0) >= config.smoothing_angle - 1e-3
    assert all(
        physical_centerline_region.covers(LineString(path[:, :2]))
        for path in infill
    )


def _scalar_effective_path_corner_candidates(
    path: np.ndarray,
    minimum_span: float,
    angle_threshold_degrees: float,
    tolerance: float,
) -> list[tuple[float, int, int, int]]:
    """Reference implementation retained to guard vectorized equivalence."""

    points = np.asarray(path[:, :2], dtype=np.float64)
    candidates: list[tuple[float, int, int, int]] = []
    for index in range(1, points.shape[0] - 1):
        previous_index = index - 1
        while (
            previous_index >= 0
            and float(np.linalg.norm(points[index] - points[previous_index]))
            < minimum_span
        ):
            previous_index -= 1
        next_index = index + 1
        while (
            next_index < points.shape[0]
            and float(np.linalg.norm(points[next_index] - points[index]))
            < minimum_span
        ):
            next_index += 1
        if previous_index < 0 or next_index >= points.shape[0]:
            continue
        incoming = points[previous_index] - points[index]
        outgoing = points[next_index] - points[index]
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if min(incoming_length, outgoing_length) <= tolerance:
            continue
        cosine = float(
            np.clip(
                np.dot(incoming, outgoing)
                / (incoming_length * outgoing_length),
                -1.0,
                1.0,
            )
        )
        angle = math.degrees(math.acos(cosine))
        if angle < angle_threshold_degrees - 1e-6:
            candidates.append((angle, index, previous_index, next_index))
    candidates.sort(key=lambda item: item[0])
    return candidates


@pytest.mark.parametrize(
    ("path", "minimum_span", "angle_threshold_degrees", "tolerance"),
    [
        pytest.param(
            np.asarray(
                [
                    [0.0, 0.0],
                    [10.0, 0.0],
                    [10.002, 0.0],
                    [10.004, 0.0],
                    [10.006, 0.0],
                    [10.006, 10.0],
                    [20.0, 10.0],
                ],
                dtype=np.float32,
            ),
            0.01,
            150.0,
            5e-4,
            id="micro-segment-chain",
        ),
        pytest.param(
            np.asarray(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                ],
                dtype=np.float64,
            ),
            0.01,
            120.0,
            1e-6,
            id="duplicate-degenerate-points",
        ),
        pytest.param(
            np.empty((0, 2), dtype=np.float32),
            0.01,
            150.0,
            5e-4,
            id="empty-path",
        ),
        pytest.param(
            np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
            0.01,
            150.0,
            5e-4,
            id="two-point-path",
        ),
    ],
)
def test_effective_corner_candidates_vectorized_matches_scalar_edge_cases(
    path: np.ndarray,
    minimum_span: float,
    angle_threshold_degrees: float,
    tolerance: float,
):
    assert slicer_module._effective_path_corner_candidates(
        path,
        minimum_span,
        angle_threshold_degrees,
        tolerance,
    ) == _scalar_effective_path_corner_candidates(
        path,
        minimum_span,
        angle_threshold_degrees,
        tolerance,
    )


def test_effective_corner_candidates_preserve_exact_minimum_span_boundary():
    delta = np.asarray([0.90535587, 0.44637457], dtype=np.float64)
    path = np.asarray(
        [
            [0.0, 0.0],
            delta,
            delta + np.asarray([0.25, 1.0]),
            delta + np.asarray([1.25, 1.2]),
        ],
        dtype=np.float64,
    )
    exact_span = float(np.linalg.norm(delta))

    for minimum_span in (
        np.nextafter(exact_span, -math.inf),
        exact_span,
        np.nextafter(exact_span, math.inf),
    ):
        assert slicer_module._effective_path_corner_candidates(
            path,
            minimum_span,
            150.0,
            1e-9,
        ) == _scalar_effective_path_corner_candidates(
            path,
            minimum_span,
            150.0,
            1e-9,
        )


def test_effective_corner_candidates_vectorized_matches_scalar_randomized():
    rng = np.random.default_rng(492910)
    for case_index in range(250):
        point_count = int(rng.integers(0, 96))
        steps = rng.normal(size=(point_count, 2))
        steps[rng.random(point_count) < 0.18] *= 1e-4
        steps[rng.random(point_count) < 0.08] = 0.0
        path = np.cumsum(steps, axis=0)
        path = path.astype(np.float32 if case_index % 2 else np.float64)
        minimum_span = float(10 ** rng.uniform(-4.0, 0.3))
        if point_count > 1 and case_index % 5 == 0:
            boundary_index = int(rng.integers(1, point_count))
            minimum_span = float(
                np.linalg.norm(path[boundary_index] - path[boundary_index - 1])
            )
        angle_threshold_degrees = float(rng.uniform(1.0, 179.0))
        tolerance = float(10 ** rng.uniform(-8.0, -3.0))

        expected = _scalar_effective_path_corner_candidates(
            path,
            minimum_span,
            angle_threshold_degrees,
            tolerance,
        )
        actual = slicer_module._effective_path_corner_candidates(
            path,
            minimum_span,
            angle_threshold_degrees,
            tolerance,
        )
        assert actual == expected, f"vectorized mismatch in random case {case_index}"


def test_solid_fill_finisher_removes_micro_segment_without_leaving_hard_corner():
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=100.0,
        infill_overlap=10.0,
        smoothing_angle=150.0,
        smoothing_radius_factor=0.35,
    )
    allowed = Polygon([(-5, -5), (20, -5), (20, 20), (-5, 20)])
    micro_segment_then_hard_turn = np.asarray(
        [[0, 0], [10, 0], [10.005, 0], [10.005, 10]],
        dtype=np.float32,
    )
    assert float(
        np.linalg.norm(
            micro_segment_then_hard_turn[2] - micro_segment_then_hard_turn[1]
        )
    ) < 0.01
    assert _minimum_nondegenerate_vertex_angle(
        micro_segment_then_hard_turn,
        minimum_segment_length=0.01,
    ) < config.smoothing_angle

    finished = _finish_solid_fill_paths(
        allowed,
        allowed,
        [],
        [micro_segment_then_hard_turn],
        LineString(),
        config,
        centerline_regions=(0.1, allowed, allowed, 1.7, LineString()),
    )

    assert len(finished) == 1
    assert LineString(finished[0]).is_simple
    assert _minimum_nondegenerate_vertex_angle(
        finished[0],
        minimum_segment_length=0.01,
    ) >= config.smoothing_angle - 1e-3


def test_solid_fill_finisher_never_turns_simple_path_into_self_intersection():
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=100.0,
        infill_overlap=10.0,
        smoothing_angle=150.0,
        smoothing_radius_factor=0.35,
    )
    allowed = Polygon([(-100, -100), (100, -100), (100, 100), (-100, 100)])
    simple_close_turns = np.asarray(
        [
            [1.0110112, -1.6085931],
            [0.9906273, 0.0302924],
            [1.0160627, -1.0956739],
            [1.7129788, -1.0671705],
            [2.0806093, -0.4887412],
            [1.6354822, -1.0645614],
        ],
        dtype=np.float32,
    )
    assert LineString(simple_close_turns).is_simple

    finished = _finish_solid_fill_paths(
        allowed,
        allowed,
        [],
        [simple_close_turns],
        LineString(),
        config,
        centerline_regions=(0.1, allowed, allowed, 1.7, LineString()),
    )

    assert finished
    assert len(finished) == 4
    assert all(LineString(path).is_simple for path in finished)
    assert all(
        np.allclose(previous[-1], following[0])
        for previous, following in zip(finished, finished[1:])
    )
    assert all(
        not slicer_module._effective_path_corner_candidates(
            path,
            config.line_width * 0.005,
            config.smoothing_angle,
            config.tolerance,
        )
        for path in finished
    )


def test_solid_fill_finisher_rolls_back_non_simple_rounding_candidate(monkeypatch):
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=100.0,
        infill_overlap=10.0,
        smoothing_angle=150.0,
        smoothing_radius_factor=0.35,
    )
    allowed = Polygon([(-10, -10), (10, -10), (10, 10), (-10, 10)])
    baseline = np.asarray([[0, 0], [6, 0]], dtype=np.float32)
    crossing_candidate = np.asarray(
        [[0, 0], [4, 4], [0, 4], [4, 0], [6, 0]],
        dtype=np.float32,
    )
    assert LineString(baseline).is_simple
    assert not LineString(crossing_candidate).is_simple

    with pytest.raises(ValueError, match="simple baseline"):
        slicer_module._split_path_at_unresolved_effective_corners(
            crossing_candidate,
            config.line_width * 0.005,
            config.smoothing_angle,
            config.tolerance,
        )

    monkeypatch.setattr(
        slicer_module,
        "_smooth_resin_infill_paths",
        lambda *args, **kwargs: [crossing_candidate],
    )
    finished = _finish_solid_fill_paths(
        allowed,
        allowed,
        [],
        [baseline],
        LineString(),
        config,
        centerline_regions=(0.1, allowed, allowed, 1.7, LineString()),
    )

    assert len(finished) == 1
    assert LineString(finished[0]).is_simple
    assert np.array_equal(finished[0], baseline)


def test_corner_smoothing_radius_is_a_physical_radius_not_a_tangent_cut():
    previous = np.asarray([0.0, 0.0], dtype=np.float32)
    corner = np.asarray([1.0, 0.0], dtype=np.float32)
    following = np.asarray(
        [1.0 - math.sqrt(0.5), math.sqrt(0.5)],
        dtype=np.float32,
    )

    rounded = slicer_module._rounded_corner_points(
        previous,
        corner,
        following,
        max_radius=0.2,
        angle_threshold_degrees=150.0,
        tolerance=1e-5,
        cut_fraction=0.8,
    )

    assert rounded is not None
    first, middle, last = (
        np.asarray(rounded[index], dtype=np.float64)
        for index in (0, len(rounded) // 2, -1)
    )
    first_leg = middle - first
    second_leg = last - middle
    twice_area = abs(
        first_leg[0] * second_leg[1] - first_leg[1] * second_leg[0]
    )
    circumradius = (
        np.linalg.norm(first_leg)
        * np.linalg.norm(second_leg)
        * np.linalg.norm(last - first)
        / (2.0 * twice_area)
    )

    assert circumradius == pytest.approx(0.2, abs=1e-4)


def test_wall_seam_dogleg_trial_rejects_hook_but_keeps_tangent_turn():
    tolerance = 5e-4
    safe_geometry = Polygon([(80, 60), (100, 60), (100, 75), (80, 75)])
    assembled = np.asarray(
        [
            [90.5353, 66.0996],
            [90.6594, 64.5229],
            [86.7823, 68.4],
            [89.3279, 68.4],
        ],
        dtype=np.float32,
    )
    hooked_bridge = np.asarray(
        [
            [89.3279, 68.4],
            [90.3834, 68.0302],
            [90.5660, 65.7093],
            [89.6064, 68.1215],
        ],
        dtype=np.float32,
    )
    trimmed = np.asarray(
        [[89.6064, 68.1215], [90.4420, 67.2860]],
        dtype=np.float32,
    )
    hooked_candidate = np.vstack(
        (assembled, hooked_bridge[1:], trimmed[1:])
    ).astype(np.float32)

    assert not slicer_module._solid_wall_seam_dogleg_is_finishable(
        hooked_candidate,
        hooked_bridge,
        assembled,
        trimmed,
        safe_geometry,
        2.0,
        150.0,
        0.3,
        tolerance,
    )

    angles = np.linspace(-np.pi * 0.5, np.pi * 0.5, 19)
    tangent_bridge = np.column_stack(
        (np.cos(angles), 1.0 + np.sin(angles))
    ).astype(np.float32)
    tangent_assembled = np.asarray([[-2, 0], [0, 0]], dtype=np.float32)
    tangent_trimmed = np.asarray([[0, 2], [-2, 2]], dtype=np.float32)
    tangent_candidate = np.vstack(
        (tangent_assembled, tangent_bridge[1:], tangent_trimmed[1:])
    ).astype(np.float32)

    assert slicer_module._solid_wall_seam_dogleg_is_finishable(
        tangent_candidate,
        tangent_bridge,
        tangent_assembled,
        tangent_trimmed,
        Polygon([(-3, -1), (2, -1), (2, 3), (-3, 3)]),
        2.0,
        150.0,
        0.3,
        tolerance,
    )


def test_residual_correction_rejects_stroke_stacked_inside_two_mm_bead():
    existing = LineString([(0.0, 0.0), (10.0, 0.0)]).buffer(
        1.0,
        cap_style="round",
        join_style="round",
    )
    stacked = LineString([(0.0, 0.1), (10.0, 0.1)]).buffer(
        1.0,
        cap_style="round",
        join_style="round",
    )
    uncovered = LineString([(0.0, 3.0), (10.0, 3.0)]).buffer(
        1.0,
        cap_style="round",
        join_style="round",
    )

    assert not _residual_correction_has_sufficient_novel_area(
        stacked.difference(existing).area,
        added_length=10.0,
        line_width=2.0,
    )
    assert _residual_correction_has_sufficient_novel_area(
        uncovered.difference(existing).area,
        added_length=10.0,
        line_width=2.0,
    )


@pytest.mark.parametrize(
    ("height", "expected_scan_ys"),
    [
        (9.3, [4.65]),
        (10.0, [5.0]),
        (11.2, [4.7, 6.5]),
        (14.5, [4.6, 6.36667, 8.13333, 9.9]),
        (14.59, [4.6, 6.39667, 8.19333, 9.99]),
    ],
)
def test_prusa_full_density_narrow_corridors_do_not_add_boundary_pileup(
    height: float,
    expected_scan_ys: list[float],
):
    contour = np.asarray(
        [[0, 0], [40, 0], [40, height], [0, height], [0, 0]],
        dtype=np.float32,
    )
    solid = Polygon(contour[:-1])
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=100.0,
        infill_overlap=10.0,
    )

    paths, roles = _build_resin_paths(
        [contour],
        config,
        layer_index=1,
        forced_zigzag_angle=0.0,
    )
    infill = _paths_with_role(paths, roles, "infill")
    scan_ys = _horizontal_scan_ys_for_paths(infill)

    assert len(infill) == 1
    assert np.allclose(scan_ys, expected_scan_ys, atol=0.02)
    assert _infill_extra_dose_ratio(infill, solid, config.line_width) <= 1e-6
    if len(scan_ys) > 1:
        expected_pitch = config.line_width * (1.0 - config.infill_overlap / 100.0)
        adjustment = _solid_spacing_adjustment_limit(
            expected_pitch,
            config.line_width,
        )
        assert min(np.diff(scan_ys)) >= expected_pitch - adjustment - 0.02


@pytest.mark.parametrize("angle", [0.0, 90.0])
def test_prusa_full_density_anchor_levels_snap_to_large_square_boundary(angle: float):
    contour = _square_contour(100.0)
    solid = Polygon(contour[:-1])
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=100.0,
        infill_overlap=10.0,
    )

    paths, roles = _build_resin_paths(
        [contour],
        config,
        layer_index=1,
        forced_zigzag_angle=angle,
    )
    infill = _paths_with_role(paths, roles, "infill")
    deposited = _round_bead_union(paths, config.line_width)

    assert len(infill) == 1
    assert solid.intersection(deposited).area / solid.area >= 0.998
    expected_pitch = config.line_width * (1.0 - config.infill_overlap / 100.0)
    adjustment = _solid_spacing_adjustment_limit(expected_pitch, config.line_width)
    assert max(_hatch_scan_level_gaps(infill, angle, config.line_width)) <= (
        expected_pitch + adjustment + 0.02
    )


@pytest.mark.parametrize("overlap_percent", [0.0, 10.0, 25.0])
@pytest.mark.parametrize(
    "pattern",
    [
        "rectilinear",
        "aligned_rectilinear",
        "line",
        "grid",
        "triangles",
        "gyroid",
        "concentric",
        "zigzag",
    ],
)
def test_prusa_infill_keeps_two_mm_wall_clearance_after_overlap(
    pattern: str,
    overlap_percent: float,
):
    contour = _square_contour(40.0)
    line_width = 2.0
    config = SliceConfig(
        layer_height=1.0,
        line_width=line_width,
        infill_pattern=pattern,
        infill_density=100.0,
        infill_overlap=overlap_percent,
        smoothing_radius_factor=0.0,
    )

    paths, roles = _build_resin_paths([contour], config, layer_index=1)
    infill = _paths_with_role(paths, roles, "infill")
    last_walls = _paths_with_role(paths, roles, "inner_contour")
    expected_pitch = line_width * (1.0 - overlap_percent / 100.0)
    clearance = unary_union([LineString(path) for path in infill]).distance(
        unary_union([LineString(path) for path in last_walls])
    )

    assert clearance >= expected_pitch - 0.02
    if pattern == "aligned_rectilinear":
        assert np.isclose(clearance, expected_pitch, atol=0.02)


@pytest.mark.parametrize("density", [50.0, 100.0])
@pytest.mark.parametrize(
    "pattern",
    ["aligned_rectilinear", "grid", "triangles", "gyroid", "concentric"],
)
def test_prusa_infill_material_length_budget_matches_density(pattern: str, density: float):
    contour = _square_contour(100.0)
    roi = Polygon([(20, 20), (80, 20), (80, 80), (20, 80)])
    line_width = 2.0
    overlap_percent = 10.0
    config = SliceConfig(
        layer_height=1.0,
        line_width=line_width,
        infill_pattern=pattern,
        infill_density=density,
        infill_overlap=overlap_percent,
        smoothing_radius_factor=0.0,
    )

    paths, roles = _build_resin_paths([contour], config, layer_index=1)
    infill = _paths_with_role(paths, roles, "infill")
    roi_length = sum(LineString(path).intersection(roi).length for path in infill)
    path_pitch = line_width * (1.0 - overlap_percent / 100.0)
    material_ratio = roi_length * path_pitch / roi.area

    assert material_ratio == pytest.approx(density / 100.0, abs=0.08)


@pytest.mark.parametrize(
    "pattern",
    [
        "rectilinear",
        "aligned_rectilinear",
        "line",
        "grid",
        "triangles",
        "gyroid",
        "concentric",
        "zigzag",
    ],
)
def test_prusa_infill_does_not_connect_separate_islands(pattern: str):
    left = _square_contour(24.0)
    right = left + np.asarray([36.0, 0.0], dtype=np.float32)
    islands = [Polygon(left[:-1]), Polygon(right[:-1])]

    paths, roles = _build_resin_paths(
        [left, right],
        SliceConfig(
            layer_height=1.0,
            line_width=2.0,
            infill_pattern=pattern,
            infill_density=100.0,
            smoothing_radius_factor=0.0,
        ),
        layer_index=1,
    )
    infill = _paths_with_role(paths, roles, "infill")
    owners: set[int] = set()
    for path in infill:
        line = LineString(path)
        containing_islands = [
            index for index, island in enumerate(islands) if island.buffer(1e-4).covers(line)
        ]
        assert len(containing_islands) == 1
        owners.add(containing_islands[0])

    assert owners == {0, 1}


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
            assert "PySLM kernel currently supports" in str(exc)
        else:
            raise AssertionError(f"expected PySLM to reject {pattern}")

    _validate_pyslm_config(
        SliceConfig(
            layer_height=0.5,
            slicing_kernel="pyslm",
            infill_pattern="isotropic",
        )
    )


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
        "机械臂空间复合材料增材制造系统切片器",
        "模型切片",
        "切片内核",
        ">Prusa<",
        ">PySLM<",
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
        "树脂填充路径",
        "各向同性填充",
        "三角形填充路径优化",
        "之字形填充路径优化",
    ):
        assert translated_label in html

    for untranslated_label in (
        "KUKA Slicer",
        "树脂切片",
        "原始内核（稳定）",
        "PySLM（实验）",
        "Slicing kernel",
        "Legacy (stable)",
        "PySLM (experimental)",
        "PySLM native settings",
        "Hatcher strategy",
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
        "raftOffsets",
        "curveMode",
        "curveAmplitude",
        "curvePeriod",
        "pathProgressControl",
        "pathProgressSlider",
        "pathProgressLabel",
        "showOuterContour",
        "showInnerContour",
        "showResinInfill",
        "showFiberPaths",
        "printSizeLabel",
        "previewSurface",
        "previewCanvas",
    ):
        assert f'id="{control_id}"' in html
    assert 'value="legacy" selected' in html
    assert 'value="pyslm"' in html
    assert 'id="trianglePathOptimization" type="checkbox" checked' in html
    assert 'value="isotropic"' in html
    for hatcher in ("basic", "stripe", "island", "basic_island"):
        assert f'value="{hatcher}"' in html
    for removed_raft_control in (
        "raftLayerCount",
        "raftTopGap",
        "raftLayerHeights",
        "raftInfillDensities",
        "raftInfillPatterns",
    ):
        assert f'id="{removed_raft_control}"' not in html
    assert "bottomCapAngle" not in html
    assert "topCapAngle" not in html
    assert 'id="resinPathSlider"' not in html
    assert 'id="fiberPathSlider"' not in html


def test_ui_preview_supports_filtered_ordered_progress_pan_zoom_and_rulers():
    html = _index_html()

    for label in (
        "所选路径进度",
        "外轮廓",
        "内轮廓",
        "树脂填充",
        "纤维路径",
        "打印范围",
        "X (mm)",
        "Y (mm)",
    ):
        assert label in html
    for interaction in (
        "selectedPrintEntries",
        "drawMeasurementGrid",
        "niceGridStep",
        "addEventListener('wheel'",
        "addEventListener('pointerdown'",
        "addEventListener('pointermove'",
        "addEventListener('contextmenu'",
        "event.button",
        "viewerState.centerX",
        "viewerState.centerY",
    ):
        assert interaction in html
    assert "const isContour" not in html
    assert "pathIndex >= visiblePaths" not in html


def test_preview_simplification_keeps_contour_corners():
    points = [[0.0, 0.0, 0.5], [0.0, 20.0, 0.5]]
    points.extend([[float(x), 20.0, 0.5] for x in range(1, 202)])
    points.append([0.0, 0.0, 0.5])

    simplified = _simplify_preview_path(points, max_points=2000)

    assert [0.0, 20.0, 0.5] in simplified


def test_raft_infill_paths_are_clipped_to_geometry():
    geometry = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
    config = SliceConfig(layer_height=1.0, line_width=1.0, infill_pattern="aligned_rectilinear")

    paths = _raft_zigzag_infill_paths(geometry, config, infill_density=100.0)

    assert len(paths) == 1
    assert paths[0].shape[0] > 2
    assert abs(_repeated_centerline_length(paths)) <= 1e-4
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


@pytest.mark.parametrize(("obstacle_distance", "expected_clear"), [(1.7, False), (1.8, True)])
def test_resin_connector_rejects_two_mm_near_miss(
    obstacle_distance: float,
    expected_clear: bool,
):
    connector = LineString([(0.0, 0.0), (2.0, 0.0)])
    path_lines = {
        0: LineString([(-1.0, 0.0), (0.0, 0.0)]),
        1: LineString([(2.0, 0.0), (3.0, 0.0)]),
        2: LineString([(0.5, obstacle_distance), (1.5, obstacle_distance)]),
    }

    is_clear = _centerline_connector_is_clear(
        connector,
        path_lines,
        {0: Point(0.0, 0.0), 1: Point(2.0, 0.0)},
        accepted=[],
        tolerance=1e-5,
        minimum_clearance=1.8,
    )

    assert is_clear is expected_clear


def test_smoothing_keeps_an_unsafe_fillet_as_one_continuous_path():
    path = np.asarray([[0, 0], [5, 0], [5, 5]], dtype=np.float32)
    narrow_safe_corridor = LineString(path).buffer(0.02, join_style="mitre")

    smoothed = _smooth_path_corners_into_paths(
        path,
        max_radius=2.0,
        angle_threshold_degrees=150.0,
        tolerance=1e-5,
        safe_geometry=narrow_safe_corridor,
        cut_fraction=0.35,
    )

    assert len(smoothed) == 1
    assert np.allclose(smoothed[0][0], path[0])
    assert np.allclose(smoothed[0][-1], path[-1])


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
    assert np.isclose(z_shift, 1.0)
    assert [group.layer_index for group in resin_groups[:4]] == [0, 1, 2, 3]
    assert np.isclose(resin_groups[0].paths[0][0, 2], 0.5)
    assert np.isclose(resin_groups[1].paths[0][0, 2], 1.0)
    assert np.isclose(resin_groups[2].paths[0][0, 2], 6.0)
    assert job.meta["raft"]["layer_count"] == 2
    assert job.meta["raft"]["top_gap"] == 0.0


def test_add_raft_rejects_non_two_layer_raft():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(layer_height=5.0, line_width=2.0, infill_pattern="concentric")
    job = slice_mesh_to_job(mesh, config)

    try:
        add_raft_to_job(
            job,
            mesh,
            config,
            [RaftLayerConfig(outward_offset=2.0)],
        )
    except ValueError as exc:
        assert "fixed at 2" in str(exc)
    else:
        raise AssertionError("expected non-two-layer raft to fail")


def test_two_raft_layers_use_fixed_zigzag_angles():
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
    assert _has_infill_direction(bottom_infill, 90.0)
    assert _has_infill_direction(top_infill, 135.0)
    assert job.meta["raft"]["layers"][0]["infill_density"] == 100.0
    assert job.meta["raft"]["layers"][1]["infill_density"] == 100.0
    assert job.meta["raft"]["fixed_patterns"][0]["angle_degrees"] == 90.0
    assert job.meta["raft"]["fixed_patterns"][1]["angle_degrees"] == -45.0


def test_raft_infill_density_follows_part_infill_density():
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        infill_pattern="zigzag",
        infill_density=40.0,
    )
    job = slice_mesh_to_job(mesh, config)

    add_raft_to_job(
        job,
        mesh,
        config,
        [
            RaftLayerConfig(outward_offset=2.0, infill_density=100.0),
            RaftLayerConfig(outward_offset=1.0, infill_density=80.0),
        ],
    )

    assert job.meta["raft"]["layers"][0]["infill_density"] == 40.0
    assert job.meta["raft"]["layers"][1]["infill_density"] == 40.0


def test_raft_outward_offset_preserves_part_holes():
    outer = [(0, 0), (40, 0), (40, 30), (0, 30)]
    hole = list(Point(20, 15).buffer(4.0, resolution=32).exterior.coords)
    footprint = Polygon(outer, holes=[hole])
    reserved_voids = _raft_reserved_void_geometry(footprint, tolerance=1e-5)

    raft_geometry = _raft_geometry_for_layer(
        footprint,
        reserved_voids,
        outward_offset=5.0,
        tolerance=1e-5,
    )
    hole_polygon = Polygon(hole)

    assert not raft_geometry.intersects(hole_polygon.buffer(-0.05))


def test_raft_boundary_paths_respect_nozzle_width_at_true_voids():
    hole = list(Point(20, 15).buffer(4.0, resolution=32).exterior.coords)
    geometry = Polygon([(0, 0), (40, 0), (40, 30), (0, 30)], holes=[hole])

    paths, roles = _perimeter_paths_from_geometry(
        geometry,
        line_width=2.0,
        path_spacing=1.8,
        perimeter_count=2,
        tolerance=1e-5,
    )
    inner_paths = _paths_with_role(paths, roles, "inner_contour")
    hole_boundary = LineString(hole)

    assert len(inner_paths) >= 2
    assert np.isclose(min(LineString(path).distance(hole_boundary) for path in paths), 1.0, atol=0.05)


def test_raft_infill_does_not_cross_true_void_boundaries():
    hole = list(Point(20, 15).buffer(4.0, resolution=32).exterior.coords)
    geometry = Polygon([(0, 0), (40, 0), (40, 30), (0, 30)], holes=[hole])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    hole_polygon = Polygon(hole)
    safe_geometry = _resin_infill_surface_geometry(geometry, config)

    paths = _raft_zigzag_infill_paths(
        safe_geometry,
        config,
        infill_density=100.0,
        angle_degrees=45.0,
    )

    assert paths
    for path in paths:
        line = LineString(path)
        assert (
            line.difference(
                safe_geometry.buffer(config.tolerance * 10.0, join_style="round")
            ).length
            <= 1e-4
        )
        assert not line.intersects(hole_polygon.buffer(config.line_width * 0.5))


def test_raft_zigzag_full_density_covers_boundary_without_retrace():
    geometry = Polygon([(0, 0), (30, 0), (30, 20), (0, 20)])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    paths = _raft_zigzag_infill_paths(
        geometry.buffer(-config.line_width * 0.5, join_style="round"),
        config,
        infill_density=100.0,
        angle_degrees=45.0,
    )

    full_width_stroke = _round_bead_union(paths, config.line_width)

    assert 1 <= len(paths) <= 3
    assert all(LineString(path).is_simple for path in paths)
    assert full_width_stroke.intersection(geometry).area / geometry.area > 0.97
    assert _maximum_uncovered_void_diameter(
        geometry,
        full_width_stroke,
        edge_inset=0.3,
    ) <= 0.6
    assert abs(_repeated_centerline_length(paths)) <= 1e-4


def test_raft_zigzag_follows_hole_boundary_with_few_full_width_strokes():
    hole_polygon = Point(30, 20).buffer(7.0, resolution=32)
    geometry = Polygon(
        [(0, 0), (60, 0), (60, 40), (0, 40)],
        holes=[list(hole_polygon.exterior.coords)],
    )
    config = SliceConfig(
        layer_height=1.0,
        line_width=2.0,
        perimeter_count=2,
        infill_overlap=10.0,
    )
    safe_geometry = _resin_infill_surface_geometry(geometry, config)

    paths = _raft_zigzag_infill_paths(
        safe_geometry,
        config,
        infill_density=100.0,
        angle_degrees=0.0,
    )
    full_width_stroke = unary_union(
        [
            LineString(path).buffer(
                config.line_width * 0.5,
                cap_style="round",
                join_style="round",
            )
            for path in paths
        ]
    )

    assert 1 <= len(paths) <= 3
    assert all(LineString(path).is_simple for path in paths)
    assert abs(_repeated_centerline_length(paths)) <= 1e-4
    assert all(safe_geometry.buffer(1e-5).covers(LineString(path)) for path in paths)
    assert geometry.buffer(1e-5).covers(full_width_stroke)
    assert not full_width_stroke.intersects(hole_polygon)
    assert full_width_stroke.intersection(safe_geometry).area / safe_geometry.area > 0.995


def test_raft_zigzag_full_density_covers_printable_surface():
    hole = list(Point(28, 16).buffer(3.0, resolution=32).exterior.coords)
    geometry = Polygon([(0, 0), (60, 0), (60, 40), (0, 40)], holes=[hole])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)
    safe_geometry = _resin_infill_surface_geometry(geometry, config)

    paths = _raft_zigzag_infill_paths(
        safe_geometry,
        config,
        infill_density=100.0,
        angle_degrees=45.0,
    )
    stroke_area = unary_union(
        [
            LineString(path[:, :2]).buffer(
                config.line_width * 0.5,
                cap_style="round",
                join_style="round",
            )
            for path in paths
        ]
    )

    assert paths
    assert stroke_area.intersection(safe_geometry).area / safe_geometry.area > 0.995


def test_raft_voids_include_openings_from_later_part_sections():
    part_projection = Polygon(
        [
            (0, 0),
            (50, 0),
            (50, 30),
            (34, 30),
            (34, 18),
            (16, 18),
            (16, 30),
            (0, 30),
        ]
    )
    opening = Polygon([(16, 18), (34, 18), (34, 35), (16, 35)])

    reserved_voids = _raft_reserved_void_geometry(part_projection, tolerance=1e-5)
    raft_geometry = _raft_geometry_for_layer(
        part_projection,
        reserved_voids,
        outward_offset=5.0,
        tolerance=1e-5,
    )

    assert reserved_voids.covers(
        Polygon([(16, 18), (34, 18), (34, 30), (16, 30)]).buffer(-0.05)
    )
    assert not raft_geometry.intersects(opening.buffer(-0.05))


def test_raft_opening_perimeters_keep_configured_spacing_through_outward_band():
    part_projection = Polygon(
        [
            (0, 0),
            (40, 0),
            (40, 30),
            (0, 30),
            (0, 20),
            (20, 20),
            (20, 10),
            (0, 10),
        ]
    )
    reserved_voids = _raft_reserved_void_geometry(part_projection, tolerance=1e-5)
    raft_geometry = _raft_geometry_for_layer(
        part_projection,
        reserved_voids,
        outward_offset=5.0,
        tolerance=1e-5,
    )

    paths, roles = _perimeter_paths_from_geometry(
        raft_geometry,
        line_width=2.0,
        path_spacing=1.8,
        perimeter_count=2,
        tolerance=1e-5,
    )
    outer_contour = LineString(_paths_with_role(paths, roles, "outer_contour")[0])
    inner_contour = LineString(_paths_with_role(paths, roles, "inner_contour")[0])

    assert not raft_geometry.intersects(
        Polygon([(-4.9, 10.05), (19.9, 10.05), (19.9, 19.95), (-4.9, 19.95)])
    )
    assert np.isclose(
        outer_contour.distance(Point(-2.0, 7.2)),
        1.8,
        atol=0.05,
    )
    assert inner_contour.distance(Point(-2.0, 7.2)) <= 0.05


def test_raft_reserved_voids_filter_degenerate_projection_fragments():
    part_projection = Polygon(
        [(0, 0), (30, 0), (30, 30), (0, 30)],
        holes=[
            [(10, 10), (10.2, 10), (10.2, 10.001), (10, 10.001)],
            [(14, 14), (18, 14), (18, 18), (14, 18)],
        ],
    )

    reserved_voids = _raft_reserved_void_geometry(part_projection, tolerance=1e-5)

    assert reserved_voids.area > 15.0
    assert reserved_voids.area < 17.0


def test_raft_lattice_density_changes_spacing():
    geometry = Polygon([(0, 0), (40, 0), (40, 40), (0, 40)])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    dense = _raft_lattice_infill_paths(geometry, config, infill_density=100.0)
    sparse = _raft_lattice_infill_paths(geometry, config, infill_density=50.0)

    assert dense
    assert sparse
    assert _infill_total_length(dense) > _infill_total_length(sparse)


def test_raft_zigzag_infill_connectors_stay_inside_printable_geometry():
    outer = [(0, 0), (42, 0), (42, 20), (0, 20)]
    hole = list(Point(30, 10).buffer(3.0, quad_segs=32).exterior.coords)
    geometry = Polygon(outer, holes=[hole])
    config = SliceConfig(layer_height=1.0, line_width=2.0, infill_overlap=10.0)

    paths = _raft_zigzag_infill_paths(geometry, config, infill_density=100.0)

    assert paths
    safe_geometry = geometry.buffer(1e-4, join_style="round")
    for path in paths:
        for start, end in zip(path[:-1], path[1:]):
            segment = LineString([tuple(start[:2]), tuple(end[:2])])
            assert safe_geometry.covers(segment)


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


def _square_contour(size: float, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    x, y = origin
    return np.asarray(
        [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]],
        dtype=np.float32,
    )


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


def _round_bead_union(paths: list[np.ndarray], line_width: float):
    return unary_union(
        [
            LineString(path[:, :2]).buffer(
                line_width * 0.5,
                cap_style="round",
                join_style="round",
                quad_segs=32,
            )
            for path in paths
            if path.shape[0] >= 2
        ]
    )


def _maximum_uncovered_void_diameter(
    solid: Polygon,
    deposited,
    *,
    edge_inset: float,
) -> float:
    region = solid.buffer(-edge_inset, join_style="mitre")
    uncovered = region.difference(deposited)
    if uncovered.is_empty:
        return 0.0
    return float(maximum_inscribed_circle(uncovered, tolerance=0.01).length * 2.0)


def _infill_extra_dose_ratio(
    paths: list[np.ndarray],
    solid: Polygon,
    line_width: float,
) -> float:
    region = solid.buffer(-0.3, join_style="mitre")
    beads = [
        LineString(path[:, :2])
        .buffer(
            line_width * 0.5,
            cap_style="round",
            join_style="round",
            quad_segs=32,
        )
        .intersection(region)
        for path in paths
        if path.shape[0] >= 2
    ]
    if not beads:
        return 0.0
    return (sum(bead.area for bead in beads) - unary_union(beads).area) / solid.area


def _segment_bead_overlap_ratio(
    paths: list[np.ndarray],
    line_width: float,
) -> float:
    """Measure overlap per printed segment, including within one continuous path."""

    segment_beads = [
        LineString([start[:2], end[:2]]).buffer(
            line_width * 0.5,
            cap_style="flat",
            join_style="mitre",
            quad_segs=8,
        )
        for path in paths
        for start, end in zip(path[:-1], path[1:])
        if float(np.linalg.norm(end[:2] - start[:2])) > 1e-6
    ]
    total_area = sum(bead.area for bead in segment_beads)
    if total_area <= 0:
        return 0.0
    return (total_area - unary_union(segment_beads).area) / total_area


def _hatch_scan_level_gaps(
    paths: list[np.ndarray],
    angle_degrees: float,
    line_width: float,
) -> list[float]:
    radians = math.radians(angle_degrees)
    direction = np.asarray([math.cos(radians), math.sin(radians)], dtype=np.float64)
    normal = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    minimum_segment_length = line_width * 1.5
    minimum_parallel_cosine = math.cos(math.radians(1.0))
    levels: list[float] = []
    for path in paths:
        for start, end in zip(path[:-1, :2], path[1:, :2]):
            delta = np.asarray(end - start, dtype=np.float64)
            length = float(np.linalg.norm(delta))
            if length < minimum_segment_length:
                continue
            if abs(float(np.dot(delta / length, direction))) < minimum_parallel_cosine:
                continue
            midpoint = (np.asarray(start, dtype=np.float64) + end) * 0.5
            levels.append(float(np.dot(midpoint, normal)))

    clustered: list[float] = []
    for level in sorted(levels):
        if not clustered or level - clustered[-1] > 0.02:
            clustered.append(level)
    return [
        upper - lower
        for lower, upper in zip(clustered[:-1], clustered[1:])
    ]


def _repeated_centerline_length(paths: list[np.ndarray]) -> float:
    lines = [LineString(path[:, :2]) for path in paths if path.shape[0] >= 2]
    return sum(line.length for line in lines) - unary_union(lines).length


def _minimum_nondegenerate_vertex_angle(
    path: np.ndarray,
    minimum_segment_length: float,
) -> float:
    """Return the sharpest continuous turn, looking across numerical micro-segments."""

    points = np.asarray(path[:, :2], dtype=np.float64)
    angles: list[float] = []
    for index in range(1, points.shape[0] - 1):
        previous_index = index - 1
        while (
            previous_index >= 0
            and float(np.linalg.norm(points[index] - points[previous_index]))
            < minimum_segment_length
        ):
            previous_index -= 1

        next_index = index + 1
        while (
            next_index < points.shape[0]
            and float(np.linalg.norm(points[next_index] - points[index]))
            < minimum_segment_length
        ):
            next_index += 1

        if previous_index < 0 or next_index >= points.shape[0]:
            continue
        incoming = points[previous_index] - points[index]
        outgoing = points[next_index] - points[index]
        incoming_length = float(np.linalg.norm(incoming))
        outgoing_length = float(np.linalg.norm(outgoing))
        if (
            incoming_length < minimum_segment_length
            or outgoing_length < minimum_segment_length
        ):
            continue
        cosine = float(
            np.clip(
                np.dot(incoming, outgoing) / (incoming_length * outgoing_length),
                -1.0,
                1.0,
            )
        )
        angles.append(math.degrees(math.acos(cosine)))
    return min(angles, default=180.0)


def _horizontal_scan_ys_for_paths(paths: list[np.ndarray]) -> list[float]:
    values = {
        round(y, 5)
        for path in paths
        for y in _horizontal_scan_ys(path)
    }
    return sorted(values)


def _diagonal_segment_sign(paths: list[np.ndarray]) -> float:
    products = [
        float(delta[0] * delta[1])
        for path in paths
        for delta in np.diff(path[:, :2], axis=0)
        if abs(float(delta[0])) > 1e-4 and abs(float(delta[1])) > 1e-4
    ]
    assert products
    return float(np.median(products))


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
