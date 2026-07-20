import math

import numpy as np
import pytest
from shapely.geometry import LineString
from shapely.ops import unary_union

pytest.importorskip("pyslm")
pytest.importorskip("trimesh")

from kuka_slicer.slicer import PySLMConfig, SliceConfig, slice_mesh_to_job
from kuka_slicer.stl_io import Mesh


def _cube_triangles(size: float) -> np.ndarray:
    s = float(size)
    vertices = np.asarray(
        [
            [0, 0, 0],
            [s, 0, 0],
            [s, s, 0],
            [0, s, 0],
            [0, 0, s],
            [s, 0, s],
            [s, s, s],
            [0, s, s],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return vertices[faces]


@pytest.mark.parametrize("hatcher", ("basic", "stripe", "island", "basic_island"))
def test_native_pyslm_hatchers_preserve_project_output_contract(hatcher: str):
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        slicing_kernel="pyslm",
        infill_pattern="rectilinear",
        pyslm=PySLMConfig(hatcher=hatcher, hatch_sort="alternate"),  # type: ignore[arg-type]
    )

    job = slice_mesh_to_job(mesh, config)

    assert job.material_paths
    assert job.meta["slicing"]["slicing_kernel"] == "pyslm"
    assert job.meta["slicing"]["pyslm"]["hatcher"] == hatcher
    assert job.meta["slicing"]["pyslm"]["native_patterns"]
    assert set(job.meta["slicing"]["pyslm"]["project_patterns"]) >= {
        "zigzag_horizontal",
        "zigzag_vertical",
        "zigzag_plus45",
        "zigzag_minus45",
    }
    assert all(path.shape[1] == 3 for group in job.material_paths for path in group.paths)


@pytest.mark.parametrize("overlap_percent", (0.0, 10.0, 25.0))
def test_native_pyslm_hatch_offset_uses_default_two_percent_contour_seam(
    overlap_percent: float,
):
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        slicing_kernel="pyslm",
        infill_pattern="rectilinear",
        infill_overlap=overlap_percent,
    )

    job = slice_mesh_to_job(mesh, config)
    group = next(group for group in job.material_paths if group.layer_index == 1)
    roles = job.meta["path_roles"]["R"]["1"]
    inner_contours = [
        LineString(path[:, :2])
        for path, role in zip(group.paths, roles)
        if role == "inner_contour"
    ]
    infill = unary_union(
        [
            LineString(path[:, :2])
            for path, role in zip(group.paths, roles)
            if role == "infill"
        ]
    )

    expected_pitch = config.line_width * 0.98 + max(
        config.tolerance * 32.0,
        config.line_width * 2e-5,
        2e-6,
    )
    assert inner_contours
    assert not infill.is_empty
    assert min(contour.distance(infill) for contour in inner_contours) == pytest.approx(
        expected_pitch,
        abs=2e-3,
    )


def test_native_pyslm_contour_override_does_not_change_fixed_cap_clearance():
    job = slice_mesh_to_job(
        Mesh(_cube_triangles(size=20.0)),
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            slicing_kernel="pyslm",
            infill_pattern="rectilinear",
            infill_overlap=10.0,
            pyslm=PySLMConfig(contour_offset=3.0),
        ),
    )

    for group in job.material_paths:
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        inner_contours = [
            LineString(path[:, :2])
            for path, role in zip(group.paths, roles)
            if role == "inner_contour"
        ]
        infill = unary_union(
            [
                LineString(path[:, :2])
                for path, role in zip(group.paths, roles)
                if role == "infill"
            ]
        )
        assert min(contour.distance(infill) for contour in inner_contours) == pytest.approx(
            1.96032,
            abs=2e-3,
        )


def test_native_pyslm_hatch_distance_override_does_not_thin_fixed_caps():
    mesh = Mesh(_cube_triangles(size=20.0))
    base_config = dict(
        layer_height=5.0,
        line_width=2.0,
        slicing_kernel="pyslm",
        infill_pattern="rectilinear",
    )
    default_job = slice_mesh_to_job(mesh, SliceConfig(**base_config))
    override_job = slice_mesh_to_job(
        mesh,
        SliceConfig(**base_config, pyslm=PySLMConfig(hatch_distance=5.0)),
    )

    for layer_index in (0, len(default_job.material_paths) - 1):
        default_group = default_job.material_paths[layer_index]
        override_group = override_job.material_paths[layer_index]
        default_roles = default_job.meta["path_roles"]["R"][str(layer_index)]
        override_roles = override_job.meta["path_roles"]["R"][str(layer_index)]
        default_infill = [
            path for path, role in zip(default_group.paths, default_roles) if role == "infill"
        ]
        override_infill = [
            path for path, role in zip(override_group.paths, override_roles) if role == "infill"
        ]

        assert len(override_infill) == len(default_infill)
        for default_path, override_path in zip(default_infill, override_infill):
            np.testing.assert_allclose(override_path, default_path)


def test_pyslm_bound_simplification_mode_is_supported():
    job = slice_mesh_to_job(
        Mesh(_cube_triangles(size=20.0)),
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            slicing_kernel="pyslm",
            infill_pattern="rectilinear",
            pyslm=PySLMConfig(
                simplification_factor=0.001,
                simplification_mode="bound",
            ),
        ),
    )

    assert job.material_paths


def test_pyslm_line_simplification_mode_is_rejected_early():
    with pytest.raises(ValueError, match="absolute or bound"):
        PySLMConfig(simplification_mode="line")  # type: ignore[arg-type]


def test_project_zigzag_rejects_contour_geometry_overrides():
    with pytest.raises(ValueError, match="do not accept contour geometry overrides"):
        slice_mesh_to_job(
            Mesh(_cube_triangles(size=20.0)),
            SliceConfig(
                layer_height=5.0,
                line_width=2.0,
                slicing_kernel="pyslm",
                infill_pattern="zigzag",
                pyslm=PySLMConfig(contour_offset=3.0),
            ),
        )


def test_project_zigzag_respects_infill_first_scan_order():
    job = slice_mesh_to_job(
        Mesh(_cube_triangles(size=20.0)),
        SliceConfig(
            layer_height=5.0,
            line_width=2.0,
            slicing_kernel="pyslm",
            infill_pattern="zigzag",
            pyslm=PySLMConfig(scan_contour_first=False),
        ),
    )

    for group in job.material_paths:
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        assert roles[0] == "infill"
        assert all(role != "infill" for role in roles[1:])


@pytest.mark.parametrize("hatcher", ("basic", "stripe", "island", "basic_island"))
def test_pyslm_zigzag_uses_project_boundary_following_chains(hatcher: str):
    mesh = Mesh(_cube_triangles(size=20.0))
    config = SliceConfig(
        layer_height=5.0,
        line_width=2.0,
        slicing_kernel="pyslm",
        infill_pattern="zigzag",
        pyslm=PySLMConfig(hatcher=hatcher),  # type: ignore[arg-type]
    )

    job = slice_mesh_to_job(mesh, config)

    for group in job.material_paths:
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        infill = [path for path, role in zip(group.paths, roles) if role == "infill"]
        assert len(infill) == 1
        assert LineString(infill[0][:, :2]).is_simple


def test_pyslm_isotropic_infill_keeps_all_layers_and_fixed_angles():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    mesh = Mesh(triangles)
    config = SliceConfig(
        layer_height=0.5,
        line_width=2.0,
        slicing_kernel="pyslm",
        infill_pattern="isotropic",
        infill_density=60.0,
    )

    job = slice_mesh_to_job(mesh, config)

    expected_angles = [0.0, 45.0, 0.0, 135.0, 90.0, 45.0, 0.0, 135.0, 90.0, 45.0]
    assert [group.layer_index for group in job.material_paths] == list(range(10))
    for group, expected_angle in zip(job.material_paths, expected_angles):
        roles = job.meta["path_roles"]["R"][str(group.layer_index)]
        infill = [path for path, role in zip(group.paths, roles) if role == "infill"]
        assert infill
        assert _has_direction(infill, expected_angle)

    schedule = job.meta["slicing"]["isotropic_schedule"]
    assert schedule["repeat_count"] == 2
    assert schedule["repeat_angles_degrees"] == [45.0, 0.0, -45.0, 90.0]


def test_pyslm_isotropic_explicit_z_bounds_keep_all_layers():
    triangles = _cube_triangles(size=20.0)
    triangles[:, :, 2] *= 0.25
    job = slice_mesh_to_job(
        Mesh(triangles),
        SliceConfig(
            layer_height=0.5,
            line_width=2.0,
            slicing_kernel="pyslm",
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


def _has_direction(paths: list[np.ndarray], expected_angle: float) -> bool:
    for path in paths:
        for delta in np.diff(path[:, :2], axis=0):
            if np.linalg.norm(delta) <= 1e-6:
                continue
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0
            if abs(angle - expected_angle) < 2.0:
                return True
    return False
