import math

import numpy as np
import pytest
from shapely.geometry import LineString

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
    assert all(path.shape[1] == 3 for group in job.material_paths for path in group.paths)


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


def _has_direction(paths: list[np.ndarray], expected_angle: float) -> bool:
    for path in paths:
        for delta in np.diff(path[:, :2], axis=0):
            if np.linalg.norm(delta) <= 1e-6:
                continue
            angle = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 180.0
            if abs(angle - expected_angle) < 2.0:
                return True
    return False
